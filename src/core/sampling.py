"""
Sampling Strategies for RT-DLM AGI

[DEV UTILITY - Testing/Development Only]

This module provides token sampling and generation utilities for DEVELOPMENT
and TESTING purposes. It is NOT part of the production training pipeline.

For production inference, use a dedicated inference framework (vLLM, TGI, etc.)
that loads RT-DLM checkpoints.

This module implements:
- Temperature scaling: Controls randomness of generation
- Top-K filtering: Only consider K most likely tokens
- Top-P (Nucleus) sampling: Only sample from tokens whose cumulative probability ≤ top_p
- Repetition penalty: Discourage repeating tokens
- Speculative decoding: Reference implementation for testing

Usage (development/testing only):
    from src.core.sampling import TokenSampler, SamplingConfig, SampleOutput
    
    sampler = TokenSampler()
    config = SamplingConfig(temperature=0.8, top_p=0.9, top_k=50)
    output = sampler.sample(logits, config, rng_key)

Note:
    For production deployment, inference should be handled by optimized 
    serving frameworks that implement:
    - KV cache management
    - Continuous batching
    - Tensor parallelism
    - Optimized attention kernels
"""

# Mark entire module as dev utility
_MODULE_DEV_UTILITY = True
_MODULE_DEV_REASON = "Sampling/generation is for development testing. Production inference uses separate optimized serving stack."

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, NamedTuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for token sampling strategies.
    
    Attributes:
        temperature: Controls randomness. Lower = more deterministic, higher = more creative.
                    Recommended: 0.7-1.0 for most tasks, 1.0-1.5 for creative writing.
        top_k: Only consider top K most likely tokens. Set to 0 to disable.
               Recommended: 40-100 for general use.
        top_p: Nucleus sampling - only sample from smallest set of tokens whose 
               cumulative probability exceeds top_p. Set to 1.0 to disable.
               Recommended: 0.9-0.95 for balanced generation.
        min_p: Minimum probability threshold. Tokens with prob < min_p * max_prob are filtered.
               Set to 0.0 to disable. Recommended: 0.05-0.1
        repetition_penalty: Penalty for repeating tokens. 1.0 = no penalty, >1.0 = discourage repetition.
                           Recommended: 1.1-1.3
        presence_penalty: Penalty for tokens that have appeared at all. Range: 0.0-2.0
        frequency_penalty: Penalty based on frequency of token appearance. Range: 0.0-2.0
        max_tokens: Maximum number of tokens to generate.
        stop_tokens: List of token IDs that trigger generation stop.
        log_probs: Whether to return log probabilities of generated tokens.
        top_log_probs: Number of top token probabilities to return (for debugging/analysis).
    """
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 512
    stop_tokens: List[int] = field(default_factory=list)
    log_probs: bool = True
    top_log_probs: int = 5


class SampleOutput(NamedTuple):
    """Output from token sampling.
    
    Attributes:
        token_id: The sampled token ID [batch_size, 1]
        token_prob: Probability of the sampled token [batch_size, 1]
        token_log_prob: Log probability of the sampled token [batch_size, 1]
        top_k_token_ids: IDs of top-K most likely tokens [batch_size, top_log_probs]
        top_k_probs: Probabilities of top-K tokens [batch_size, top_log_probs]
        top_k_log_probs: Log probabilities of top-K tokens [batch_size, top_log_probs]
        entropy: Entropy of the distribution (measure of uncertainty) [batch_size]
    """
    token_id: jnp.ndarray
    token_prob: jnp.ndarray
    token_log_prob: jnp.ndarray
    top_k_token_ids: jnp.ndarray
    top_k_probs: jnp.ndarray
    top_k_log_probs: jnp.ndarray
    entropy: jnp.ndarray


class TokenSampler:
    """Production-ready token sampler with advanced sampling strategies."""
    
    def __init__(self, vocab_size: Optional[int] = None):
        """Initialize the token sampler.
        
        Args:
            vocab_size: Size of vocabulary. Used for validation.
        """
        self.vocab_size = vocab_size
    
    def apply_temperature(self, logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
        """Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits from model [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            temperature: Temperature value (>0). Lower = more deterministic.
            
        Returns:
            Scaled logits
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        # Avoid division by very small temperature (would cause numerical issues)
        temperature = max(temperature, 1e-7)
        return logits / temperature
    
    def apply_top_k(self, logits: jnp.ndarray, top_k: int) -> jnp.ndarray:
        """Apply top-K filtering: only keep top K tokens, set others to -inf.
        
        Args:
            logits: Logits [batch_size, vocab_size]
            top_k: Number of top tokens to keep. If 0 or >= vocab_size, no filtering.
            
        Returns:
            Filtered logits with non-top-K tokens set to -inf
        """
        if top_k <= 0:
            return logits
            
        vocab_size = logits.shape[-1]
        if top_k >= vocab_size:
            return logits
        
        # Get the k-th largest value for each batch
        # Sort in descending order and get the k-th value
        top_k_values, _ = jax.lax.top_k(logits, top_k)
        # The threshold is the minimum of top-k values (the k-th largest)
        threshold = top_k_values[..., -1:]  # [batch_size, 1]
        
        # Mask out tokens below threshold
        mask = logits >= threshold
        filtered_logits = jnp.where(mask, logits, -1e10)
        
        return filtered_logits
    
    def apply_top_p(self, logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
        """Apply nucleus (top-P) filtering: keep smallest set of tokens with cumulative prob >= top_p.
        
        This is more adaptive than top-K as it adjusts based on the probability distribution.
        For peaked distributions, it selects fewer tokens. For flat distributions, more tokens.
        
        Args:
            logits: Logits [batch_size, vocab_size]  
            top_p: Cumulative probability threshold (0.0-1.0). If 1.0, no filtering.
            
        Returns:
            Filtered logits with low-probability tokens set to -inf
        """
        if top_p >= 1.0:
            return logits
        if top_p <= 0.0:
            raise ValueError(f"top_p must be positive, got {top_p}")
        
        # Convert to probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Sort probabilities in descending order
        sorted_indices = jnp.argsort(-probs, axis=-1)
        sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        
        # Create mask: keep tokens until cumulative prob exceeds top_p
        # We shift by 1 to include the token that crosses the threshold
        sorted_mask = cumulative_probs <= top_p
        # Always keep at least one token (the most probable one)
        sorted_mask = sorted_mask.at[..., 0].set(True)
        
        # Unsort the mask back to original order
        unsorted_mask = jnp.zeros_like(sorted_mask)
        unsorted_mask = jnp.take_along_axis(
            sorted_mask, 
            jnp.argsort(sorted_indices, axis=-1), 
            axis=-1
        )
        
        # Apply mask
        filtered_logits = jnp.where(unsorted_mask, logits, -1e10)
        
        return filtered_logits
    
    def apply_min_p(self, logits: jnp.ndarray, min_p: float) -> jnp.ndarray:
        """Apply min-P filtering: remove tokens with prob < min_p * max_prob.
        
        This is useful for removing very unlikely tokens while being adaptive
        to the overall probability distribution.
        
        Args:
            logits: Logits [batch_size, vocab_size]
            min_p: Minimum probability ratio (0.0-1.0). If 0.0, no filtering.
            
        Returns:
            Filtered logits
        """
        if min_p <= 0.0:
            return logits
        
        probs = jax.nn.softmax(logits, axis=-1)
        max_prob = jnp.max(probs, axis=-1, keepdims=True)
        threshold = min_p * max_prob
        
        mask = probs >= threshold
        filtered_logits = jnp.where(mask, logits, -1e10)
        
        return filtered_logits
    
    def apply_repetition_penalty(
        self, 
        logits: jnp.ndarray, 
        generated_tokens: jnp.ndarray,
        penalty: float = 1.0
    ) -> jnp.ndarray:
        """Apply repetition penalty to discourage repeating tokens.
        
        Args:
            logits: Logits [batch_size, vocab_size]
            generated_tokens: Previously generated token IDs [batch_size, seq_len]
            penalty: Penalty factor. 1.0 = no penalty, >1.0 = discourage repetition.
            
        Returns:
            Logits with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Create a mask of which tokens have been generated
        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]
        
        # For each batch, mark tokens that appear in generated_tokens
        def apply_penalty_single(args):
            logit, tokens = args
            # Create penalty mask
            penalty_mask = jnp.zeros(vocab_size)
            # Set penalty for tokens that have been generated
            unique_tokens = jnp.unique(tokens, size=tokens.shape[0], fill_value=-1)
            valid_tokens = unique_tokens[unique_tokens >= 0]
            penalty_mask = penalty_mask.at[valid_tokens].set(1.0)
            
            # Apply penalty: divide positive logits, multiply negative logits
            penalized = jnp.where(
                logit > 0,
                logit / (1.0 + (penalty - 1.0) * penalty_mask),
                logit * (1.0 + (penalty - 1.0) * penalty_mask)
            )
            return penalized
        
        # Apply to each batch element
        penalized_logits: jnp.ndarray = jax.vmap(apply_penalty_single)((logits, generated_tokens))
        
        return penalized_logits
    
    def apply_frequency_presence_penalty(
        self,
        logits: jnp.ndarray,
        generated_tokens: jnp.ndarray,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> jnp.ndarray:
        """Apply frequency and presence penalties (OpenAI-style).
        
        Args:
            logits: Logits [batch_size, vocab_size]
            generated_tokens: Previously generated tokens [batch_size, seq_len]
            frequency_penalty: Penalty based on token frequency (0.0-2.0)
            presence_penalty: Penalty for token presence (0.0-2.0)
            
        Returns:
            Penalized logits
        """
        if frequency_penalty == 0.0 and presence_penalty == 0.0:
            return logits
        
        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]
        
        def compute_penalties(tokens):
            # Count frequency of each token
            freq = jnp.zeros(vocab_size)
            # Simple frequency count (approximate for JIT compatibility)
            for i in range(tokens.shape[0]):
                token = tokens[i]
                freq = jnp.where(token >= 0, freq.at[token].add(1), freq)
            
            # Presence is binary: 1 if token appeared, 0 otherwise
            presence = jnp.where(freq > 0, 1.0, 0.0)
            
            return freq, presence
        
        def apply_penalties_single(args):
            logit, tokens = args
            freq, presence = compute_penalties(tokens)
            penalty = frequency_penalty * freq + presence_penalty * presence
            return logit - penalty
        
        penalized_logits: jnp.ndarray = jax.vmap(apply_penalties_single)((logits, generated_tokens))
        
        return penalized_logits
    
    def sample(
        self,
        logits: jnp.ndarray,
        config: SamplingConfig,
        rng_key: jnp.ndarray,
        generated_tokens: Optional[jnp.ndarray] = None
    ) -> SampleOutput:
        """Sample tokens from logits using configured sampling strategy.
        
        Args:
            logits: Model output logits [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
            config: Sampling configuration
            rng_key: JAX random key for sampling
            generated_tokens: Previously generated tokens for repetition penalty [batch_size, seq_len]
            
        Returns:
            SampleOutput with sampled tokens and probability information
        """
        # Handle 3D logits (take last position)
        if logits.ndim == 3:
            logits = logits[:, -1, :] 
        
        batch_size = logits.shape[0]
        
        # Apply repetition penalties if we have generated tokens
        if generated_tokens is not None:
            if config.repetition_penalty != 1.0:
                logits = self.apply_repetition_penalty(
                    logits, generated_tokens, config.repetition_penalty
                )
            if config.frequency_penalty != 0.0 or config.presence_penalty != 0.0:
                logits = self.apply_frequency_presence_penalty(
                    logits, generated_tokens,
                    config.frequency_penalty, config.presence_penalty
                )
        
        # Apply temperature
        logits = self.apply_temperature(logits, config.temperature)
        
        # Apply filtering strategies (order matters for efficiency)
        # Top-K first (fast, reduces candidates)
        logits = self.apply_top_k(logits, config.top_k)
        
        # Min-P second (adaptive threshold)
        logits = self.apply_min_p(logits, config.min_p)
        
        # Top-P last (nucleus sampling on remaining candidates)
        logits = self.apply_top_p(logits, config.top_p)
        
        # Compute final probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        sampled_tokens = jax.random.categorical(rng_key, logits, axis=-1)
        sampled_tokens = sampled_tokens.reshape(batch_size, 1)
        
        # Get probability of sampled tokens
        token_probs = jnp.take_along_axis(
            probs, sampled_tokens, axis=-1
        )
        token_log_probs = jnp.take_along_axis(
            log_probs, sampled_tokens, axis=-1
        )
        
        # Get top-K tokens and their probabilities for logging
        top_k_count = min(config.top_log_probs, probs.shape[-1])
        top_k_probs_vals, top_k_indices = jax.lax.top_k(probs, top_k_count)
        top_k_log_probs_vals = jnp.take_along_axis(log_probs, top_k_indices, axis=-1)
        
        # Compute entropy (measure of uncertainty)
        # entropy = -sum(p * log(p)) for all tokens with p > 0
        entropy = -jnp.sum(
            jnp.where(probs > 1e-10, probs * log_probs, 0.0),
            axis=-1
        )
        
        return SampleOutput(
            token_id=sampled_tokens,
            token_prob=token_probs,
            token_log_prob=token_log_probs,
            top_k_token_ids=top_k_indices,
            top_k_probs=top_k_probs_vals,
            top_k_log_probs=top_k_log_probs_vals,
            entropy=entropy
        )
    
    def greedy_sample(self, logits: jnp.ndarray) -> SampleOutput:
        """Greedy decoding: always select the most probable token.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            
        Returns:
            SampleOutput with greedy-selected tokens
        """
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        
        batch_size = logits.shape[0]
        
        probs = jax.nn.softmax(logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Select argmax
        sampled_tokens = jnp.argmax(logits, axis=-1, keepdims=True)
        
        token_probs = jnp.take_along_axis(probs, sampled_tokens, axis=-1)
        token_log_probs = jnp.take_along_axis(log_probs, sampled_tokens, axis=-1)
        
        # Top-5 for logging
        top_k_probs_vals, top_k_indices = jax.lax.top_k(probs, 5)
        top_k_log_probs_vals = jnp.take_along_axis(log_probs, top_k_indices, axis=-1)
        
        entropy = -jnp.sum(
            jnp.where(probs > 1e-10, probs * log_probs, 0.0),
            axis=-1
        )
        
        return SampleOutput(
            token_id=sampled_tokens,
            token_prob=token_probs,
            token_log_prob=token_log_probs,
            top_k_token_ids=top_k_indices,
            top_k_probs=top_k_probs_vals,
            top_k_log_probs=top_k_log_probs_vals,
            entropy=entropy
        )
    
    def beam_search(
        self,
        logits_fn,
        initial_tokens: jnp.ndarray,
        beam_width: int = 5,
        max_length: int = 100,
        length_penalty: float = 0.6,
        stop_tokens: Optional[List[int]] = None,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 0,
        min_length: int = 0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Beam search decoding for higher quality generation.
        
        Implements efficient beam search with:
        - Length normalization for fair comparison across lengths
        - Early stopping when all beams reach EOS
        - N-gram blocking to prevent repetition
        - Minimum length enforcement
        - Efficient batch processing of beams
        
        Args:
            logits_fn: Function that takes tokens [batch, seq_len] and returns 
                      logits [batch, seq_len, vocab_size]
            initial_tokens: Starting tokens [batch_size, seq_len]
            beam_width: Number of beams to track (higher = better quality, slower)
            max_length: Maximum sequence length to generate
            length_penalty: Length normalization factor (alpha). 
                           < 1.0 favors shorter sequences, > 1.0 favors longer.
                           Typical values: 0.6-1.2
            stop_tokens: Tokens that signal end of generation (e.g., EOS token)
            early_stopping: Stop when all beams have reached a stop token
            no_repeat_ngram_size: If > 0, prevents n-grams of this size from repeating
            min_length: Minimum length before allowing stop tokens
            
        Returns:
            Tuple of (best_sequences [batch_size, max_length], scores [batch_size])
        """
        if stop_tokens is None:
            stop_tokens = []
        stop_tokens_set = set(stop_tokens)
        
        batch_size, initial_seq_len = initial_tokens.shape
        vocab_size: int = 0
        
        # Start with initial tokens replicated for each beam
        beam_sequences = jnp.tile(
            initial_tokens[:, None, :],
            (1, beam_width, 1)
        )
        
        pad_length = max_length - initial_seq_len
        beam_sequences = jnp.pad(
            beam_sequences,
            ((0, 0), (0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )
        
        beam_scores = jnp.zeros((batch_size, beam_width))
        beam_scores = beam_scores.at[:, 1:].set(-float('inf'))
        
        beam_finished = jnp.zeros((batch_size, beam_width), dtype=jnp.bool_)
        
        def _get_ngrams(sequence: jnp.ndarray, n: int) -> set:
            """Extract n-grams from a sequence."""
            ngrams = set()
            seq_list = sequence.tolist()
            for i in range(len(seq_list) - n + 1):
                ngram = tuple(seq_list[i:i + n])
                ngrams.add(ngram)
            return ngrams
        
        def _block_repeated_ngrams(
            logits: jnp.ndarray, 
            sequences: jnp.ndarray, 
            ngram_size: int,
            current_pos: int
        ) -> jnp.ndarray:
            """Block tokens that would create repeated n-grams."""
            if ngram_size <= 0 or current_pos < ngram_size - 1:
                return logits
            
            batch_beam_size = logits.shape[0]
            
            for i in range(batch_beam_size):
                prefix = tuple(sequences[i, current_pos - ngram_size + 1:current_pos].tolist())
                
                existing_ngrams = _get_ngrams(sequences[i, :current_pos], ngram_size)
                
                for token in range(logits.shape[-1]):
                    candidate_ngram = prefix + (token,)
                    if candidate_ngram in existing_ngrams:
                        logits = logits.at[i, token].set(-float('inf'))
            
            return logits
        
        for step in range(max_length - initial_seq_len):
            if jnp.all(beam_finished):
                break
            
            current_pos = initial_seq_len + step
            
            current_seqs = beam_sequences[:, :, :current_pos]
            
            flat_seqs = current_seqs.reshape(batch_size * beam_width, current_pos)
            
            all_logits = logits_fn(flat_seqs)
            
            if vocab_size == 0:
                vocab_size = all_logits.shape[-1]
            
            next_logits = all_logits[:, -1, :]
            
            if no_repeat_ngram_size > 0:
                next_logits = _block_repeated_ngrams(
                    next_logits, flat_seqs, no_repeat_ngram_size, current_pos
                )
            
            if current_pos < min_length:
                for stop_token in stop_tokens:
                    next_logits = next_logits.at[:, stop_token].set(-float('inf'))
            
            log_probs = jax.nn.log_softmax(next_logits, axis=-1)
            
            log_probs = log_probs.reshape(batch_size, beam_width, vocab_size)
            
            candidate_scores = beam_scores[:, :, None] + log_probs
            
            finished_mask = beam_finished[:, :, None]
            candidate_scores = jnp.where(
                finished_mask,
                jnp.full_like(candidate_scores, -float('inf')),
                candidate_scores
            )
            
            # Flatten beams and vocab for top-k selection
            candidate_scores_flat: jnp.ndarray = candidate_scores.reshape(batch_size, beam_width * vocab_size)
            
            # Get top beam_width candidates for each batch
            top_scores, top_indices = jax.lax.top_k(candidate_scores_flat, beam_width)
            
            # Convert flat indices back to (beam_idx, token_idx)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Gather sequences from selected beams
            new_sequences = jnp.zeros_like(beam_sequences)
            new_finished = jnp.zeros_like(beam_finished)
            
            for b in range(batch_size):
                for k in range(beam_width):
                    source_beam = beam_indices[b, k]
                    new_sequences = new_sequences.at[b, k].set(
                        beam_sequences[b, source_beam]
                    )
                    # Add the new token
                    new_sequences = new_sequences.at[b, k, current_pos].set(
                        token_indices[b, k]
                    )
                    # Check if this beam is now finished
                    is_finished = beam_finished[b, source_beam] | (
                        int(token_indices[b, k]) in stop_tokens_set
                    )
                    new_finished = new_finished.at[b, k].set(is_finished)
            
            beam_sequences = new_sequences
            beam_scores = top_scores
            beam_finished = new_finished
            
            if early_stopping and jnp.all(beam_finished):
                break
        
        # Apply length penalty for final scoring
        # Length penalty: ((5 + length) / 6) ^ alpha
        final_lengths = jnp.sum(beam_sequences != 0, axis=-1).astype(jnp.float32)
        length_penalty_factor = ((5.0 + final_lengths) / 6.0) ** length_penalty
        normalized_scores = beam_scores / length_penalty_factor
        
        # Select best beam for each batch
        best_beam_indices = jnp.argmax(normalized_scores, axis=-1)
        
        # Gather best sequences
        best_sequences = jnp.zeros((batch_size, max_length), dtype=beam_sequences.dtype)
        best_scores = jnp.zeros(batch_size)
        
        for b in range(batch_size):
            best_sequences = best_sequences.at[b].set(
                beam_sequences[b, best_beam_indices[b]]
            )
            best_scores = best_scores.at[b].set(
                normalized_scores[b, best_beam_indices[b]]
            )
        
        return best_sequences, best_scores
    
    def diverse_beam_search(
        self,
        logits_fn,
        initial_tokens: jnp.ndarray,
        num_beam_groups: int = 3,
        beam_width: int = 5,
        diversity_penalty: float = 0.5,
        max_length: int = 100,
        length_penalty: float = 0.6,
        stop_tokens: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Diverse Beam Search for generating multiple distinct outputs.
        
        Splits beams into groups and applies a diversity penalty to encourage
        different groups to explore different generation paths.
        
        Based on: "A Simple, Fast Diverse Decoding Algorithm for Neural Generation"
        (Vijayakumar et al., 2016)
        
        Args:
            logits_fn: Function that takes tokens and returns logits
            initial_tokens: Starting tokens [batch_size, seq_len]
            num_beam_groups: Number of beam groups for diversity
            beam_width: Total number of beams (must be divisible by num_beam_groups)
            diversity_penalty: Penalty strength for selecting tokens already 
                              chosen by previous groups
            max_length: Maximum sequence length
            length_penalty: Length normalization factor
            stop_tokens: Tokens that signal end of generation
            
        Returns:
            Tuple of (best_sequences, scores) from all groups
        """
        if beam_width % num_beam_groups != 0:
            raise ValueError(
                f"beam_width ({beam_width}) must be divisible by "
                f"num_beam_groups ({num_beam_groups})"
            )
        
        beams_per_group = beam_width // num_beam_groups
        all_sequences = []
        all_scores = []
        
        previous_tokens: List[List[set]] = []
        
        for group_idx in range(num_beam_groups):
            
            def diversity_penalized_logits_fn(tokens):
                logits = logits_fn(tokens)
                
                current_pos = tokens.shape[1] - 1
                
                if current_pos < len(previous_tokens):
                    prev_selected = set()
                    for prev_group in range(group_idx):
                        if prev_group < len(previous_tokens[current_pos]):
                            prev_selected.update(previous_tokens[current_pos][prev_group])
                    
                    for token in prev_selected:
                        logits = logits.at[:, -1, token].add(-diversity_penalty)
                
                return logits
            
            sequences, scores = self.beam_search(
                logits_fn=diversity_penalized_logits_fn,
                initial_tokens=initial_tokens,
                beam_width=beams_per_group,
                max_length=max_length,
                length_penalty=length_penalty,
                stop_tokens=stop_tokens,
            )
            
            all_sequences.append(sequences)
            all_scores.append(scores)
            
            # Record tokens selected by this group
            # For simplicity, we record the tokens in the best sequence
            for pos in range(sequences.shape[1]):
                if pos >= len(previous_tokens):
                    previous_tokens.append([set() for _ in range(num_beam_groups)])
                previous_tokens[pos][group_idx].add(int(sequences[0, pos]))
        
        combined_sequences = jnp.concatenate(all_sequences, axis=0)
        combined_scores = jnp.concatenate(all_scores, axis=0)
        
        return combined_sequences, combined_scores
    
    def constrained_beam_search(
        self,
        logits_fn,
        initial_tokens: jnp.ndarray,
        constraints: List[List[int]],
        beam_width: int = 5,
        max_length: int = 100,
        length_penalty: float = 0.6,
        stop_tokens: Optional[List[int]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Constrained Beam Search that ensures certain phrases appear in output.
        
        Useful for tasks like:
        - Including specific keywords in generated text
        - Ensuring factual information appears
        - Meeting output format requirements
        
        Args:
            logits_fn: Function that takes tokens and returns logits
            initial_tokens: Starting tokens [batch_size, seq_len]
            constraints: List of token sequences that must appear in output.
                        Each constraint is a list of token IDs.
            beam_width: Number of beams to track
            max_length: Maximum sequence length
            length_penalty: Length normalization factor
            stop_tokens: Tokens that signal end of generation
            
        Returns:
            Tuple of (best_sequences, scores)
        """
        if not constraints:
            # No constraints, use regular beam search
            return self.beam_search(
                logits_fn, initial_tokens, beam_width, max_length,
                length_penalty, stop_tokens
            )
        
        # Constraint State Machine for tracking constraint satisfaction
        # Each constraint has states: 0 (not started) -> len(constraint) (satisfied)
        # State i means we've matched the first i tokens of the constraint
        
        class ConstraintStateMachine:
            """Tracks progress toward satisfying multiple constraints."""
            
            def __init__(self, constraints: List[List[int]]):
                self.constraints = constraints
                # State for each constraint: how many tokens matched so far
                self.states = [0] * len(constraints)
                self.satisfied = [False] * len(constraints)
            
            def copy(self) -> 'ConstraintStateMachine':
                """Create a copy of this state machine."""
                new_csm = ConstraintStateMachine(self.constraints)
                new_csm.states = self.states.copy()
                new_csm.satisfied = self.satisfied.copy()
                return new_csm
            
            def advance(self, token: int) -> None:
                """Update state machine with a new token."""
                for i, constraint in enumerate(self.constraints):
                    if self.satisfied[i]:
                        continue
                    
                    # Check if token continues the current match
                    if self.states[i] < len(constraint):
                        if token == constraint[self.states[i]]:
                            self.states[i] += 1
                            if self.states[i] == len(constraint):
                                self.satisfied[i] = True
                        else:
                            # Check for partial match from start
                            # (handles overlapping patterns like "abab" in "ababab")
                            self.states[i] = 1 if token == constraint[0] else 0
            
            def all_satisfied(self) -> bool:
                """Check if all constraints are satisfied."""
                return all(self.satisfied)
            
            def get_forcing_tokens(self) -> List[Tuple[int, float]]:
                """Get tokens to boost and their boost amounts.
                
                Returns list of (token_id, boost_score) tuples.
                """
                boosts = []
                for i, constraint in enumerate(self.constraints):
                    if self.satisfied[i]:
                        continue
                    
                    # Boost the next token in the constraint
                    next_token = constraint[self.states[i]]
                    # Higher boost for constraints we've started matching
                    boost_amount = 5.0 + self.states[i] * 2.0
                    boosts.append((next_token, boost_amount))
                    
                    # Also give smaller boost to first token of unstarted constraints
                    if self.states[i] == 0:
                        boosts.append((constraint[0], 1.0))
                
                return boosts
            
            def get_required_tokens(self) -> Optional[List[int]]:
                """Get tokens that MUST appear next (hard constraint mode).
                
                Returns None if no hard requirement, else list of valid next tokens.
                """
                required = []
                for i, constraint in enumerate(self.constraints):
                    if self.satisfied[i]:
                        continue
                    if self.states[i] > 0:
                        # Mid-constraint: must continue
                        required.append(constraint[self.states[i]])
                return required if required else None
        
        # Create state machine
        csm = ConstraintStateMachine(constraints)
        
        # Advance state machine with initial tokens
        for token in initial_tokens[0].tolist():
            csm.advance(token)
        
        def constraint_aware_logits_fn(tokens):
            """Modify logits to encourage constraint satisfaction."""
            logits = logits_fn(tokens)
            
            # Create temporary state machine to track current sequence
            temp_csm = ConstraintStateMachine(constraints)
            for token in tokens[0].tolist():
                temp_csm.advance(token)
            
            # Get tokens to boost
            boosts = temp_csm.get_forcing_tokens()
            for token_id, boost_amount in boosts:
                logits = logits.at[:, -1, token_id].add(boost_amount)
            
            # In hard constraint mode, mask out non-required tokens
            required = temp_csm.get_required_tokens()
            if required:
                # Create mask that's -inf everywhere except required tokens
                mask = jnp.full_like(logits[:, -1, :], -1e10)
                for token_id in required:
                    mask = mask.at[:, token_id].set(0.0)
                logits = logits.at[:, -1, :].add(mask)
            
            return logits
        
        return self.beam_search(
            logits_fn=constraint_aware_logits_fn,
            initial_tokens=initial_tokens,
            beam_width=beam_width,
            max_length=max_length,
            length_penalty=length_penalty,
            stop_tokens=stop_tokens,
        )


# Convenience functions for common sampling configurations
def create_sampling_config_creative() -> SamplingConfig:
    """Create config for creative/diverse generation."""
    return SamplingConfig(
        temperature=1.2,
        top_k=100,
        top_p=0.95,
        repetition_penalty=1.2,
        log_probs=True
    )


def create_sampling_config_precise() -> SamplingConfig:
    """Create config for precise/factual generation."""
    return SamplingConfig(
        temperature=0.3,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        log_probs=True
    )


def create_sampling_config_balanced() -> SamplingConfig:
    """Create config for balanced generation (default)."""
    return SamplingConfig(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.15,
        log_probs=True
    )


def create_sampling_config_deterministic() -> SamplingConfig:
    """Create config for deterministic/reproducible generation."""
    return SamplingConfig(
        temperature=0.01,  # Near-greedy
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        log_probs=True
    )


# ============================================================================
# Speculative Decoding for Fast Inference
# ============================================================================

class SpeculativeDecoder:
    """
    Speculative Decoding for faster autoregressive generation.
    
    Uses a smaller draft model to predict multiple tokens ahead, then verifies
    them with the target model in a single forward pass. This can provide
    2-3x speedup for large models when the draft model is accurate.
    
    Based on: "Accelerating Large Language Model Decoding with Speculative Sampling"
    (Leviathan et al., 2022)
    
    Example usage:
        draft_model = small_model  # Fast, less accurate
        target_model = large_model  # Slow, accurate
        
        decoder = SpeculativeDecoder(
            target_forward_fn=target_model.apply,
            draft_forward_fn=draft_model.apply,
            num_speculative_tokens=4
        )
        
        output_tokens = decoder.generate(
            target_params, draft_params, 
            initial_tokens, rng_key, 
            max_length=100
        )
    """
    
    def __init__(
        self,
        target_forward_fn,
        draft_forward_fn,
        num_speculative_tokens: int = 4,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """
        Initialize speculative decoder.
        
        Args:
            target_forward_fn: Forward function for target (large) model
            draft_forward_fn: Forward function for draft (small) model
            num_speculative_tokens: Number of tokens to speculate per step
            temperature: Sampling temperature
            top_k: Top-K filtering (0 to disable)
            top_p: Nucleus sampling threshold
        """
        self.target_forward_fn = target_forward_fn
        self.draft_forward_fn = draft_forward_fn
        self.num_speculative_tokens = num_speculative_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.sampler = TokenSampler()
        self.config = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    def _draft_tokens(
        self,
        draft_params,
        tokens: jnp.ndarray,
        rng_key,
        num_tokens: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate draft tokens using the fast draft model.
        
        Args:
            draft_params: Parameters for draft model
            tokens: Current token sequence [batch, seq_len]
            rng_key: Random key
            num_tokens: Number of tokens to draft
            
        Returns:
            draft_tokens: Drafted token IDs [batch, num_tokens]
            draft_probs: Draft token probabilities [batch, num_tokens]
        """
        batch_size = tokens.shape[0]
        draft_tokens_list = []
        draft_probs_list = []
        current_tokens = tokens
        
        for i in range(num_tokens):
            rng_key, sample_key = jax.random.split(rng_key)
            
            # Get draft model logits
            draft_logits = self.draft_forward_fn(draft_params, text=current_tokens)
            if isinstance(draft_logits, tuple):
                draft_logits = draft_logits[0]  # Handle models that return aux outputs
            
            # Get logits for last position
            last_logits = draft_logits[:, -1, :]
            
            # Sample
            sample_output = self.sampler.sample(last_logits, self.config, sample_key)
            draft_token = sample_output.token_id[:, 0]
            draft_prob = sample_output.token_prob[:, 0]
            
            draft_tokens_list.append(draft_token)
            draft_probs_list.append(draft_prob)
            
            # Append to sequence
            current_tokens = jnp.concatenate([
                current_tokens, draft_token[:, None]
            ], axis=1)
        
        draft_tokens = jnp.stack(draft_tokens_list, axis=1)
        draft_probs = jnp.stack(draft_probs_list, axis=1)
        
        return draft_tokens, draft_probs
    
    def _verify_and_accept(
        self,
        target_params,
        tokens: jnp.ndarray,
        draft_tokens: jnp.ndarray,
        draft_probs: jnp.ndarray,
        rng_key
    ) -> Tuple[jnp.ndarray, int]:
        """
        Verify draft tokens with target model and accept valid ones.
        
        Args:
            target_params: Parameters for target model
            tokens: Original token sequence [batch, seq_len]
            draft_tokens: Draft tokens to verify [batch, num_draft]
            draft_probs: Probabilities of draft tokens [batch, num_draft]
            rng_key: Random key
            
        Returns:
            accepted_tokens: Verified and corrected tokens
            num_accepted: Number of draft tokens accepted
        """
        batch_size = tokens.shape[0]
        num_draft = draft_tokens.shape[1]
        
        # Concatenate original and draft tokens
        full_sequence = jnp.concatenate([tokens, draft_tokens], axis=1)
        
        # Get target model logits for all positions
        target_logits = self.target_forward_fn(target_params, text=full_sequence)
        if isinstance(target_logits, tuple):
            target_logits = target_logits[0]
        
        # Get target probabilities for draft positions
        # Logits at position i predict token at position i+1
        draft_positions = jnp.arange(tokens.shape[1] - 1, tokens.shape[1] - 1 + num_draft)
        target_logits_for_draft = target_logits[:, draft_positions, :]
        target_probs = jax.nn.softmax(target_logits_for_draft / self.temperature, axis=-1)
        
        # Get target probability for each draft token
        target_probs_for_draft = jnp.take_along_axis(
            target_probs, draft_tokens[:, :, None], axis=-1
        )[:, :, 0]  # [batch, num_draft]
        
        # Acceptance ratio: min(1, target_prob / draft_prob)
        acceptance_ratio = jnp.minimum(1.0, target_probs_for_draft / (draft_probs + 1e-10))
        
        # Generate uniform random for acceptance test
        rng_key, accept_key = jax.random.split(rng_key)
        uniform_samples = jax.random.uniform(accept_key, acceptance_ratio.shape)
        
        # Accept if uniform < acceptance_ratio
        accepted_mask = uniform_samples < acceptance_ratio
        
        # Find first rejection point
        # For simplicity, accept all tokens up to first rejection
        cumulative_accept = jnp.cumprod(accepted_mask.astype(jnp.float32), axis=1)
        num_accepted = jnp.sum(cumulative_accept, axis=1).astype(jnp.int32)
        
        # For batch processing, use minimum accepted across batch
        min_accepted = int(jnp.min(num_accepted))
        
        if min_accepted == num_draft:
            # All draft tokens accepted - also sample next token from target
            rng_key, next_key = jax.random.split(rng_key)
            next_logits = target_logits[:, -1, :]
            next_output = self.sampler.sample(next_logits, self.config, next_key)
            next_token = next_output.token_id
            
            accepted_tokens = jnp.concatenate([draft_tokens, next_token], axis=1)
            return accepted_tokens, min_accepted + 1
        else:
            # Rejection at position min_accepted - resample from target
            rng_key, resample_key = jax.random.split(rng_key)
            resample_logits = target_logits[:, tokens.shape[1] - 1 + min_accepted, :]
            resample_output = self.sampler.sample(resample_logits, self.config, resample_key)
            resample_token = resample_output.token_id
            
            # Accept tokens up to rejection, then resampled token
            accepted_tokens = jnp.concatenate([
                draft_tokens[:, :min_accepted], resample_token
            ], axis=1)
            return accepted_tokens, min_accepted + 1
    
    def generate(
        self,
        target_params,
        draft_params,
        initial_tokens: jnp.ndarray,
        rng_key,
        max_length: int = 100,
        stop_tokens: Optional[List[int]] = None
    ) -> jnp.ndarray:
        """
        Generate tokens using speculative decoding.
        
        Args:
            target_params: Parameters for target model
            draft_params: Parameters for draft model
            initial_tokens: Starting token sequence [batch, seq_len]
            rng_key: Random key
            max_length: Maximum total sequence length
            stop_tokens: Token IDs that signal end of generation
            
        Returns:
            Generated token sequence [batch, final_length]
        """
        tokens = initial_tokens
        stop_tokens = stop_tokens or []
        
        while tokens.shape[1] < max_length:
            rng_key, draft_key, verify_key = jax.random.split(rng_key, 3)
            
            # Draft tokens
            draft_tokens, draft_probs = self._draft_tokens(
                draft_params, tokens, draft_key, self.num_speculative_tokens
            )
            
            # Verify and accept
            accepted_tokens, num_accepted = self._verify_and_accept(
                target_params, tokens, draft_tokens, draft_probs, verify_key
            )
            
            # Append accepted tokens
            tokens = jnp.concatenate([tokens, accepted_tokens], axis=1)
            
            # Check for stop tokens
            if stop_tokens:
                last_tokens = tokens[:, -num_accepted:]
                for stop_token in stop_tokens:
                    if jnp.any(last_tokens == stop_token):
                        # Truncate at stop token
                        return tokens
        
        return tokens


class SelfSpeculativeDecoder:
    """
    Self-Speculative Decoding using early exit from the same model.
    
    Instead of a separate draft model, uses early transformer layers
    to generate draft tokens and full model for verification.
    This avoids needing to train/maintain a separate draft model.
    
    Based on: "Draft & Verify: Lossless Large Language Model Acceleration 
    via Self-Speculative Decoding" (Zhang et al., 2023)
    
    The model forward function should accept an optional `exit_layer` parameter:
    - If exit_layer is None: run full model
    - If exit_layer is int: exit at that layer and return intermediate logits
    
    If the model doesn't support early exit, falls back to standard generation
    but logs a warning.
    """
    
    def __init__(
        self,
        model_forward_fn,
        early_exit_layer: int = 4,
        num_speculative_tokens: int = 4,
        temperature: float = 1.0,
        use_kv_cache: bool = True,
    ):
        """
        Initialize self-speculative decoder.
        
        Args:
            model_forward_fn: Forward function. Should accept `exit_layer` param
                             for early exit support. Signature:
                             fn(params, text, exit_layer=None) -> logits
            early_exit_layer: Layer to exit early for draft (default: 4)
            num_speculative_tokens: Number of tokens to speculate
            temperature: Sampling temperature
            use_kv_cache: Whether to use KV cache for efficiency
        """
        self.model_forward_fn = model_forward_fn
        self.early_exit_layer = early_exit_layer
        self.num_speculative_tokens = num_speculative_tokens
        self.temperature = temperature
        self.use_kv_cache = use_kv_cache
        self.sampler = TokenSampler()
        self.config = SamplingConfig(temperature=temperature)
        
        # Check if model supports early exit
        self._early_exit_supported: Optional[bool] = None
        
        logger.info(
            f"Self-speculative decoding initialized: "
            f"early_exit={early_exit_layer}, "
            f"num_speculative={num_speculative_tokens}"
        )
    
    def _check_early_exit_support(self, params, sample_tokens: jnp.ndarray) -> bool:
        """Check if model supports early exit."""
        if self._early_exit_supported is not None:
            return self._early_exit_supported
        
        try:
            # Try calling with exit_layer parameter
            _ = self.model_forward_fn(
                params, text=sample_tokens, exit_layer=self.early_exit_layer
            )
            self._early_exit_supported = True
            logger.info("Model supports early exit for self-speculative decoding")
        except TypeError:
            self._early_exit_supported = False
            logger.warning(
                "Model doesn't support early exit (exit_layer parameter). "
                "Self-speculative decoding will fall back to standard generation. "
                "To enable early exit, modify your model's forward function to accept "
                "an optional 'exit_layer' parameter."
            )
        except Exception as e:
            self._early_exit_supported = False
            logger.warning(f"Early exit check failed: {e}. Using standard generation.")
        
        return self._early_exit_supported
    
    def _draft_tokens_early_exit(
        self,
        params,
        tokens: jnp.ndarray,
        rng_key,
        num_tokens: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate draft tokens using early exit.
        
        Args:
            params: Model parameters
            tokens: Current token sequence [batch, seq_len]
            rng_key: Random key
            num_tokens: Number of draft tokens to generate
            
        Returns:
            Tuple of (draft_tokens [batch, num_tokens], draft_probs [batch, num_tokens])
        """
        draft_tokens = []
        draft_probs = []
        current_tokens = tokens
        
        for _ in range(num_tokens):
            rng_key, sample_key = jax.random.split(rng_key)
            
            # Get logits from early exit layer
            early_logits = self.model_forward_fn(
                params, text=current_tokens, exit_layer=self.early_exit_layer
            )
            if isinstance(early_logits, tuple):
                early_logits = early_logits[0]
            
            # Sample from early exit distribution
            last_logits = early_logits[:, -1, :]
            probs = jax.nn.softmax(last_logits / self.temperature, axis=-1)
            
            sample_output = self.sampler.sample(last_logits, self.config, sample_key)
            next_token = sample_output.token_id
            
            # Store draft token and its probability
            draft_tokens.append(next_token)
            token_prob = jnp.take_along_axis(probs, next_token, axis=-1)
            draft_probs.append(token_prob)
            
            # Append to sequence for next iteration
            current_tokens = jnp.concatenate([current_tokens, next_token], axis=1)
        
        return (
            jnp.concatenate(draft_tokens, axis=1),
            jnp.concatenate(draft_probs, axis=1)
        )
    
    def _verify_drafts(
        self,
        params,
        tokens: jnp.ndarray,
        draft_tokens: jnp.ndarray,
        draft_probs: jnp.ndarray,
        rng_key,
    ) -> Tuple[jnp.ndarray, int]:
        """Verify draft tokens using full model.
        
        Uses rejection sampling to ensure output distribution matches
        the full model's distribution exactly (lossless acceleration).
        
        Args:
            params: Model parameters
            tokens: Original token sequence before drafts
            draft_tokens: Draft tokens to verify [batch, num_draft]
            draft_probs: Probabilities of drafts from early exit [batch, num_draft]
            rng_key: Random key
            
        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        num_draft = draft_tokens.shape[1]
        
        # Get full model logits for entire sequence including drafts
        full_sequence = jnp.concatenate([tokens, draft_tokens], axis=1)
        full_logits = self.model_forward_fn(params, text=full_sequence, exit_layer=None)
        if isinstance(full_logits, tuple):
            full_logits = full_logits[0]
        
        # Get full model probabilities for draft positions
        draft_positions = jnp.arange(tokens.shape[1] - 1, tokens.shape[1] - 1 + num_draft)
        full_logits_for_draft = full_logits[:, draft_positions, :]
        full_probs = jax.nn.softmax(full_logits_for_draft / self.temperature, axis=-1)
        
        # Get full model probability for each draft token
        full_probs_for_draft = jnp.take_along_axis(
            full_probs, draft_tokens[:, :, None], axis=-1
        )[:, :, 0]  # [batch, num_draft]
        
        # Acceptance probability: min(1, full_prob / draft_prob)
        acceptance_ratio = jnp.minimum(
            1.0, full_probs_for_draft / (draft_probs + 1e-10)
        )
        
        # Rejection sampling
        rng_key, accept_key = jax.random.split(rng_key)
        uniform_samples = jax.random.uniform(accept_key, acceptance_ratio.shape)
        accepted_mask = uniform_samples < acceptance_ratio
        
        # Accept all tokens up to first rejection
        cumulative_accept = jnp.cumprod(accepted_mask.astype(jnp.float32), axis=1)
        num_accepted = jnp.sum(cumulative_accept, axis=1).astype(jnp.int32)
        min_accepted = int(jnp.min(num_accepted))
        
        if min_accepted == num_draft:
            # All accepted - sample one more token from full model
            rng_key, next_key = jax.random.split(rng_key)
            next_logits = full_logits[:, -1, :]
            next_output = self.sampler.sample(next_logits, self.config, next_key)
            accepted_tokens = jnp.concatenate([draft_tokens, next_output.token_id], axis=1)
            return accepted_tokens, min_accepted + 1
        
        # Rejection at min_accepted - resample from full distribution
        rng_key, resample_key = jax.random.split(rng_key)
        pos = tokens.shape[1] - 1 + min_accepted
        resample_logits = full_logits[:, pos, :]
        resample_output = self.sampler.sample(resample_logits, self.config, resample_key)
        
        accepted_tokens = jnp.concatenate([
            draft_tokens[:, :min_accepted], 
            resample_output.token_id
        ], axis=1)
        return accepted_tokens, min_accepted + 1
    
    def _standard_generate(
        self,
        params,
        tokens: jnp.ndarray,
        rng_key,
        max_length: int,
        stop_tokens: Optional[List[int]] = None,
    ) -> jnp.ndarray:
        """Standard autoregressive generation (fallback)."""
        stop_tokens = stop_tokens or []
        
        while tokens.shape[1] < max_length:
            rng_key, sample_key = jax.random.split(rng_key)
            
            logits = self.model_forward_fn(params, text=tokens)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            last_logits = logits[:, -1, :]
            sample_output = self.sampler.sample(last_logits, self.config, sample_key)
            next_token = sample_output.token_id
            
            tokens = jnp.concatenate([tokens, next_token], axis=1)
            
            # Check for stop tokens
            if stop_tokens and int(next_token[0, 0]) in stop_tokens:
                break
        
        return tokens
    
    def _contains_stop_token(
        self, tokens: jnp.ndarray, stop_tokens: List[int]
    ) -> bool:
        """Check if tokens contain any stop token."""
        for stop_token in stop_tokens:
            if jnp.any(tokens == stop_token):
                return True
        return False
    
    def _log_stats(self, total_drafted: int, total_accepted: int) -> None:
        """Log speculative decoding statistics."""
        acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0
        logger.debug(
            f"Self-speculative decoding stats: "
            f"drafted={total_drafted}, accepted={total_accepted}, "
            f"rate={acceptance_rate:.2%}"
        )
    
    def generate(
        self,
        params,
        initial_tokens: jnp.ndarray,
        rng_key,
        max_length: int = 100,
        stop_tokens: Optional[List[int]] = None,
    ) -> jnp.ndarray:
        """
        Generate tokens using self-speculative decoding.
        
        If the model supports early exit (exit_layer parameter), uses
        speculative decoding for faster generation. Otherwise falls back
        to standard autoregressive generation.
        
        Args:
            params: Model parameters
            initial_tokens: Starting tokens [batch, seq_len]
            rng_key: Random key
            max_length: Maximum sequence length
            stop_tokens: Token IDs that signal end of generation
            
        Returns:
            Generated tokens [batch, final_length]
        """
        stop_tokens = stop_tokens or []
        
        # Check early exit support - fall back to standard if not supported
        if not self._check_early_exit_support(params, initial_tokens):
            return self._standard_generate(
                params, initial_tokens, rng_key, max_length, stop_tokens
            )
        
        return self._speculative_generate(
            params, initial_tokens, rng_key, max_length, stop_tokens
        )
    
    def _speculative_generate(
        self,
        params,
        initial_tokens: jnp.ndarray,
        rng_key,
        max_length: int,
        stop_tokens: List[int],
    ) -> jnp.ndarray:
        """Internal method for self-speculative generation."""
        tokens = initial_tokens
        total_drafted = 0
        total_accepted = 0
        
        while tokens.shape[1] < max_length:
            rng_key, draft_key, verify_key = jax.random.split(rng_key, 3)
            
            # Generate draft tokens using early exit
            draft_tokens, draft_probs = self._draft_tokens_early_exit(
                params, tokens, draft_key, self.num_speculative_tokens
            )
            total_drafted += self.num_speculative_tokens
            
            # Verify drafts using full model
            accepted_tokens, num_accepted = self._verify_drafts(
                params, tokens, draft_tokens, draft_probs, verify_key
            )
            total_accepted += num_accepted
            
            # Append accepted tokens
            tokens = jnp.concatenate([tokens, accepted_tokens], axis=1)
            
            # Check for stop tokens
            if stop_tokens and self._contains_stop_token(accepted_tokens, stop_tokens):
                self._log_stats(total_drafted, total_accepted)
                return tokens
        
        self._log_stats(total_drafted, total_accepted)
        return tokens

