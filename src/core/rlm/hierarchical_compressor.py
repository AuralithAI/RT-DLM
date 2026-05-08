"""
Hierarchical Context Compressor for RLM

Automatically compresses context when it exceeds a threshold, storing
compressed summaries in the ContextStore with hierarchical tags.  This
prevents memory blow-up during deep recursive reasoning while preserving
the most salient information at each compression tier.

Compression tiers:
    Tier-0: Raw context (original text).
    Tier-1: Extractive summary (~4× compression).
    Tier-2: Ultra-compressed key-point summary (~16× compression).

Trigger conditions (any one is sufficient):
    1. Context length > auto_compress_threshold (default 32 000 chars).
    2. ContextStore utilisation > store_utilisation_threshold (default 70%).
    3. Explicit call from RLMOrchestrator when uncertainty is high.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.core.rlm.context_store import ContextStore
from src.core.rlm.context_tools import ContextTools

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of a hierarchical compression pass."""

    tier: int
    original_length: int
    compressed_length: int
    compression_ratio: float
    summary_var: str
    success: bool
    error: Optional[str] = None


@dataclass
class CompressionStats:
    """Cumulative compression statistics."""

    total_compressions: int = 0
    total_chars_saved: int = 0
    tier_counts: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})
    avg_compression_ratio: float = 0.0


class HierarchicalCompressor:
    """Hierarchical context compressor integrated with ContextStore.

    Usage::

        compressor = HierarchicalCompressor(
            context_store=store,
            context_tools=tools,
            auto_compress_threshold=32000,
        )

        # Auto-check: compresses only if context exceeds threshold
        result = compressor.maybe_compress("main_context")

        # Force a specific tier
        result = compressor.compress("main_context", target_tier=2)

        # Check if compression is needed
        if compressor.should_compress("main_context"):
            compressor.maybe_compress("main_context")
    """

    def __init__(
        self,
        context_store: ContextStore,
        context_tools: ContextTools,
        auto_compress_threshold: int = 32_000,
        store_utilisation_threshold: float = 0.70,
        tier1_max_tokens: int = 2000,
        tier2_max_tokens: int = 500,
    ):
        self._store = context_store
        self._tools = context_tools
        self.auto_compress_threshold = auto_compress_threshold
        self.store_utilisation_threshold = store_utilisation_threshold
        self.tier1_max_tokens = tier1_max_tokens
        self.tier2_max_tokens = tier2_max_tokens
        self._stats = CompressionStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_compress(self, context_var: str) -> bool:
        """Check whether *context_var* should be compressed.

        Returns True when **any** trigger fires:
        1. Content length > ``auto_compress_threshold``.
        2. Store utilisation > ``store_utilisation_threshold``.
        """
        metadata = self._store.get_metadata(context_var)
        if metadata is None:
            return False

        # Trigger 1 — context too large
        if metadata.total_length > self.auto_compress_threshold:
            return True

        # Trigger 2 — store running out of space
        stats = self._store.stats()
        utilisation = stats["total_size"] / max(stats["max_total_size"], 1)
        if utilisation > self.store_utilisation_threshold:
            return True

        return False

    def maybe_compress(
        self,
        context_var: str,
        target_tier: Optional[int] = None,
    ) -> Optional[CompressionResult]:
        """Compress *context_var* only if a trigger condition fires.

        If ``target_tier`` is not given, picks the tier automatically:
        - context_length ≤ threshold  → no compression (returns None)
        - threshold < len ≤ 4×threshold  → tier-1
        - len > 4×threshold  → tier-2

        Returns:
            CompressionResult on success, None when no compression needed.
        """
        if not self.should_compress(context_var):
            return None

        if target_tier is None:
            target_tier = self._pick_tier(context_var)

        return self.compress(context_var, target_tier)

    def compress(
        self,
        context_var: str,
        target_tier: int = 1,
    ) -> CompressionResult:
        """Force-compress *context_var* to the given tier.

        Tier-1: extractive summary (summarize tool, ``tier1_max_tokens``).
        Tier-2: ultra-compressed (summarize again on the tier-1 result).
        """
        var = self._store.get(context_var)
        if var is None:
            return CompressionResult(
                tier=target_tier,
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                summary_var="",
                success=False,
                error=f"Variable '{context_var}' not found",
            )

        original_length = len(var.content)

        if target_tier <= 0:
            # Tier-0 = raw text; nothing to do.
            return CompressionResult(
                tier=0,
                original_length=original_length,
                compressed_length=original_length,
                compression_ratio=1.0,
                summary_var=context_var,
                success=True,
            )

        # ---- Tier 1 ----
        tier1_result = self._summarize_to_tier(
            context_var,
            tier=1,
            max_tokens=self.tier1_max_tokens,
        )
        if not tier1_result.success:
            return tier1_result

        if target_tier <= 1:
            self._record_stats(tier1_result)
            return tier1_result

        # ---- Tier 2 ----
        tier2_result = self._summarize_to_tier(
            tier1_result.summary_var,
            tier=2,
            max_tokens=self.tier2_max_tokens,
        )
        if not tier2_result.success:
            return tier2_result

        # Fix up the result to reflect total compression from raw → tier-2
        tier2_result = CompressionResult(
            tier=2,
            original_length=original_length,
            compressed_length=tier2_result.compressed_length,
            compression_ratio=original_length / max(tier2_result.compressed_length, 1),
            summary_var=tier2_result.summary_var,
            success=True,
        )
        self._record_stats(tier2_result)
        return tier2_result

    def get_compressed_var(self, context_var: str, tier: int = 1) -> Optional[str]:
        """Return the name of the compressed variable, if it exists."""
        name = f"{context_var}_compressed_t{tier}"
        if name in self._store:
            return name
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_compressions": self._stats.total_compressions,
            "total_chars_saved": self._stats.total_chars_saved,
            "tier_counts": dict(self._stats.tier_counts),
            "avg_compression_ratio": self._stats.avg_compression_ratio,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pick_tier(self, context_var: str) -> int:
        """Auto-select compression tier based on content length."""
        metadata = self._store.get_metadata(context_var)
        if metadata is None:
            return 1

        length = metadata.total_length
        if length > self.auto_compress_threshold * 4:
            return 2
        return 1

    def _summarize_to_tier(
        self,
        context_var: str,
        tier: int,
        max_tokens: int,
    ) -> CompressionResult:
        """Run the summarize tool and store the result as a compressed var."""
        var = self._store.get(context_var)
        if var is None:
            return CompressionResult(
                tier=tier,
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                summary_var="",
                success=False,
                error=f"Variable '{context_var}' not found",
            )

        original_length = len(var.content)

        # Use ContextTools.summarize (reuses existing extractive / model-based logic)
        tool_result = self._tools.summarize(context_var, max_tokens=max_tokens)
        if not tool_result.success:
            return CompressionResult(
                tier=tier,
                original_length=original_length,
                compressed_length=original_length,
                compression_ratio=1.0,
                summary_var=context_var,
                success=False,
                error=tool_result.error,
            )

        # The summarize tool stores the result as {context_var}_summary.
        # We rename it to {root_var}_compressed_t{tier} for stable naming.
        summary_data = tool_result.data
        summary_text = summary_data["summary"] if isinstance(summary_data, dict) else str(summary_data)

        # Derive root var name (strip existing _compressed_tN suffix)
        root_var = context_var
        for suffix in ("_compressed_t1", "_compressed_t2", "_summary"):
            if root_var.endswith(suffix):
                root_var = root_var[: -len(suffix)]

        compressed_var_name = f"{root_var}_compressed_t{tier}"
        self._store.store(
            name=compressed_var_name,
            content=summary_text,
            source=f"tier-{tier} compression of {context_var}",
            tags=["compressed", f"tier_{tier}", f"source_{root_var}"],
            parent_var=root_var,
        )

        compressed_length = len(summary_text)
        return CompressionResult(
            tier=tier,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=original_length / max(compressed_length, 1),
            summary_var=compressed_var_name,
            success=True,
        )

    def _record_stats(self, result: CompressionResult) -> None:
        self._stats.total_compressions += 1
        self._stats.total_chars_saved += result.original_length - result.compressed_length
        self._stats.tier_counts[result.tier] = self._stats.tier_counts.get(result.tier, 0) + 1
        # Running average
        n = self._stats.total_compressions
        self._stats.avg_compression_ratio = (self._stats.avg_compression_ratio * (n - 1) + result.compression_ratio) / n
