from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Mapping
import haiku as hk
import jax
import jax.numpy as jnp

from config.rlm_config import ToolType, RLMConfig

Params = Mapping[str, Any]


@dataclass
class ToolSelection:
    tool: ToolType
    confidence: float
    parameters: Dict[str, Any]
    reasoning: Optional[str] = None


class ToolSelector(hk.Module):
    def __init__(
        self,
        d_model: int,
        num_tools: int = len(ToolType),
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_tools = num_tools

        self.query_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        ])

        self.context_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        ])

        self.state_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
        ])

        self.tool_classifier = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(num_tools),
        ])

        self.parameter_heads = {
            "peek_start": hk.Linear(1),
            "peek_length": hk.Linear(1),
            "grep_regex": hk.Linear(1),
            "partition_strategy": hk.Linear(4),
            "summarize_length": hk.Linear(1),
        }

        self.termination_head = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid,
        ])

    def __call__(
        self,
        query: jnp.ndarray,
        context_metadata: jnp.ndarray,
        recursion_state: jnp.ndarray,
        tool_history: Optional[jnp.ndarray] = None,
        temperature: float = 0.1,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        if query.ndim == 3:
            query = query.mean(axis=1)
        if context_metadata.ndim == 3:
            context_metadata = context_metadata.mean(axis=1)

        encoded_query = self.query_encoder(query)
        encoded_context = self.context_encoder(context_metadata)
        encoded_state = self.state_encoder(recursion_state)

        if tool_history is not None:
            if tool_history.ndim == 3:
                tool_history = tool_history.mean(axis=1)
            combined = jnp.concatenate([
                encoded_query, encoded_context, encoded_state, tool_history
            ], axis=-1)
            projection = hk.Linear(self.d_model * 2)
            combined = projection(combined)
        else:
            combined = jnp.concatenate([encoded_query, encoded_context], axis=-1)

        tool_logits = self.tool_classifier(combined)
        tool_probs = jax.nn.softmax(tool_logits / temperature, axis=-1)

        termination_prob = self.termination_head(encoded_state)

        parameters = {}
        param_input = jnp.concatenate([encoded_query, encoded_context], axis=-1)
        param_projection = hk.Linear(self.d_model)
        param_features = param_projection(param_input)

        for name, head in self.parameter_heads.items():
            parameters[name] = head(param_features)

        return tool_probs, termination_prob, parameters

    def select_tool(
        self,
        tool_probs: jnp.ndarray,
        termination_prob: jnp.ndarray,
        parameters: Dict[str, jnp.ndarray],
        available_tools: List[ToolType],
        deterministic: bool = True,
    ) -> ToolSelection:
        tool_list = list(ToolType)

        if float(termination_prob.mean()) > 0.8:
            return ToolSelection(
                tool=ToolType.TERMINATE,
                confidence=float(termination_prob.mean()),
                parameters={},
            )

        mask = jnp.zeros(len(tool_list))
        for i, tool in enumerate(tool_list):
            if tool in available_tools:
                mask = mask.at[i].set(1.0)

        masked_probs = tool_probs * mask
        masked_probs = masked_probs / (masked_probs.sum(axis=-1, keepdims=True) + 1e-8)

        if deterministic:
            tool_idx = int(jnp.argmax(masked_probs, axis=-1).squeeze())
        else:
            raise RuntimeError(
                "Non-deterministic tool selection requires Haiku RNG context. "
                "Use deterministic=True or call within a Haiku-transformed function."
            )

        selected_tool = tool_list[tool_idx]
        confidence = float(masked_probs.squeeze()[tool_idx])

        tool_params = self._extract_parameters(selected_tool, parameters)

        return ToolSelection(
            tool=selected_tool,
            confidence=confidence,
            parameters=tool_params,
        )

    def _extract_parameters(
        self,
        tool: ToolType,
        parameters: Dict[str, jnp.ndarray],
    ) -> Dict[str, Any]:
        params = {}

        if tool == ToolType.PEEK:
            params["start"] = max(0, int(parameters["peek_start"].squeeze() * 10000))
            params["length"] = max(100, int(jax.nn.sigmoid(parameters["peek_length"].squeeze()) * 4000))

        elif tool == ToolType.GREP:
            params["regex"] = bool(jax.nn.sigmoid(parameters["grep_regex"].squeeze()) > 0.5)

        elif tool == ToolType.PARTITION:
            strategy_logits = parameters["partition_strategy"].squeeze()
            strategy_idx = int(jnp.argmax(strategy_logits))
            strategies = ["semantic", "fixed_size", "paragraph", "sentence"]
            params["strategy"] = strategies[min(strategy_idx, len(strategies) - 1)]

        elif tool == ToolType.SUMMARIZE:
            params["max_tokens"] = max(100, int(jax.nn.sigmoid(parameters["summarize_length"].squeeze()) * 1000))

        return params


class ToolSelectorWrapper:
    def __init__(self, d_model: int, config: RLMConfig):
        self.d_model = d_model
        self.config = config
        self._init_fn = None
        self._apply_fn = None
        self._params: Optional[Params] = None

    def init(self, rng: jnp.ndarray) -> Params:
        def forward(query, context_metadata, recursion_state):
            selector = ToolSelector(self.d_model)
            return selector(query, context_metadata, recursion_state)

        self._init_fn, self._apply_fn = hk.transform(forward)

        dummy_query = jnp.zeros((1, self.d_model))
        dummy_context = jnp.zeros((1, self.d_model))
        dummy_state = jnp.zeros((1, self.d_model))

        self._params = self._init_fn(rng, dummy_query, dummy_context, dummy_state)
        return self._params

    def select(
        self,
        params: Params,
        rng: jnp.ndarray,
        query: jnp.ndarray,
        context_metadata: jnp.ndarray,
        recursion_state: jnp.ndarray,
        deterministic: bool = True,
    ) -> ToolSelection:
        def forward_and_select(query, context_metadata, recursion_state):
            selector = ToolSelector(self.d_model)
            tool_probs, term_prob, parameters = selector(
                query, context_metadata, recursion_state,
                temperature=self.config.tool_temperature,
            )
            return tool_probs, term_prob, parameters

        _, apply_fn = hk.transform(forward_and_select)
        tool_probs, term_prob, parameters = apply_fn(
            params, rng, query, context_metadata, recursion_state
        )

        def select_fn(tool_probs, term_prob, parameters):
            selector = ToolSelector(self.d_model)
            return selector.select_tool(
                tool_probs, term_prob, parameters,
                self.config.available_tools, deterministic
            )

        return select_fn(tool_probs, term_prob, parameters)
