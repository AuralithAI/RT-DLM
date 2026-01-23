from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Mapping
import time
import haiku as hk
import jax
import jax.numpy as jnp

from config.rlm_config import RLMConfig, ToolType
from core.rlm.context_store import ContextStore
from core.rlm.context_tools import ContextTools, ToolResult
from core.rlm.tool_selector import ToolSelector, ToolSelection
from core.rlm.recursive_manager import RecursiveCallManager, RecursionContext

Params = Mapping[str, Any]


@dataclass
class RLMResult:
    answer: Any
    success: bool
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    total_tool_calls: int = 0
    max_recursion_depth_reached: int = 0
    execution_time: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    intermediate_results: List[Any] = field(default_factory=list)


class RecursiveLanguageModel(hk.Module):
    def __init__(
        self,
        d_model: int,
        config: Optional[RLMConfig] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.config = config or RLMConfig()
        self.config.validate()

        self.tool_selector = ToolSelector(d_model, len(ToolType))

        self.query_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        ])

        self.context_metadata_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
        ])

        self.recursion_state_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
        ])

        self.answer_synthesizer = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        ])

        self.result_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
        ])

    def __call__(
        self,
        query_embedding: jnp.ndarray,
        context_length: int,
        recursion_depth: int = 0,
        tool_calls_used: int = 0,
    ) -> tuple:
        if query_embedding.ndim == 3:
            query_embedding = query_embedding.mean(axis=1)

        encoded_query = self.query_encoder(query_embedding)

        context_features = jnp.array([[
            float(context_length) / 10000.0,
            float(context_length > self.config.auto_partition_threshold),
            float(context_length > self.config.direct_context_threshold),
        ]])
        context_projection = hk.Linear(self.d_model)
        context_metadata = self.context_metadata_encoder(context_projection(context_features))

        recursion_features = jnp.array([[
            float(recursion_depth) / self.config.max_recursion_depth,
            float(tool_calls_used) / self.config.tool_budget,
            float(recursion_depth >= self.config.max_recursion_depth - 1),
        ]])
        recursion_projection = hk.Linear(self.d_model)
        recursion_state = self.recursion_state_encoder(recursion_projection(recursion_features))

        tool_probs, termination_prob, parameters = self.tool_selector(
            encoded_query,
            context_metadata,
            recursion_state,
            temperature=self.config.tool_temperature,
        )

        return tool_probs, termination_prob, parameters, encoded_query

    def synthesize_answer(
        self,
        query_embedding: jnp.ndarray,
        intermediate_results: List[jnp.ndarray],
    ) -> jnp.ndarray:
        if query_embedding.ndim == 3:
            query_embedding = query_embedding.mean(axis=1)

        encoded_query = self.query_encoder(query_embedding)

        if intermediate_results:
            encoded_results = []
            for result in intermediate_results:
                if result.ndim == 1:
                    result = result[None, :]
                encoded = self.result_encoder(result)
                encoded_results.append(encoded)
            stacked_results = jnp.stack(encoded_results, axis=1)
            aggregated_results = stacked_results.mean(axis=1)
        else:
            aggregated_results = jnp.zeros_like(encoded_query)

        combined = jnp.concatenate([encoded_query, aggregated_results], axis=-1)
        return self.answer_synthesizer(combined)


class RLMOrchestrator:
    def __init__(
        self,
        d_model: int,
        config: Optional[RLMConfig] = None,
        embedding_fn: Optional[Callable[[str], jnp.ndarray]] = None,
        summarize_fn: Optional[Callable[[str, int], str]] = None,
    ):
        self.d_model = d_model
        self.config = config or RLMConfig()
        self.config.validate()

        self.context_store = ContextStore(
            enable_caching=self.config.enable_caching,
            cache_ttl=self.config.cache_ttl_seconds,
        )
        if embedding_fn:
            self.context_store.set_embedding_function(embedding_fn)

        self.tools = ContextTools(self.context_store)
        if summarize_fn:
            self.tools.set_summarize_function(summarize_fn)

        self.recursive_manager = RecursiveCallManager(self.config)

        self._init_fn = None
        self._apply_fn = None
        self._params: Optional[Params] = None
        self._embedding_fn = embedding_fn

    def init(self, rng: jnp.ndarray) -> Params:
        def forward(query_emb, context_len, depth, calls):
            model = RecursiveLanguageModel(self.d_model, self.config)
            return model(query_emb, context_len, depth, calls)

        self._init_fn, self._apply_fn = hk.transform(forward)

        dummy_query = jnp.zeros((1, self.d_model))
        self._params = self._init_fn(rng, dummy_query, 1000, 0, 0)
        return self._params

    def _prepare_query_embedding(self, query: str) -> jnp.ndarray:
        if not self._embedding_fn:
            return jnp.zeros((1, self.d_model))

        query_embedding = self._embedding_fn(query)
        query_embedding = jnp.array(query_embedding) if not isinstance(query_embedding, jnp.ndarray) else query_embedding
        return query_embedding[None, :] if query_embedding.ndim == 1 else query_embedding

    def _should_use_direct_pass(self, context_len: int) -> bool:
        return context_len <= self.config.direct_context_threshold and self.config.fallback_to_direct

    def solve(
        self,
        query: str,
        context: str,
        params: Optional[Params] = None,
        rng: Optional[jnp.ndarray] = None,
    ) -> RLMResult:
        start_time = time.time()
        params = params or self._params
        if params is None:
            raise ValueError("Model not initialized. Call init() first.")

        rng = rng if rng is not None else jax.random.PRNGKey(int(time.time() * 1000))

        context_var = "main_context"
        self.context_store.store(context_var, context, source="input")
        query_embedding = self._prepare_query_embedding(query)

        if self._should_use_direct_pass(len(context)):
            return RLMResult(
                answer=context,
                success=True,
                tool_trace=[{"tool": "direct_pass", "reason": "context_small_enough"}],
                execution_time=time.time() - start_time,
            )

        recursion_context = self.recursive_manager.create_context(query, context_var)

        result = self._solve_recursive(
            query=query,
            query_embedding=query_embedding,
            context_var=context_var,
            params=params,
            rng=rng,
            recursion_context=recursion_context,
        )

        result.execution_time = time.time() - start_time
        return result

    def _run_tool_selection_step(
        self,
        query_embedding: jnp.ndarray,
        context_var: str,
        params: Params,
        rng: jnp.ndarray,
        recursion_context: RecursionContext,
    ) -> tuple:
        context_metadata = self.context_store.get_metadata(context_var)
        context_len = context_metadata.total_length if context_metadata else 0

        rng, subkey = jax.random.split(rng)
        if self._apply_fn is None:
            raise ValueError("Model not initialized")

        tool_probs, term_prob, parameters, _ = self._apply_fn(
            params, subkey,
            query_embedding, context_len,
            recursion_context.depth, recursion_context.tool_calls_used
        )
        return tool_probs, term_prob, parameters, rng

    def _process_tool_result(
        self,
        tool_result: Any,
        intermediate_results: List[jnp.ndarray],
    ) -> None:
        if not hasattr(tool_result, 'data') or tool_result.data is None:
            return

        if isinstance(tool_result.data, str) and self._embedding_fn:
            result_emb = self._embedding_fn(tool_result.data)
            intermediate_results.append(jnp.array(result_emb))
        elif isinstance(tool_result.data, jnp.ndarray):
            intermediate_results.append(tool_result.data)

    def _solve_recursive(
        self,
        query: str,
        query_embedding: jnp.ndarray,
        context_var: str,
        params: Params,
        rng: jnp.ndarray,
        recursion_context: RecursionContext,
    ) -> RLMResult:
        intermediate_results: List[jnp.ndarray] = []
        tool_trace = []

        while recursion_context.can_recurse():
            tool_probs, term_prob, parameters, rng = self._run_tool_selection_step(
                query_embedding, context_var, params, rng, recursion_context
            )

            if float(term_prob.mean()) > 0.8:
                break

            selection = self._select_tool(tool_probs, term_prob, parameters)
            if selection.tool == ToolType.TERMINATE:
                break

            tool_result = self._execute_tool(
                selection, query, context_var, recursion_context, params, rng
            )

            recursion_context.record_tool_call(
                selection.tool.value, selection.parameters, tool_result
            )

            tool_trace.append({
                "tool": selection.tool.value,
                "params": selection.parameters,
                "confidence": selection.confidence,
                "success": getattr(tool_result, 'success', True),
            })

            self._process_tool_result(tool_result, intermediate_results)

        def synthesize_fn(query_emb, results):
            model = RecursiveLanguageModel(self.d_model, self.config)
            return model.synthesize_answer(query_emb, results)

        _, synth_apply = hk.transform(synthesize_fn)
        rng, subkey = jax.random.split(rng)

        synth_results = intermediate_results if intermediate_results else []
        final_answer = synth_apply(params, subkey, query_embedding, synth_results)

        return RLMResult(
            answer=final_answer,
            success=True,
            tool_trace=tool_trace,
            total_tool_calls=recursion_context.tool_calls_used,
            max_recursion_depth_reached=recursion_context.depth,
            intermediate_results=intermediate_results,
            tokens_used=sum(r.tokens_used for r in self.tools.history),
        )

    def _select_tool(
        self,
        tool_probs: jnp.ndarray,
        term_prob: jnp.ndarray,
        parameters: Dict[str, jnp.ndarray],
    ) -> ToolSelection:
        tool_list = list(ToolType)

        if float(term_prob.mean()) > 0.8:
            return ToolSelection(
                tool=ToolType.TERMINATE,
                confidence=float(term_prob.mean()),
                parameters={},
            )

        mask = jnp.zeros(len(tool_list))
        for i, tool in enumerate(tool_list):
            if tool in self.config.available_tools:
                mask = mask.at[i].set(1.0)

        masked_probs = tool_probs * mask
        masked_probs = masked_probs / (masked_probs.sum(axis=-1, keepdims=True) + 1e-8)

        tool_idx = int(jnp.argmax(masked_probs, axis=-1).squeeze())
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
            params["length"] = self.config.context_peek_size

        elif tool == ToolType.PARTITION:
            params["strategy"] = self.config.partition_strategy

        elif tool == ToolType.SUMMARIZE:
            params["max_tokens"] = 500

        return params

    def _execute_tool(
        self,
        selection: ToolSelection,
        query: str,
        context_var: str,
        recursion_context: RecursionContext,
        params: Params,
        rng: jnp.ndarray,
    ) -> Any:
        tool = selection.tool
        tool_params = selection.parameters

        if tool == ToolType.PEEK:
            return self.tools.peek(
                context_var,
                start=tool_params.get("start", 0),
                length=tool_params.get("length", self.config.context_peek_size),
            )

        elif tool == ToolType.GREP:
            pattern = self._extract_search_pattern(query)
            return self.tools.grep(context_var, pattern)

        elif tool == ToolType.PARTITION:
            return self.tools.partition(
                context_var,
                strategy=tool_params.get("strategy", self.config.partition_strategy),
            )

        elif tool == ToolType.SUMMARIZE:
            return self.tools.summarize(
                context_var,
                max_tokens=tool_params.get("max_tokens", 500),
            )

        elif tool == ToolType.COUNT:
            pattern = self._extract_search_pattern(query)
            return self.tools.count(context_var, pattern)

        elif tool == ToolType.RECURSIVE_CALL:
            chunk_names = self._get_chunk_names(context_var)
            if not chunk_names:
                partition_result = self.tools.partition(context_var)
                if partition_result.success:
                    chunk_names = partition_result.data

            if chunk_names:
                subcalls = [{"query": query, "context_var": cn} for cn in chunk_names]
                return self._execute_recursive_subcalls(
                    subcalls, recursion_context, params, rng
                )

        return ToolResult.error_result(tool, f"Unsupported tool: {tool}")

    def _extract_search_pattern(self, query: str) -> str:
        import re
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            return quoted[0]

        words = query.lower().split()
        stop_words = {"what", "how", "many", "is", "are", "the", "a", "an", "in", "of", "to", "for"}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return "|".join(keywords[:3]) if keywords else query[:20]

    def _get_chunk_names(self, context_var: str) -> List[str]:
        all_vars = self.context_store.list_variables()
        return [v for v in all_vars if v.startswith(f"{context_var}_chunk_")]

    def _execute_recursive_subcalls(
        self,
        subcalls: List[Dict[str, str]],
        recursion_context: RecursionContext,
        params: Params,
        rng: jnp.ndarray,
    ) -> Dict[str, Any]:
        def solve_subcall(sub_query: str, sub_context_var: str, ctx: RecursionContext) -> Any:
            sub_query_emb = self._prepare_query_embedding(sub_query)
            return self._solve_recursive(
                sub_query, sub_query_emb, sub_context_var,
                params, rng, ctx
            )

        results = self.recursive_manager.spawn_parallel_subcalls(
            recursion_context, subcalls, solve_subcall
        )

        return self.recursive_manager.aggregate_results(results)

    def reset(self) -> None:
        self.context_store.clear()
        self.tools.clear_history()
        self.recursive_manager.clear_cache()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "context_store": self.context_store.stats(),
            "tool_stats": self.tools.get_tool_stats(),
            "recursive_manager": self.recursive_manager.get_stats(),
        }
