from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import time
import threading
import jax.numpy as jnp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.rlm_config import RLMConfig


@dataclass
class RecursionContext:
    depth: int = 0
    max_depth: int = 5
    tool_calls_used: int = 0
    tool_budget: int = 20
    parent_query: Optional[str] = None
    parent_context_var: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    timeout: float = 60.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def can_recurse(self) -> bool:
        if self.depth >= self.max_depth:
            return False
        if self.tool_calls_used >= self.tool_budget:
            return False
        if time.time() - self.start_time > self.timeout:
            return False
        return True

    def remaining_budget(self) -> int:
        return max(0, self.tool_budget - self.tool_calls_used)

    def remaining_depth(self) -> int:
        return max(0, self.max_depth - self.depth)

    def child_context(self, query: str, context_var: str) -> "RecursionContext":
        return RecursionContext(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            tool_calls_used=self.tool_calls_used,
            tool_budget=self.tool_budget,
            parent_query=query,
            parent_context_var=context_var,
            trace=self.trace.copy(),
            start_time=self.start_time,
            timeout=self.timeout,
        )

    def record_tool_call(self, tool: str, params: Dict[str, Any], result: Any) -> None:
        self.tool_calls_used += 1
        self.trace.append({
            "depth": self.depth,
            "tool": tool,
            "params": params,
            "success": getattr(result, 'success', True),
            "timestamp": time.time(),
        })


@dataclass
class SubCallResult:
    query: str
    context_var: str
    result: Any
    success: bool
    error: Optional[str] = None
    recursion_context: Optional[RecursionContext] = None
    execution_time: float = 0.0


class RecursiveCallManager:
    def __init__(self, config: RLMConfig):
        self.config = config
        self._active_contexts: Dict[str, RecursionContext] = {}
        self._results_cache: Dict[str, SubCallResult] = {}

    def create_context(
        self,
        query: str,
        context_var: str,
        timeout: float = 60.0,
    ) -> RecursionContext:
        context = RecursionContext(
            depth=0,
            max_depth=self.config.max_recursion_depth,
            tool_budget=self.config.tool_budget,
            parent_query=query,
            parent_context_var=context_var,
            timeout=timeout,
        )
        context_id = f"{id(context)}_{time.time()}"
        self._active_contexts[context_id] = context
        return context

    def spawn_subcall(
        self,
        parent_context: RecursionContext,
        sub_query: str,
        sub_context_var: str,
        solve_fn: Callable[[str, str, RecursionContext], Any],
    ) -> SubCallResult:
        if not parent_context.can_recurse():
            return SubCallResult(
                query=sub_query,
                context_var=sub_context_var,
                result=None,
                success=False,
                error="Recursion limit reached",
            )

        cache_key = f"{sub_query}:{sub_context_var}"
        if self.config.enable_caching and cache_key in self._results_cache:
            return self._results_cache[cache_key]

        child_context = parent_context.child_context(sub_query, sub_context_var)
        start_time = time.time()

        try:
            result = solve_fn(sub_query, sub_context_var, child_context)
            sub_result = SubCallResult(
                query=sub_query,
                context_var=sub_context_var,
                result=result,
                success=True,
                recursion_context=child_context,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            sub_result = SubCallResult(
                query=sub_query,
                context_var=sub_context_var,
                result=None,
                success=False,
                error=str(e),
                recursion_context=child_context,
                execution_time=time.time() - start_time,
            )

        parent_context.tool_calls_used = child_context.tool_calls_used
        parent_context.trace.extend(child_context.trace[len(parent_context.trace):])

        if self.config.enable_caching:
            self._results_cache[cache_key] = sub_result

        return sub_result

    def spawn_parallel_subcalls(
        self,
        parent_context: RecursionContext,
        subcalls: List[Dict[str, str]],
        solve_fn: Callable[[str, str, RecursionContext], Any],
    ) -> List[SubCallResult]:
        if not self.config.parallel_subcalls:
            return [
                self.spawn_subcall(parent_context, sc["query"], sc["context_var"], solve_fn)
                for sc in subcalls
            ]

        max_parallel = min(self.config.max_parallel_subcalls, len(subcalls))
        results = []

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for sc in subcalls:
                if not parent_context.can_recurse():
                    results.append(SubCallResult(
                        query=sc["query"],
                        context_var=sc["context_var"],
                        result=None,
                        success=False,
                        error="Recursion limit reached",
                    ))
                    continue

                child_context = parent_context.child_context(sc["query"], sc["context_var"])
                future = executor.submit(
                    self._execute_subcall,
                    solve_fn,
                    sc["query"],
                    sc["context_var"],
                    child_context,
                )
                futures[future] = (sc, child_context)

            for future in as_completed(futures):
                sc, child_context = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    with parent_context._lock:
                        parent_context.tool_calls_used = max(
                            parent_context.tool_calls_used,
                            child_context.tool_calls_used
                        )
                except Exception as e:
                    results.append(SubCallResult(
                        query=sc["query"],
                        context_var=sc["context_var"],
                        result=None,
                        success=False,
                        error=str(e),
                    ))

        return results

    def _execute_subcall(
        self,
        solve_fn: Callable,
        query: str,
        context_var: str,
        context: RecursionContext,
    ) -> SubCallResult:
        start_time = time.time()
        try:
            result = solve_fn(query, context_var, context)
            return SubCallResult(
                query=query,
                context_var=context_var,
                result=result,
                success=True,
                recursion_context=context,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return SubCallResult(
                query=query,
                context_var=context_var,
                result=None,
                success=False,
                error=str(e),
                recursion_context=context,
                execution_time=time.time() - start_time,
            )

    def aggregate_results(
        self,
        results: List[SubCallResult],
        query_embedding: Optional[jnp.ndarray] = None,
        result_embeddings: Optional[List[jnp.ndarray]] = None,
    ) -> Dict[str, Any]:
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if not successful:
            return {
                "aggregated_result": None,
                "success": False,
                "num_successful": 0,
                "num_failed": len(failed),
                "errors": [r.error for r in failed],
            }

        if self.config.aggregation_strategy == "weighted_mean" and result_embeddings and query_embedding is not None:
            aggregated = self._weighted_aggregate(
                query_embedding, result_embeddings
            )
        elif self.config.aggregation_strategy == "concat":
            aggregated = self._concat_aggregate(successful)
        else:
            aggregated = self._simple_aggregate(successful)

        return {
            "aggregated_result": aggregated,
            "success": True,
            "num_successful": len(successful),
            "num_failed": len(failed),
            "individual_results": [r.result for r in successful],
        }

    def _simple_aggregate(self, results: List[SubCallResult]) -> Any:
        return [r.result for r in results]

    def _concat_aggregate(self, results: List[SubCallResult]) -> str:
        parts = []
        for r in results:
            if isinstance(r.result, str):
                parts.append(r.result)
            elif isinstance(r.result, dict) and "answer" in r.result:
                parts.append(str(r.result["answer"]))
            else:
                parts.append(str(r.result))
        return "\n".join(parts)

    def _weighted_aggregate(
        self,
        query_embedding: jnp.ndarray,
        result_embeddings: List[jnp.ndarray],
    ) -> jnp.ndarray:
        query_flat = np.asarray(query_embedding).flatten()
        weights = []

        for emb in result_embeddings:
            emb_flat = np.asarray(emb).flatten()
            similarity = np.dot(query_flat, emb_flat) / (
                np.linalg.norm(query_flat) * np.linalg.norm(emb_flat) + 1e-8
            )
            weights.append(max(0, similarity))

        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        stacked = np.stack([np.asarray(e) for e in result_embeddings])
        aggregated = np.tensordot(weights, stacked, axes=([0], [0]))

        return jnp.array(aggregated)

    def clear_cache(self) -> None:
        self._results_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_contexts": len(self._active_contexts),
            "cached_results": len(self._results_cache),
        }
