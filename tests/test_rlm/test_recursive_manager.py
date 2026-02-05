import pytest
import time
from unittest.mock import MagicMock

from core.rlm.recursive_manager import (
    RecursiveCallManager,
    RecursionContext,
    SubCallResult,
)
from config.rlm_config import RLMConfig


class TestRecursionContext:
    def test_default_values(self):
        ctx = RecursionContext()
        assert ctx.depth == 0
        assert ctx.max_depth == 5
        assert ctx.tool_calls_used == 0
        assert ctx.tool_budget == 20
        assert ctx.can_recurse() is True

    def test_can_recurse_depth_limit(self):
        ctx = RecursionContext(depth=5, max_depth=5)
        assert ctx.can_recurse() is False

    def test_can_recurse_budget_limit(self):
        ctx = RecursionContext(tool_calls_used=20, tool_budget=20)
        assert ctx.can_recurse() is False

    def test_can_recurse_timeout(self):
        ctx = RecursionContext(start_time=time.time() - 100, timeout=60)
        assert ctx.can_recurse() is False

    def test_remaining_budget(self):
        ctx = RecursionContext(tool_calls_used=5, tool_budget=20)
        assert ctx.remaining_budget() == 15

    def test_remaining_depth(self):
        ctx = RecursionContext(depth=3, max_depth=5)
        assert ctx.remaining_depth() == 2

    def test_child_context(self):
        parent = RecursionContext(depth=1, max_depth=5, tool_calls_used=3, tool_budget=20)
        child = parent.child_context("sub query", "sub_context")

        assert child.depth == 2
        assert child.max_depth == 5
        assert child.tool_calls_used == 3
        assert child.parent_query == "sub query"
        assert child.parent_context_var == "sub_context"

    def test_record_tool_call(self):
        ctx = RecursionContext()
        mock_result = MagicMock()
        mock_result.success = True

        ctx.record_tool_call("peek", {"start": 0}, mock_result)

        assert ctx.tool_calls_used == 1
        assert len(ctx.trace) == 1
        assert ctx.trace[0]["tool"] == "peek"
        assert ctx.trace[0]["success"] is True


class TestSubCallResult:
    def test_success_result(self):
        result = SubCallResult(
            query="test query",
            context_var="test_var",
            result={"answer": "42"},
            success=True,
            execution_time=0.5,
        )
        assert result.success is True
        assert result.result == {"answer": "42"}

    def test_error_result(self):
        result = SubCallResult(
            query="test query",
            context_var="test_var",
            result=None,
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestRecursiveCallManager:
    @pytest.fixture
    def manager(self):
        config = RLMConfig(
            max_recursion_depth=3,
            tool_budget=10,
            parallel_subcalls=False,
        )
        return RecursiveCallManager(config)

    def test_create_context(self, manager):
        ctx = manager.create_context("query", "context_var", timeout=30.0)
        assert ctx.depth == 0
        assert ctx.max_depth == 3
        assert ctx.tool_budget == 10
        assert abs(ctx.timeout - 30.0) < 0.001

    def test_spawn_subcall_success(self, manager):
        parent_ctx = manager.create_context("parent query", "parent_var")

        def solve_fn(query, context_var, ctx):
            return {"answer": f"solved {query}"}

        result = manager.spawn_subcall(
            parent_ctx, "sub query", "sub_var", solve_fn
        )

        assert result.success is True
        assert result.result == {"answer": "solved sub query"}
        assert result.execution_time > 0

    def test_spawn_subcall_recursion_limit(self, manager):
        parent_ctx = RecursionContext(depth=3, max_depth=3)

        def solve_fn(query, context_var, ctx):
            return {"answer": "should not reach"}

        result = manager.spawn_subcall(
            parent_ctx, "sub query", "sub_var", solve_fn
        )

        assert result.success is False
        assert "limit" in result.error.lower()

    def test_spawn_subcall_error_handling(self, manager):
        parent_ctx = manager.create_context("parent query", "parent_var")

        def solve_fn(query, context_var, ctx):
            raise ValueError("Intentional error")

        result = manager.spawn_subcall(
            parent_ctx, "sub query", "sub_var", solve_fn
        )

        assert result.success is False
        assert "Intentional error" in result.error

    def test_spawn_subcall_tool_count_propagation(self, manager):
        parent_ctx = manager.create_context("parent query", "parent_var")
        initial_count = parent_ctx.tool_calls_used

        def solve_fn(query, context_var, ctx):
            ctx.tool_calls_used += 5
            return {"answer": "done"}

        manager.spawn_subcall(parent_ctx, "sub query", "sub_var", solve_fn)

        assert parent_ctx.tool_calls_used == initial_count + 5

    def test_spawn_parallel_subcalls_sequential_mode(self, manager):
        parent_ctx = manager.create_context("parent query", "parent_var")
        subcalls = [
            {"query": "q1", "context_var": "v1"},
            {"query": "q2", "context_var": "v2"},
            {"query": "q3", "context_var": "v3"},
        ]

        def solve_fn(query, context_var, ctx):
            return {"answer": f"solved {query}"}

        results = manager.spawn_parallel_subcalls(
            parent_ctx, subcalls, solve_fn
        )

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_spawn_parallel_subcalls_parallel_mode(self):
        config = RLMConfig(
            max_recursion_depth=3,
            tool_budget=10,
            parallel_subcalls=True,
            max_parallel_subcalls=2,
        )
        manager = RecursiveCallManager(config)

        parent_ctx = manager.create_context("parent query", "parent_var")
        subcalls = [
            {"query": "q1", "context_var": "v1"},
            {"query": "q2", "context_var": "v2"},
        ]

        def solve_fn(query, context_var, ctx):
            time.sleep(0.01)
            return {"answer": f"solved {query}"}

        results = manager.spawn_parallel_subcalls(
            parent_ctx, subcalls, solve_fn
        )

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_aggregate_results_simple(self, manager):
        results = [
            SubCallResult("q1", "v1", {"a": 1}, True),
            SubCallResult("q2", "v2", {"a": 2}, True),
        ]

        aggregated = manager.aggregate_results(results)

        assert aggregated["success"] is True
        assert aggregated["num_successful"] == 2
        assert aggregated["num_failed"] == 0

    def test_aggregate_results_with_failures(self, manager):
        results = [
            SubCallResult("q1", "v1", {"a": 1}, True),
            SubCallResult("q2", "v2", None, False, error="Failed"),
        ]

        aggregated = manager.aggregate_results(results)

        assert aggregated["success"] is True
        assert aggregated["num_successful"] == 1
        assert aggregated["num_failed"] == 1

    def test_aggregate_results_all_failed(self, manager):
        results = [
            SubCallResult("q1", "v1", None, False, error="Failed 1"),
            SubCallResult("q2", "v2", None, False, error="Failed 2"),
        ]

        aggregated = manager.aggregate_results(results)

        assert aggregated["success"] is False
        assert aggregated["num_failed"] == 2
        assert len(aggregated["errors"]) == 2

    def test_caching(self):
        config = RLMConfig(enable_caching=True)
        manager = RecursiveCallManager(config)

        parent_ctx = manager.create_context("parent query", "parent_var")
        call_count = 0

        def solve_fn(query, context_var, ctx):
            nonlocal call_count
            call_count += 1
            return {"answer": "cached"}

        manager.spawn_subcall(parent_ctx, "cached_query", "cached_var", solve_fn)
        manager.spawn_subcall(parent_ctx, "cached_query", "cached_var", solve_fn)

        assert call_count == 1

    def test_clear_cache(self):
        config = RLMConfig(enable_caching=True)
        manager = RecursiveCallManager(config)

        parent_ctx = manager.create_context("parent query", "parent_var")

        def solve_fn(query, context_var, ctx):
            return {"answer": "result"}

        manager.spawn_subcall(parent_ctx, "q", "v", solve_fn)
        assert len(manager._results_cache) == 1

        manager.clear_cache()
        assert len(manager._results_cache) == 0

    def test_stats(self, manager):
        stats = manager.get_stats()
        assert "active_contexts" in stats
        assert "cached_results" in stats
