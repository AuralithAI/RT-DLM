import pytest
import time

from core.rlm.context_store import ContextStore
from core.rlm.context_tools import ContextTools, ToolResult
from config.rlm_config import ToolType, PartitionStrategy


class TestToolResult:
    def test_success_result(self):
        result = ToolResult.success_result(
            ToolType.PEEK,
            "some data",
            execution_time=0.1,
            extra_meta="value",
        )
        assert result.success is True
        assert result.tool == ToolType.PEEK
        assert result.data == "some data"
        assert abs(result.execution_time - 0.1) < 0.001
        assert result.metadata["extra_meta"] == "value"

    def test_error_result(self):
        result = ToolResult.error_result(ToolType.GREP, "Not found")
        assert result.success is False
        assert result.tool == ToolType.GREP
        assert result.error == "Not found"
        assert result.data is None


class TestContextTools:
    @pytest.fixture
    def tools(self):
        store = ContextStore()
        store.store("test_content", "Hello World. This is a test document with multiple sentences. It contains various words.")
        return ContextTools(store)

    def test_peek_success(self, tools):
        result = tools.peek("test_content", start=0, length=20)
        assert result.success is True
        assert result.tool == ToolType.PEEK
        assert len(result.data) <= 20
        assert result.metadata["start"] == 0
        assert result.tokens_used > 0

    def test_peek_nonexistent(self, tools):
        result = tools.peek("nonexistent")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_grep_success(self, tools):
        result = tools.grep("test_content", "test")
        assert result.success is True
        assert result.tool == ToolType.GREP
        assert result.metadata["num_matches"] >= 1

    def test_grep_regex(self, tools):
        tools._store.store("numbers", "ID: 123, ID: 456, ID: 789")
        result = tools.grep("numbers", r"\d{3}", regex=True)
        assert result.success is True
        assert result.metadata["num_matches"] == 3

    def test_grep_max_results(self, tools):
        content = "\n".join([f"Line {i}: match" for i in range(100)])
        tools._store.store("many_matches", content)
        result = tools.grep("many_matches", "match", max_results=10)
        assert result.success is True
        assert len(result.data) == 10
        assert result.metadata["truncated"] is True

    def test_partition_success(self, tools):
        tools._store.store("to_partition", "A" * 500)
        result = tools.partition("to_partition", chunk_size=100)
        assert result.success is True
        assert result.tool == ToolType.PARTITION
        assert result.metadata["num_chunks"] > 1

    def test_partition_with_strategy(self, tools):
        content = "Para 1.\n\nPara 2.\n\nPara 3."
        tools._store.store("paragraphs", content)
        result = tools.partition(
            "paragraphs",
            strategy=PartitionStrategy.PARAGRAPH,
            chunk_size=1000,
        )
        assert result.success is True

    def test_summarize_extractive(self, tools):
        long_content = " ".join(["This is sentence number {}.".format(i) for i in range(50)])
        tools._store.store("long_doc", long_content)

        result = tools.summarize("long_doc", max_tokens=100)
        assert result.success is True
        assert result.tool == ToolType.SUMMARIZE
        assert "summary" in result.data
        assert "summary_var" in result.data
        assert result.metadata["compression_ratio"] > 1

    def test_summarize_with_custom_function(self, tools):
        def custom_summarize(text, max_tokens):
            return f"Custom summary of {len(text)} chars"

        tools.set_summarize_function(custom_summarize)
        tools._store.store("to_summarize", "Some long content here")

        result = tools.summarize("to_summarize", max_tokens=50)
        assert result.success is True
        assert "Custom summary" in result.data["summary"]

    def test_summarize_nonexistent(self, tools):
        result = tools.summarize("nonexistent")
        assert result.success is False

    def test_count_success(self, tools):
        tools._store.store("count_doc", "apple banana apple cherry apple")
        result = tools.count("count_doc", "apple")
        assert result.success is True
        assert result.tool == ToolType.COUNT
        assert result.data == 3

    def test_count_regex(self, tools):
        tools._store.store("count_doc", "123 abc 456 def 789")
        result = tools.count("count_doc", r"\d+", regex=True)
        assert result.success is True
        assert result.data == 3

    def test_filter_success(self, tools):
        tools._store.store("filter_doc", "apple\nbanana\napricot\ncherry")
        result = tools.filter("filter_doc", lambda line: line.startswith("a"))
        assert result.success is True
        assert result.tool == ToolType.FILTER
        assert result.data["num_lines"] == 2

    def test_filter_custom_output(self, tools):
        tools._store.store("filter_doc", "line1\nline2\nline3")
        result = tools.filter(
            "filter_doc",
            lambda line: "2" in line,
            output_var="custom_output",
        )
        assert result.success is True
        assert result.data["output_var"] == "custom_output"
        assert "custom_output" in tools._store.list_variables()

    def test_execute_dispatch(self, tools):
        result = tools.execute(ToolType.PEEK, context_var="test_content", length=10)
        assert result.success is True
        assert result.tool == ToolType.PEEK

    def test_execute_unsupported(self, tools):
        result = tools.execute(ToolType.RECURSIVE_CALL, context_var="test")
        assert result.success is False

    def test_history_tracking(self, tools):
        tools.clear_history()
        tools.peek("test_content")
        tools.grep("test_content", "test")
        tools.count("test_content", "is")

        history = tools.history
        assert len(history) == 3
        assert history[0].tool == ToolType.PEEK
        assert history[1].tool == ToolType.GREP
        assert history[2].tool == ToolType.COUNT

    def test_tool_stats(self, tools):
        tools.clear_history()
        tools.peek("test_content")
        tools.peek("test_content")
        tools.grep("test_content", "test")

        stats = tools.get_tool_stats()
        assert stats["peek"]["calls"] == 2
        assert stats["peek"]["successes"] == 2
        assert stats["grep"]["calls"] == 1

    def test_execution_time_tracking(self, tools):
        result = tools.peek("test_content")
        assert result.execution_time >= 0
