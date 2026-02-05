import numpy as np

from core.rlm.context_store import ContextStore, ContextVariable, ContextMetadata
from config.rlm_config import PartitionStrategy


class TestContextMetadata:
    def test_default_values(self):
        metadata = ContextMetadata()
        assert metadata.source == ""
        assert metadata.content_type == "text"
        assert metadata.total_length == 0
        assert metadata.access_count == 0
        assert metadata.tags == []

    def test_custom_values(self):
        metadata = ContextMetadata(
            source="test_source",
            total_length=1000,
            tags=["tag1", "tag2"],
        )
        assert metadata.source == "test_source"
        assert metadata.total_length == 1000
        assert metadata.tags == ["tag1", "tag2"]


class TestContextVariable:
    def test_length(self):
        var = ContextVariable(
            name="test",
            content="Hello World",
            metadata=ContextMetadata(),
        )
        assert len(var) == 11

    def test_content_hash(self):
        var = ContextVariable(
            name="test",
            content="Hello World",
            metadata=ContextMetadata(),
        )
        hash1 = var.content_hash()
        assert len(hash1) == 16

        var2 = ContextVariable(
            name="test2",
            content="Hello World",
            metadata=ContextMetadata(),
        )
        assert var.content_hash() == var2.content_hash()


class TestContextStore:
    def test_store_and_get(self):
        store = ContextStore()
        var = store.store("test_var", "Test content", source="unit_test")

        assert var.name == "test_var"
        assert var.content == "Test content"
        assert var.metadata.source == "unit_test"

        retrieved = store.get("test_var")
        assert retrieved is not None
        assert retrieved.content == "Test content"

    def test_get_nonexistent(self):
        store = ContextStore()
        assert store.get("nonexistent") is None

    def test_peek(self):
        store = ContextStore()
        long_content = "A" * 5000
        store.store("long_var", long_content)

        peeked = store.peek("long_var", start=0, length=100)
        assert peeked is not None
        assert len(peeked) == 100
        assert peeked == "A" * 100

        peeked = store.peek("long_var", start=4900, length=200)
        assert peeked is not None
        assert len(peeked) == 100

    def test_peek_nonexistent(self):
        store = ContextStore()
        assert store.peek("nonexistent") is None

    def test_delete(self):
        store = ContextStore()
        store.store("to_delete", "content")
        assert "to_delete" in store

        result = store.delete("to_delete")
        assert result is True
        assert "to_delete" not in store

        result = store.delete("nonexistent")
        assert result is False

    def test_list_variables(self):
        store = ContextStore()
        store.store("var1", "content1")
        store.store("var2", "content2")
        store.store("var3", "content3")

        variables = store.list_variables()
        assert len(variables) == 3
        assert "var1" in variables
        assert "var2" in variables
        assert "var3" in variables

    def test_partition_fixed_size(self):
        store = ContextStore()
        content = "A" * 1000
        store.store("to_partition", content)

        chunk_names = store.partition(
            "to_partition",
            strategy=PartitionStrategy.FIXED_SIZE,
            chunk_size=300,
            overlap=50,
        )

        assert len(chunk_names) > 1
        for name in chunk_names:
            assert name in store

    def test_partition_paragraph(self):
        store = ContextStore()
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        store.store("paragraphs", content)

        chunk_names = store.partition(
            "paragraphs",
            strategy=PartitionStrategy.PARAGRAPH,
            chunk_size=1000,
        )

        assert len(chunk_names) >= 1

    def test_grep_simple(self):
        store = ContextStore()
        content = """Line 1: Hello World
Line 2: Foo Bar
Line 3: Hello Again
Line 4: Baz Qux"""
        store.store("grep_test", content)

        results = store.grep("grep_test", "Hello")
        assert len(results) == 2
        assert results[0]["line_number"] == 0
        assert results[1]["line_number"] == 2

    def test_grep_regex(self):
        store = ContextStore()
        content = """ID: 12345
ID: 67890
ID: ABCDE"""
        store.store("grep_test", content)

        results = store.grep("grep_test", r"\d{5}", regex=True)
        assert len(results) == 2

    def test_grep_with_context(self):
        store = ContextStore()
        content = """Line 1
Line 2
Target Line
Line 4
Line 5"""
        store.store("grep_test", content)

        results = store.grep("grep_test", "Target", context_lines=1)
        assert len(results) == 1
        assert "Line 2" in results[0]["context"]
        assert "Line 4" in results[0]["context"]

    def test_count(self):
        store = ContextStore()
        content = "apple banana apple cherry apple"
        store.store("count_test", content)

        count = store.count("count_test", "apple")
        assert count == 3

    def test_count_regex(self):
        store = ContextStore()
        content = "123 abc 456 def 789"
        store.store("count_test", content)

        count = store.count("count_test", r"\d+", regex=True)
        assert count == 3

    def test_eviction(self):
        store = ContextStore(max_variables=3)
        store.store("var1", "content1")
        store.store("var2", "content2")
        store.store("var3", "content3")

        assert len(store) == 3

        store.store("var4", "content4")
        assert len(store) == 3
        assert "var1" not in store

    def test_size_eviction(self):
        store = ContextStore(max_total_size=1000)
        store.store("var1", "A" * 400)
        store.store("var2", "B" * 400)

        assert len(store) == 2

        store.store("var3", "C" * 400)
        assert len(store) <= 2

    def test_access_count_tracking(self):
        store = ContextStore()
        store.store("accessed", "content")

        for _ in range(5):
            store.get("accessed")

        var = store.get("accessed")
        assert var is not None
        assert var.metadata.access_count == 6

    def test_stats(self):
        store = ContextStore()
        store.store("var1", "content1")
        store.store("var2", "content2")

        stats = store.stats()
        assert stats["num_variables"] == 2
        assert stats["total_size"] == len("content1") + len("content2")

    def test_clear(self):
        store = ContextStore()
        store.store("var1", "content1")
        store.store("var2", "content2")

        store.clear()
        assert len(store) == 0

    def test_iteration(self):
        store = ContextStore()
        store.store("var1", "content1")
        store.store("var2", "content2")

        vars_list = list(store)
        assert len(vars_list) == 2

    def test_embedding_function(self):
        store = ContextStore()

        def mock_embedding(text):
            return np.random.randn(384)

        store.set_embedding_function(mock_embedding)
        var = store.store("with_embedding", "Test content")

        assert var.embedding is not None
        assert var.embedding.shape == (384,)

    def test_parent_child_relationship(self):
        store = ContextStore()
        store.store("parent", "Parent content")

        chunk_names = store.partition("parent", chunk_size=5)

        for name in chunk_names:
            var = store.get(name)
            assert var is not None
            assert var.metadata.parent_var == "parent"
            assert var.metadata.chunk_index is not None
