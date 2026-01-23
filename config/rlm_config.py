from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class PartitionStrategy(str, Enum):
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class ToolType(str, Enum):
    PEEK = "peek"
    GREP = "grep"
    PARTITION = "partition"
    SUMMARIZE = "summarize"
    COUNT = "count"
    FILTER = "filter"
    RECURSIVE_CALL = "recursive_call"
    TERMINATE = "terminate"


@dataclass
class RLMConfig:
    enabled: bool = True
    max_recursion_depth: int = 5
    context_peek_size: int = 2000
    tool_budget: int = 20
    partition_strategy: PartitionStrategy = PartitionStrategy.SEMANTIC
    min_chunk_size: int = 100
    max_chunk_size: int = 4000
    semantic_similarity_threshold: float = 0.7
    auto_partition_threshold: int = 8000
    enable_caching: bool = True
    cache_ttl_seconds: float = 300.0
    parallel_subcalls: bool = True
    max_parallel_subcalls: int = 4
    aggregation_strategy: str = "weighted_mean"
    fallback_to_direct: bool = True
    direct_context_threshold: int = 2000
    tool_temperature: float = 0.1
    available_tools: List[ToolType] = field(default_factory=lambda: [
        ToolType.PEEK,
        ToolType.GREP,
        ToolType.PARTITION,
        ToolType.SUMMARIZE,
        ToolType.RECURSIVE_CALL,
        ToolType.TERMINATE,
    ])

    @classmethod
    def minimal(cls) -> "RLMConfig":
        return cls(
            max_recursion_depth=3,
            tool_budget=10,
            max_parallel_subcalls=2,
        )

    @classmethod
    def aggressive(cls) -> "RLMConfig":
        return cls(
            max_recursion_depth=8,
            tool_budget=50,
            max_parallel_subcalls=8,
            context_peek_size=4000,
        )

    def validate(self) -> None:
        if self.max_recursion_depth < 1:
            raise ValueError("max_recursion_depth must be >= 1")
        if self.tool_budget < 1:
            raise ValueError("tool_budget must be >= 1")
        if self.context_peek_size < 100:
            raise ValueError("context_peek_size must be >= 100")
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be < max_chunk_size")
        if not 0.0 <= self.tool_temperature <= 2.0:
            raise ValueError("tool_temperature must be in [0.0, 2.0]")
