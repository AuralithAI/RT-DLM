from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import re
import time

from config.rlm_config import ToolType, PartitionStrategy
from core.rlm.context_store import ContextStore


@dataclass
class ToolResult:
    tool: ToolType
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls, tool: ToolType, data: Any, execution_time: float = 0.0, **metadata
    ) -> "ToolResult":
        return cls(
            tool=tool,
            success=True,
            data=data,
            execution_time=execution_time,
            metadata=metadata,
        )

    @classmethod
    def error_result(cls, tool: ToolType, error: str) -> "ToolResult":
        return cls(tool=tool, success=False, data=None, error=error)


class ContextTools:
    def __init__(self, context_store: ContextStore):
        self._store = context_store
        self._summarize_fn: Optional[Callable[[str, int], str]] = None
        self._tool_history: List[ToolResult] = []

    def set_summarize_function(self, fn: Callable[[str, int], str]) -> None:
        self._summarize_fn = fn

    @property
    def history(self) -> List[ToolResult]:
        return self._tool_history.copy()

    def clear_history(self) -> None:
        self._tool_history.clear()

    def peek(
        self,
        context_var: str,
        start: int = 0,
        length: int = 2000,
    ) -> ToolResult:
        start_time = time.time()

        content = self._store.peek(context_var, start, length)
        if content is None:
            result = ToolResult.error_result(
                ToolType.PEEK, f"Variable '{context_var}' not found"
            )
        else:
            metadata = self._store.get_metadata(context_var)
            total_len = metadata.total_length if metadata else 0
            has_more = start + length < total_len
            result = ToolResult.success_result(
                ToolType.PEEK,
                content,
                time.time() - start_time,
                start=start,
                length=len(content),
                total_length=total_len,
                has_more=has_more,
            )
            result.tokens_used = len(content) // 4

        self._tool_history.append(result)
        return result

    def grep(
        self,
        context_var: str,
        pattern: str,
        regex: bool = False,
        context_lines: int = 0,
        max_results: int = 50,
    ) -> ToolResult:
        start_time = time.time()

        try:
            matches = self._store.grep(context_var, pattern, regex, context_lines)
            matches = matches[:max_results]
            result = ToolResult.success_result(
                ToolType.GREP,
                matches,
                time.time() - start_time,
                pattern=pattern,
                regex=regex,
                num_matches=len(matches),
                truncated=len(matches) >= max_results,
            )
            result.tokens_used = sum(len(m.get('context', '')) for m in matches) // 4
        except Exception as e:
            result = ToolResult.error_result(ToolType.GREP, str(e))

        self._tool_history.append(result)
        return result

    def partition(
        self,
        context_var: str,
        strategy: PartitionStrategy = PartitionStrategy.FIXED_SIZE,
        chunk_size: int = 2000,
        overlap: int = 200,
    ) -> ToolResult:
        start_time = time.time()

        try:
            chunk_names = self._store.partition(
                context_var, strategy, chunk_size, overlap
            )
            result = ToolResult.success_result(
                ToolType.PARTITION,
                chunk_names,
                time.time() - start_time,
                strategy=strategy.value,
                num_chunks=len(chunk_names),
            )
        except Exception as e:
            result = ToolResult.error_result(ToolType.PARTITION, str(e))

        self._tool_history.append(result)
        return result

    def summarize(
        self,
        context_var: str,
        max_tokens: int = 500,
    ) -> ToolResult:
        start_time = time.time()

        var = self._store.get(context_var)
        if var is None:
            result = ToolResult.error_result(
                ToolType.SUMMARIZE, f"Variable '{context_var}' not found"
            )
            self._tool_history.append(result)
            return result

        if self._summarize_fn is None:
            summary = self._extractive_summarize(var.content, max_tokens)
        else:
            try:
                summary = self._summarize_fn(var.content, max_tokens)
            except Exception as e:
                result = ToolResult.error_result(ToolType.SUMMARIZE, str(e))
                self._tool_history.append(result)
                return result

        summary_var_name = f"{context_var}_summary"
        self._store.store(
            name=summary_var_name,
            content=summary,
            source=f"summary of {context_var}",
            parent_var=context_var,
        )

        result = ToolResult.success_result(
            ToolType.SUMMARIZE,
            {"summary": summary, "summary_var": summary_var_name},
            time.time() - start_time,
            original_length=len(var.content),
            summary_length=len(summary),
            compression_ratio=len(var.content) / max(len(summary), 1),
        )
        result.tokens_used = max_tokens

        self._tool_history.append(result)
        return result

    def count(
        self,
        context_var: str,
        pattern: str,
        regex: bool = False,
    ) -> ToolResult:
        start_time = time.time()

        count = self._store.count(context_var, pattern, regex)
        if count == 0:
            var = self._store.get(context_var)
            if var is None:
                result = ToolResult.error_result(
                    ToolType.COUNT, f"Variable '{context_var}' not found"
                )
                self._tool_history.append(result)
                return result

        result = ToolResult.success_result(
            ToolType.COUNT,
            count,
            time.time() - start_time,
            pattern=pattern,
            regex=regex,
        )

        self._tool_history.append(result)
        return result

    def filter(
        self,
        context_var: str,
        condition: Callable[[str], bool],
        output_var: Optional[str] = None,
    ) -> ToolResult:
        start_time = time.time()

        var = self._store.get(context_var)
        if var is None:
            result = ToolResult.error_result(
                ToolType.FILTER, f"Variable '{context_var}' not found"
            )
            self._tool_history.append(result)
            return result

        try:
            lines = var.content.split('\n')
            filtered_lines = [line for line in lines if condition(line)]
            filtered_content = '\n'.join(filtered_lines)

            if output_var is None:
                output_var = f"{context_var}_filtered"

            self._store.store(
                name=output_var,
                content=filtered_content,
                source=f"filtered from {context_var}",
                parent_var=context_var,
            )

            result = ToolResult.success_result(
                ToolType.FILTER,
                {"output_var": output_var, "num_lines": len(filtered_lines)},
                time.time() - start_time,
                original_lines=len(lines),
                filtered_lines=len(filtered_lines),
            )
        except Exception as e:
            result = ToolResult.error_result(ToolType.FILTER, str(e))

        self._tool_history.append(result)
        return result

    def _extractive_summarize(self, content: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        sentences = re.split(r'(?<=[.!?])\s+', content)

        if len(content) <= max_chars:
            return content

        scored_sentences = []
        for i, sent in enumerate(sentences):
            position_score = 1.0 / (i + 1)
            length_score = min(len(sent) / 100, 1.0)
            keyword_score = self._keyword_score(sent, content)
            total_score = position_score * 0.3 + length_score * 0.2 + keyword_score * 0.5
            scored_sentences.append((sent, total_score))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        summary_sentences = []
        current_length = 0
        for sent, _ in scored_sentences:
            if current_length + len(sent) > max_chars:
                break
            summary_sentences.append(sent)
            current_length += len(sent)

        original_order = []
        for sent in sentences:
            if sent in summary_sentences:
                original_order.append(sent)

        return ' '.join(original_order)

    def _keyword_score(self, sentence: str, full_content: str) -> float:
        words = re.findall(r'\b\w+\b', sentence.lower())
        content_words = re.findall(r'\b\w+\b', full_content.lower())
        
        if not words or not content_words:
            return 0.0

        word_freq = {}
        for word in content_words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        score = sum(word_freq.get(w, 0) for w in words if len(w) > 3)
        return min(score / (len(words) + 1), 1.0)

    def execute(
        self,
        tool: ToolType,
        **kwargs,
    ) -> ToolResult:
        tool_map = {
            ToolType.PEEK: self.peek,
            ToolType.GREP: self.grep,
            ToolType.PARTITION: self.partition,
            ToolType.SUMMARIZE: self.summarize,
            ToolType.COUNT: self.count,
            ToolType.FILTER: self.filter,
        }

        if tool not in tool_map:
            return ToolResult.error_result(tool, f"Tool '{tool}' not supported")

        return tool_map[tool](**kwargs)

    def get_tool_stats(self) -> Dict[str, Any]:
        stats = {tool.value: {"calls": 0, "successes": 0, "total_time": 0.0} for tool in ToolType}
        
        for result in self._tool_history:
            key = result.tool.value
            stats[key]["calls"] += 1
            if result.success:
                stats[key]["successes"] += 1
            stats[key]["total_time"] += result.execution_time

        return stats
