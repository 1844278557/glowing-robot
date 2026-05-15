"""RAG文档分块策略

支持多种分块方式：
- MarkdownHeaderSplitter: 按Markdown标题层级分割，保留结构
- SemanticChunking: 基于余弦相似度阈值检测主题边界
- ASTChunking: 基于语法树按函数/类边界分割代码
- SlidingWindow: 通用滑动窗口，带重叠
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class Chunk:
    """单个文本块"""

    content: str
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return len(self.content) // 4


class BaseChunker:
    """分块器基类"""

    def chunk(self, text: str, file_path: str = "", **kwargs) -> list[Chunk]:
        raise NotImplementedError


class MarkdownHeaderSplitter(BaseChunker):
    """按Markdown标题分割

    保留标题上下文（每个块包含父级标题路径），
    合并过小的块到前一个块。
    """

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    MIN_CHUNK_SIZE = 50

    def __init__(
        self,
        max_chunk_size: int = 1500,
        merge_small: bool = True,
    ):
        self.max_chunk_size = max_chunk_size
        self.merge_small = merge_small

    def chunk(self, text: str, file_path: str = "", **kwargs) -> list[Chunk]:
        lines = text.split("\n")
        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_headers: list[str] = []
        header_stack: list[str] = []

        for line in lines:
            match = self.HEADER_PATTERN.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                if current_lines and any(l.strip() for l in current_lines):
                    header_context = " > ".join(header_stack) if header_stack else ""
                    content = "\n".join(current_lines).strip()
                    if content:
                        chunks.append(Chunk(
                            content=content,
                            metadata={"headers": header_context, "source": file_path},
                        ))

                current_lines = [line]

                while header_stack and len(header_stack) >= level:
                    header_stack.pop()
                header_stack.append(title)
                current_headers = list(header_stack)
            else:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                header_context = " > ".join(current_headers) if current_headers else ""
                chunks.append(Chunk(
                    content=content,
                    metadata={"headers": header_context, "source": file_path},
                ))

        if self.merge_small:
            chunks = self._merge_small_chunks(chunks)

        for i, c in enumerate(chunks):
            c.chunk_index = i

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        merged: list[Chunk] = []
        for c in chunks:
            if (
                merged
                and c.token_estimate < self.MIN_CHUNK_SIZE
                and merged[-1].token_estimate + c.token_estimate < self.max_chunk_size
            ):
                merged[-1].content += "\n\n" + c.content
                headers = merged[-1].metadata.get("headers", "")
                new_headers = c.metadata.get("headers", "")
                if new_headers and new_headers != headers:
                    merged[-1].metadata["headers"] = new_headers
            else:
                merged.append(c)
        return merged


class SemanticSplitter(BaseChunker):
    """基于语义相似度的分块

    使用句子边界检测 + 相邻句子的余弦相似度，
    在相似度低于阈值时切分。回退为固定窗口。
    """

    SIMILARITY_THRESHOLD = 0.35
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?。！？])\s+')
    WINDOW_SIZE = 3

    def __init__(
        self,
        max_chunk_size: int = 1000,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        window_size: int = WINDOW_SIZE,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
    ):
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url or "https://api.siliconflow.cn/v1"
        self._embedding_fn = None

    def _get_embeddings(self, sentences: list[str]) -> list[list[float]] | None:
        model = self.embedding_model

        if model.startswith("Qwen/") or model.startswith("siliconflow:"):
            return self._get_siliconflow_embeddings(sentences)

        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            if self._embedding_fn is None:
                self._embedding_fn = SentenceTransformer(
                    self.embedding_model,
                    cache_folder=None,
                )

            embeddings = self._embedding_fn.encode(
                sentences,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist() if isinstance(embeddings, type(np.array([]))) else embeddings

        except ImportError:
            logger.debug("sentence_transformers not available, using fallback splitting")
            return None
        except Exception as e:
            logger.warning(f"Sentence embedding failed: {e}")
            return None

    def _get_siliconflow_embeddings(self, sentences: list[str]) -> list[list[float]] | None:
        """使用SiliconFlow API获取嵌入向量"""
        import httpx

        model = self.embedding_model.replace("siliconflow:", "")
        api_key = self.embedding_api_key
        base_url = self.embedding_base_url

        if not api_key:
            logger.debug("SiliconFlow API key not provided, using fallback splitting")
            return None

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "input": sentences,
            }

            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{base_url}/embeddings",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings

        except Exception as e:
            logger.warning(f"SiliconFlow embedding failed: {e}, using fallback splitting")
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def chunk(self, text: str, file_path: str = "", **kwargs) -> list[Chunk]:
        sentences = self.SENTENCE_ENDINGS.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        embeddings = self._get_embeddings(sentences)

        if embeddings is None:
            return self._fallback_split(sentences, file_path)

        boundaries = [0]
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))

        chunks: list[Chunk] = []
        for start_idx in range(len(boundaries) - 1):
            end_idx = boundaries[start_idx + 1]
            segment_sentences = sentences[start_idx:end_idx]
            content = " ".join(segment_sentences)
            if content.strip():
                chunks.append(Chunk(content=content, metadata={"source": file_path}))

        chunks = self._enforce_max_size(chunks)

        for i, c in enumerate(chunks):
            c.chunk_index = i

        return chunks

    def _fallback_split(self, sentences: list[str], file_path: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buf_tokens = 0

        for s in sentences:
            s_tokens = len(s) // 4
            if buf_tokens + s_tokens > self.max_chunk_size and buffer:
                chunks.append(Chunk(
                    content=" ".join(buffer),
                    metadata={"source": file_path},
                ))
                buffer = [s]
                buf_tokens = s_tokens
            else:
                buffer.append(s)
                buf_tokens += s_tokens

        if buffer:
            chunks.append(Chunk(content=" ".join(buffer), metadata={"source": file_path}))

        for i, c in enumerate(chunks):
            c.chunk_index = i
        return chunks

    def _enforce_max_size(self, chunks: list[Chunk]) -> list[Chunk]:
        result: list[Chunk] = []
        for c in chunks:
            if c.token_estimate <= self.max_chunk_size:
                result.append(c)
            else:
                sub_parts = self._split_large(c.content, c.metadata.get("source", ""))
                result.extend(sub_parts)
        return result

    def _split_large(self, text: str, source: str) -> list[Chunk]:
        sentences = self.SENTENCE_ENDINGS.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buf_tokens = 0

        for s in sentences:
            s_tokens = len(s) // 4
            if buf_tokens + s_tokens > self.max_chunk_size and buffer:
                chunks.append(Chunk(content=" ".join(buffer), metadata={"source": source}))
                buffer = [s]
                buf_tokens = s_tokens
            else:
                buffer.append(s)
                buf_tokens += s_tokens

        if buffer:
            chunks.append(Chunk(content=" ".join(buffer), metadata={"source": source}))
        return chunks


class CodeASTSplitter(BaseChunker):
    """基于AST的代码分块

    解析Python源码的抽象语法树，
    按顶层定义（类、函数）边界分割。
    非Python文件回退为滑动窗口。
    """

    SUPPORTED_EXTENSIONS = {".py"}
    MAX_CHUNK_TOKENS = 1200

    def __init__(self, max_chunk_size: int = MAX_CHUNK_TOKENS):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, file_path: str = "", **kwargs) -> list[Chunk]:
        ext = Path(file_path).suffix.lower() if file_path else ""

        if ext not in self.SUPPORTED_EXTENSIONS:
            return SlidingWindowSplitter(chunk_size=self.max_chunk_size).chunk(
                text, file_path=file_path
            )

        return self._split_by_ast(text, file_path)

    def _split_by_ast(self, source: str, file_path: str) -> list[Chunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return SlidingWindowSplitter(chunk_size=self.max_chunk_size).chunk(
                source, file_path=file_path
            )

        chunks: list[Chunk] = []
        module_docstring = ast.get_docstring(tree)

        if module_docstring:
            chunks.append(Chunk(
                content=f'"""\n{module_docstring}\n"""',
                metadata={"type": "module_docstring", "source": file_path},
            ))

        for node in tree.body:
            start_line = getattr(node, "lineno", 0) - 1
            end_line = getattr(node, "end_lineno", start_line + 1)
            lines = source.split("\n")[start_line:end_line]
            code_text = "\n".join(lines)

            node_type = type(node).__name__
            name = getattr(node, "name", "")

            meta: dict[str, Any] = {
                "type": node_type.lower(),
                "source": file_path,
                "line_start": start_line + 1,
                "line_end": end_line,
            }
            if name:
                meta["name"] = name

            if len(code_text) // 4 <= self.max_chunk_size:
                chunks.append(Chunk(content=code_text, metadata=meta))
            else:
                sub_chunks = SlidingWindowSplitter(
                    chunk_size=self.max_chunk_size,
                ).chunk(code_text, file_path=file_path)
                for sc in sub_chunks:
                    sc.metadata.update(meta)
                    chunks.append(sc)

        if not chunks:
            chunks.append(Chunk(content=source[:self.max_chunk_size * 4], metadata={
                "type": "raw", "source": file_path,
            }))

        for i, c in enumerate(chunks):
            c.chunk_index = i
        return chunks


class SlidingWindowSplitter(BaseChunker):
    """通用滑动窗口分块

    按字符/token数切割，相邻窗口有重叠部分，
    尽量在句子/段落边界处断开。
    """

    SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?。！？\n])\s+')
    PARAGRAPH_BOUNDARY = re.compile(r'\n\s*\n')

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        separator: str = "\n\n",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

    def chunk(self, text: str, file_path: str = "", **kwargs) -> list[Chunk]:
        if not text or not text.strip():
            return []

        segments = self.PARAGRAPH_BOUNDARY.split(text)
        segments = [s.strip() for s in segments if s.strip()]

        if not segments:
            segments = [text.strip()]

        chunks: list[Chunk] = []
        current_buffer: list[str] = []
        current_length = 0

        for seg in segments:
            seg_len = len(seg)

            if current_length + seg_len + len(self.separator) <= self.chunk_size * 4:
                current_buffer.append(seg)
                current_length += seg_len + len(self.separator)
            else:
                if current_buffer:
                    content = self.separator.join(current_buffer)
                    chunks.append(Chunk(content=content, metadata={"source": file_path}))

                if seg_len > self.chunk_size * 4:
                    sub_chunks = self._split_oversized(seg, file_path)
                    chunks.extend(sub_chunks)
                    current_buffer = []
                    current_length = 0
                else:
                    keep_overlap = self._compute_overlap(current_buffer)
                    current_buffer = keep_overlap + [seg]
                    current_length = sum(len(s) + len(self.separator) for s in current_buffer)

        if current_buffer:
            content = self.separator.join(current_buffer)
            chunks.append(Chunk(content=content, metadata={"source": file_path}))

        for i, c in enumerate(chunks):
            c.chunk_index = i
        return chunks

    def _compute_overlap(self, previous_buffer: list[str]) -> list[str]:
        if not previous_buffer or self.overlap <= 0:
            return []

        overlap_chars = self.overlap * 4
        combined = self.separator.join(previous_buffer)

        if len(combined) <= overlap_chars:
            return list(previous_buffer)

        tail = combined[-overlap_chars:]
        boundary_match = list(self.SENTENCE_BOUNDARY.finditer(tail))

        if boundary_match:
            cut_pos = boundary_match[0].start()
            overlap_text = tail[cut_pos:].strip()
        else:
            overlap_text = tail.strip()

        return [overlap_text] if overlap_text else []

    def _split_oversized(self, text: str, file_path: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        parts = self.SENTENCE_BOUNDARY.split(text)
        parts = [p.strip() for p in parts if p.strip()]

        buffer: list[str] = []
        buf_len = 0

        for part in parts:
            part_len = len(part)
            if buf_len + part_len <= self.chunk_size * 4:
                buffer.append(part)
                buf_len += part_len
            else:
                if buffer:
                    chunks.append(Chunk(
                        content=" ".join(buffer),
                        metadata={"source": file_path},
                    ))
                if part_len > self.chunk_size * 4:
                    for i in range(0, part_len, self.chunk_size * 4):
                        chunk_text = part[i:i + self.chunk_size * 4].strip()
                        if chunk_text:
                            chunks.append(Chunk(content=chunk_text, metadata={"source": file_path}))
                    buffer = []
                    buf_len = 0
                else:
                    buffer = [part]
                    buf_len = part_len

        if buffer:
            chunks.append(Chunk(content=" ".join(buffer), metadata={"source": file_path}))

        return chunks


def get_chunker_for_file(file_path: str, **kwargs) -> BaseChunker:
    """根据文件类型自动选择最佳分块器"""
    ext = Path(file_path).suffix.lower() if file_path else ""

    md_extensions = {".md", ".markdown", ".mdx"}
    code_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"}

    if ext in md_extensions:
        return MarkdownHeaderSplitter(**kwargs)
    elif ext in code_extensions:
        if ext == ".py":
            return CodeASTSplitter(**kwargs)
        return SlidingWindowSplitter(**kwargs)
    else:
        return SemanticSplitter(**kwargs)


def get_chunker_by_name(name: str, **kwargs) -> BaseChunker:
    """按名称获取分块器实例"""
    chunkers = {
        "markdown": MarkdownHeaderSplitter,
        "semantic": SemanticSplitter,
        "ast": CodeASTSplitter,
        "sliding_window": SlidingWindowSplitter,
    }
    cls = chunkers.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown chunker: {name}. Available: {list(chunkers.keys())}")
    return cls(**kwargs)
