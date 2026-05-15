"""RAG文档索引器 - 编排解析、分块和存储流程"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.rag.chunker import BaseChunker, Chunk, get_chunker_for_file
from nanobot.agent.rag.store import DocumentChunk, DocumentRecord, RAGStore


class DocumentIndexer:
    """文档索引器

    职责：
    1. 解析文档内容（委托给document_parser）
    2. 选择合适的分块策略并执行分块
    3. 将分块结果存入向量存储
    4. 维护文档注册表（去重/增量更新）
    """

    def __init__(self, store: RAGStore):
        self.store = store

    def index_file(
        self,
        file_path: str | Path,
        doc_type: str | None = None,
        chunker: BaseChunker | None = None,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """索引单个文件

        Args:
            file_path: 文件路径
            doc_type: 文档类型（自动检测为None）
            chunker: 指定分块器（自动选择为None）
            force_reindex: 强制重新索引

        Returns:
            索引结果统计
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if not path.is_file():
            return {"success": False, "error": f"Not a file: {file_path}"}

        file_size = path.stat().st_size
        if file_size == 0:
            return {"success": False, "error": f"Empty file: {file_path}"}

        file_mtime = path.stat().st_mtime

        if not force_reindex and self.store.is_document_indexed(str(path), file_mtime):
            logger.info(f"Document already indexed (unchanged): {path.name}")
            record = self.store.get_document_record(str(path))
            return {
                "success": True,
                "skipped": True,
                "reason": "already_indexed",
                "file_name": path.name,
                "chunk_count": record.chunk_count if record else 0,
            }

        try:
            text = self._parse_file(str(path), doc_type or self._detect_doc_type(path))
        except Exception as e:
            logger.error(f"Failed to parse {path.name}: {e}")
            return {"success": False, "error": f"Parse error: {e}"}

        if not text or not text.strip():
            return {"success": False, "error": f"Empty content after parsing: {file_path}"}

        if chunker is None:
            chunker_kwargs = {}
            if hasattr(self.store, 'embedding_api_key'):
                chunker_kwargs['embedding_api_key'] = self.store.embedding_api_key
            if hasattr(self.store, 'embedding_base_url'):
                chunker_kwargs['embedding_base_url'] = self.store.embedding_base_url
            if hasattr(self.store, 'embedding_model'):
                chunker_kwargs['embedding_model'] = self.store.embedding_model
            chunker = get_chunker_for_file(str(path), **chunker_kwargs)

        raw_chunks = chunker.chunk(text, file_path=str(path))

        if not raw_chunks:
            return {"success": False, "error": f"No chunks produced from: {file_path}"}

        if force_reindex:
            existing = self.store.delete_by_file(str(path))
            if existing > 0:
                logger.info(f"Removed {existing} old chunks for reindex: {path.name}")

        doc_chunks = [
            DocumentChunk(
                chunk_id=self.store._make_chunk_id(str(path), c.chunk_index),
                file_path=str(path),
                content=c.content,
                chunk_index=c.chunk_index,
                doc_type=doc_type or self._detect_doc_type(path),
                metadata=c.metadata,
            )
            for c in raw_chunks
        ]

        added = self.store.add_chunks(doc_chunks)

        final_type = doc_type or self._detect_doc_type(path)
        self.store.register_document(
            file_path=str(path),
            doc_type=final_type,
            chunk_count=added,
            file_mtime=file_mtime,
            file_size=file_size,
        )

        logger.info(
            f"Indexed {path.name}: {added} chunks "
            f"({final_type}, {file_size} bytes)"
        )

        return {
            "success": True,
            "file_name": path.name,
            "file_path": str(path),
            "doc_type": final_type,
            "chunk_count": added,
            "file_size": file_size,
        }

    def remove_document(self, file_path: str | Path) -> dict[str, Any]:
        """移除已索引文档"""
        path_str = str(file_path)
        deleted = self.store.delete_by_file(path_str)

        if deleted > 0:
            logger.info(f"Removed document: {Path(file_path).name} ({deleted} chunks)")
            return {
                "success": True,
                "file_name": Path(file_path).name,
                "chunks_deleted": deleted,
            }

        return {
            "success": False,
            "error": f"Document not found in index: {file_path}",
        }

    def list_documents(self) -> list[dict[str, Any]]:
        """列出所有已索引文档"""
        stats = self.store.get_stats()
        return stats.get("documents", [])

    @staticmethod
    def _detect_doc_type(path: Path) -> str:
        ext = path.suffix.lower()
        type_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "doc",
            ".md": "markdown",
            ".markdown": "markdown",
            ".txt": "text",
            ".py": "code",
            ".js": "code",
            ".ts": "code",
            ".jsx": "code",
            ".tsx": "code",
            ".java": "code",
            ".go": "code",
            ".rs": "code",
            ".c": "code",
            ".cpp": "code",
            ".h": "code",
            ".cs": "code",
            ".rb": "code",
            ".php": "code",
            ".html": "html",
            ".htm": "html",
            ".xml": "xml",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".csv": "csv",
        }
        return type_map.get(ext, "text")

    @staticmethod
    def _parse_file(file_path: str, doc_type: str) -> str:
        """解析文件内容为纯文本

        根据文档类型选择不同的解析策略：
        - PDF: markitdown
        - DOCX: python-docx
        - 其他: 直接读取文本
        """
        path = Path(file_path)

        if doc_type == "pdf":
            return _parse_pdf(file_path)

        if doc_type in ("docx", "doc"):
            return _parse_docx(file_path)

        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        for enc in encodings:
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue

        raise ValueError(f"Cannot decode file with any supported encoding: {file_path}")


def _parse_pdf(file_path: str) -> str:
    """使用MarkItDown解析PDF"""
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(file_path)
        return result.text_content
    except ImportError:
        raise ValueError(
            "markitdown not installed. Install with: pip install markitdown"
        )
    except Exception as e:
        raise ValueError(f"PDF parse failed: {e}")


def _parse_docx(file_path: str) -> str:
    """使用python-docx解析DOCX"""
    try:
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except ImportError:
        raise ValueError(
            "python-docx not installed. Install with: pip install python-docx"
        )
    except Exception as e:
        raise ValueError(f"DOCX parse failed: {e}")
