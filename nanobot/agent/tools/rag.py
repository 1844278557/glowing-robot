"""RAG工具 - LLM可调用的索引和检索接口"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.rag.indexer import DocumentIndexer
from nanobot.agent.rag.retriever import RAGRetriever
from nanobot.agent.rag.store import RAGStore


class IndexDocumentTool(Tool):
    """索引文档到RAG知识库"""

    name = "index_document"
    description = (
        "将用户上传的文档（PDF、DOCX、MD、TXT、代码等）"
        "索引到RAG知识库，以便后续进行基于文档内容的问答。"
        "支持自动检测文件类型并选择最优分块策略。"
        "如果文档已索引且未修改，会跳过重复处理。"
        "文档有默认30天过期时间，过期后不参与检索。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要索引的文件的完整路径（如 /path/to/document.pdf）",
            },
            "force_reindex": {
                "type": "boolean",
                "description": "是否强制重新索引（覆盖已有数据，重置过期时间）",
                "default": False,
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, store: RAGStore):
        self.store = store
        self._indexer: DocumentIndexer | None = None

    @property
    def indexer(self) -> DocumentIndexer:
        if self._indexer is None:
            self._indexer = DocumentIndexer(self.store)
        return self._indexer

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("file_path", "")
        force_reindex = kwargs.get("force_reindex", False)

        if not file_path:
            return json.dumps({"error": "file_path is required"}, ensure_ascii=False)

        path = Path(file_path)
        if not path.exists():
            return json.dumps(
                {"error": f"File not found: {file_path}"}, ensure_ascii=False
            )

        try:
            result = self.indexer.index_file(
                file_path=file_path,
                force_reindex=force_reindex,
            )
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Index document failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class SearchRAGTool(Tool):
    """搜索RAG知识库"""

    name = "search_rag"
    description = (
        "在已索引的RAG知识库中搜索与用户问题相关的文档片段。"
        "使用混合检索（关键词+语义）+ 可选重排序返回最相关的内容。"
        "已过期的文档不会出现在检索结果中。"
        "适用于用户针对已上传文档提出具体问题时使用。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询文本（用户的问题）",
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认5",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
            "file_filter": {
                "type": "string",
                "description": "可选，限定在指定文件中搜索（文件名或路径）",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        store: RAGStore,
        rerank_api_key: str | None = None,
        rerank_model: str = "Qwen/Qwen3-Reranker-8B",
        rerank_base_url: str = "https://api.siliconflow.cn/v1",
    ):
        self.store = store
        self.retriever = RAGRetriever(
            store=store,
            rerank_api_key=rerank_api_key,
            rerank_model=rerank_model,
            rerank_base_url=rerank_base_url,
        )

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        top_k = int(kwargs.get("top_k", 5))
        file_filter = kwargs.get("file_filter")

        if not query:
            return json.dumps({"error": "query is required"}, ensure_ascii=False)

        try:
            result = await self.retriever.search(
                query=query,
                top_k=top_k,
                use_query_expansion=False,
                use_rerank=True,
                file_filter=file_filter,
            )

            output = {
                "query": result.query,
                "total_found": result.total_found,
                "method": result.method,
                "reranked": result.reranked,
                "results": [r.to_dict() for r in result.results],
            }

            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Search RAG failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class RemoveRAGDocumentTool(Tool):
    """从RAG知识库移除已索引文档"""

    name = "remove_rag_document"
    description = (
        "从RAG知识库中移除指定文档及其所有分块数据。"
        "适用于文档已过期、需要更新或用户主动要求删除时。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要移除的文件路径（必须是之前通过 index_document 索引过的文件）",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, store: RAGStore):
        self.store = store

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("file_path", "")
        if not file_path:
            return json.dumps({"error": "file_path is required"}, ensure_ascii=False)

        try:
            indexer = DocumentIndexer(self.store)
            result = indexer.remove_document(file_path)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Remove RAG document failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class ListRAGDocumentsTool(Tool):
    """列出RAG知识库中所有已索引文档"""

    name = "list_rag_documents"
    description = (
        "列出RAG知识库中所有已索引的文档，包括文件名、类型、分块数量、"
        "索引时间、过期时间和状态（active/expired）。"
        "用于查看当前知识库中有哪些文档及其健康状态。"
    )

    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, store: RAGStore):
        self.store = store

    async def execute(self, **kwargs: Any) -> str:
        try:
            stats = self.store.get_stats()
            return json.dumps(stats, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"List RAG documents failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class ClearRAGStoreTool(Tool):
    """清空RAG知识库"""

    name = "clear_rag_store"
    description = (
        "清空RAG知识库中的所有文档和向量数据。"
        "这是一个危险操作，执行后所有已索引数据将被永久删除。"
        "仅在用户明确要求重置或清理时使用。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "confirm": {
                "type": "string",
                'description': '必须传入 "yes" 确认清空操作',
            },
        },
        "required": ["confirm"],
    }

    def __init__(self, store: RAGStore):
        self.store = store

    async def execute(self, **kwargs: Any) -> str:
        confirm = kwargs.get("confirm", "")
        if confirm != "yes":
            return json.dumps(
                {"error": '必须传入 confirm="yes" 才能执行清空操作'},
                ensure_ascii=False,
            )

        try:
            result = self.store.clear_all()
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Clear RAG store failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class CleanupExpiredRAGTool(Tool):
    """清理RAG知识库中的过期文档"""

    name = "cleanup_expired_rag"
    description = (
        "清理RAG知识库中所有已过期的文档及其向量数据。"
        "过期文档不再参与检索结果，但占用存储空间。"
        "此工具释放过期文档占用的空间。"
    )

    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, store: RAGStore):
        self.store = store

    async def execute(self, **kwargs: Any) -> str:
        try:
            result = self.store.cleanup_expired()
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Cleanup expired RAG failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
