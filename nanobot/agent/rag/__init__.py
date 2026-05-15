"""RAG (Retrieval-Augmented Generation) 模块

提供文档索引、向量存储、混合检索和重排序能力。
"""

from nanobot.agent.rag.store import RAGStore, DocumentChunk, DocumentRecord
from nanobot.agent.rag.chunker import (
    BaseChunker,
    Chunk,
    MarkdownHeaderSplitter,
    SemanticSplitter,
    CodeASTSplitter,
    SlidingWindowSplitter,
    get_chunker_for_file,
    get_chunker_by_name,
)
from nanobot.agent.rag.indexer import DocumentIndexer
from nanobot.agent.rag.retriever import (
    RAGRetriever,
    QueryExpander,
    BM25Retriever,
    RetrievalResult,
    RAGSearchResult,
)
from nanobot.agent.rag.document_parser import DocumentParser, ParsedDocument

__all__ = [
    "RAGStore",
    "DocumentChunk",
    "DocumentRecord",
    "BaseChunker",
    "Chunk",
    "MarkdownHeaderSplitter",
    "SemanticSplitter",
    "CodeASTSplitter",
    "SlidingWindowSplitter",
    "get_chunker_for_file",
    "get_chunker_by_name",
    "DocumentIndexer",
    "RAGRetriever",
    "QueryExpander",
    "BM25Retriever",
    "RetrievalResult",
    "RAGSearchResult",
    "DocumentParser",
    "ParsedDocument",
]
