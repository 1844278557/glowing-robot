"""RAG检索模块 - 查询扩展、混合搜索、重排序"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.agent.rag.store import RAGStore


@dataclass
class RetrievalResult:
    """单条检索结果"""

    content: str
    source_file: str
    chunk_index: int = 0
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


@dataclass
class RAGSearchResult:
    """RAG搜索结果集"""

    query: str
    results: list[RetrievalResult] = field(default_factory=list)
    total_found: int = 0
    query_expansion_used: bool = False
    reranked: bool = False
    method: str = ""

    def format_for_context(self, max_chars: int = 4000) -> str:
        """格式化为LLM上下文文本"""
        if not self.results:
            return ""

        parts = [f"## RAG 检索结果 (共 {len(self.results)} 条相关片段)\n"]
        total = 0

        for i, r in enumerate(self.results, 1):
            entry = (
                f"### [{i}] 来源: {r.source_file}"
                f" (相关性: {r.score:.2f})\n"
                f"{r.content}\n\n"
            )
            if total + len(entry) > max_chars:
                parts.append(f"\n*...省略剩余 {len(self.results) - i + 1} 条结果*\n")
                break
            parts.append(entry)
            total += len(entry)

        return "".join(parts)


class QueryExpander:
    """查询扩展器 - Multi-Query策略

    使用LLM生成多个查询变体，
    提高召回率。
    """

    SYSTEM_PROMPT = """You are a query expansion assistant. Given a user's search query about documents, generate {n} alternative queries that could help find relevant information. Each query should approach the topic from a slightly different angle (synonyms, broader/narrower scope, different terminology).

Return ONLY a JSON array of strings, no other text."""

    async def expand(
        self,
        query: str,
        provider: Any,
        model: str,
        n_queries: int = 3,
    ) -> list[str]:
        """生成查询变体"""
        if n_queries <= 1:
            return [query]

        prompt = self.SYSTEM_PROMPT.format(n=n_queries)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Original query: {query}"},
        ]

        try:
            response = await provider.chat_with_retry(
                messages=messages,
                model=model,
            )

            content = (response.content or "").strip()
            queries = json.loads(content)

            if isinstance(queries, list):
                expanded = [q.strip() for q in queries if q.strip()]
                return [query] + expanded[:n_queries]

            return [query]

        except (json.JSONDecodeError, TypeError):
            logger.debug("Query expansion: failed to parse LLM response")
            return [query]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]


class BM25Retriever:
    """BM25关键词检索器

    对已存储的文档块构建倒排索引，
    支持关键词精确匹配和模糊匹配。
    """

    def __init__(self, store: RAGStore):
        self.store = store
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._bm25 = None
        self._initialized = False

    def _ensure_index(self) -> None:
        if self._initialized:
            return

        try:
            from rank_bm25 import BM25Okapi
            import jieba
        except ImportError:
            logger.debug("rank_bm25 or jieba not available, BM25 disabled")
            self._initialized = True
            return

        collection = self.store._ensure_collection()

        try:
            all_data = collection.get(include=["documents"])
            if not all_data or not all_data.get("ids"):
                self._initialized = True
                return

            self._chunk_ids = all_data["ids"]
            self._corpus = all_data.get("documents", [])

            tokenized = [
                list(jieba.cut(doc)) for doc in self._corpus
            ]
            self._bm25 = BM25Okapi(tokenized)
            self._initialized = True

        except Exception as e:
            logger.warning(f"BM25 index build failed: {e}")
            self._initialized = True

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """BM25搜索，返回 (chunk_id, score) 列表"""
        self._ensure_index()

        if self._bm25 is None:
            return []

        try:
            import jieba
            tokens = list(jieba.cut(query))
            scores = self._bm25.get_scores(tokens)

            indexed = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            return [(self._chunk_ids[i], float(score)) for i, score in indexed if score > 0]

        except ImportError:
            return []
        except Exception as e:
            logger.warning(f"BM25 search error: {e}")
            return []


class RAGRetriever:
    """RAG综合检索器

    流程：
    1. 可选：查询扩展（Multi-Query）
    2. 并行：向量语义检索 + BM25关键词检索（混合）
    3. 融合：RRF（Reciprocal Rank Fusion）合并结果
    4. 可选：SiliconFlow重排序
    """

    DEFAULT_TOP_K = 10
    RERANK_TOP_N = 20
    RRF_K = 60

    def __init__(
        self,
        store: RAGStore,
        rerank_api_key: str | None = None,
        rerank_model: str = "Qwen/Qwen3-Reranker-8B",
        rerank_base_url: str = "https://api.siliconflow.cn/v1",
    ):
        self.store = store
        self.bm25 = BM25Retriever(store)
        self.rerank_api_key = rerank_api_key
        self.rerank_model = rerank_model
        self.rerank_base_url = rerank_base_url

    async def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        use_query_expansion: bool = False,
        use_rerank: bool = True,
        provider: Any = None,
        model: str = "",
        file_filter: str | None = None,
    ) -> RAGSearchResult:
        """执行RAG检索

        Args:
            query: 用户查询
            top_k: 返回结果数
            use_query_expansion: 是否启用查询扩展
            use_rerank: 是否使用重排序
            provider: LLM provider（用于查询扩展）
            model: 模型名称（用于查询扩展）
            file_filter: 过滤指定文件的结果

        Returns:
            RAGSearchResult
        """
        queries = [query]

        if use_query_expansion and provider and model:
            expander = QueryExpander()
            queries = await expander.expand(query, provider, model, n_queries=3)

        vector_results = self._vector_search(queries, top_k=self.RERANK_TOP_N)
        bm25_results = self._bm25_search(queries[0], top_k=self.RERANK_TOP_N)

        fused = self._rrf_fuse(vector_results, bm25_results)

        if file_filter:
            fused = [r for r in fused if r.metadata.get("file_path") == file_filter]

        results_obj = RAGSearchResult(
            query=query,
            total_found=len(fused),
            method="hybrid",
            query_expansion_used=len(queries) > 1,
        )

        if use_rerank and self.rerank_api_key and fused:
            fused = await self._rerank(query, fused[:self.RERANK_TOP_N])
            results_obj.reranked = True

        results_obj.results = fused[:top_k]
        return results_obj

    def _vector_search(
        self, queries: list[str], top_k: int
    ) -> list[RetrievalResult]:
        """向量语义检索"""
        all_results: dict[str, RetrievalResult] = {}

        for q in queries:
            raw = self.store.search(q, n_results=top_k)

            for item in raw:
                cid = item["id"]
                existing = all_results.get(cid)
                score = 1.0 - item.get("distance", 1.0)

                if existing is None or score > existing.score:
                    meta = item.get("metadata", {})
                    all_results[cid] = RetrievalResult(
                        content=item.get("document", ""),
                        source_file=meta.get("file_path", ""),
                        chunk_index=meta.get("chunk_index", 0),
                        score=score,
                        metadata=meta,
                    )

        return sorted(all_results.values(), key=lambda r: r.score, reverse=True)

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """BM25关键词检索"""
        bm25_hits = self.bm25.search(query, top_k=top_k)

        if not bm25_hits:
            return []

        results: list[RetrievalResult] = []
        for chunk_id, score in bm25_hits:
            raw = self.store._ensure_collection().get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )

            if raw and raw.get("ids"):
                meta = (raw.get("metadatas") or [[]])[0]
                doc = (raw.get("documents") or [""])[0]
                results.append(RetrievalResult(
                    content=doc,
                    source_file=meta.get("file_path", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    score=min(score / 20.0, 1.0),
                    metadata=meta,
                ))

        return results

    @staticmethod
    def _rrf_fuse(
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        k: int = RRF_K,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion融合排序"""
        scores: dict[str, float] = {}
        items: dict[str, RetrievalResult] = {}

        for rank, r in enumerate(vector_results):
            rid = f"{r.source_file}::{r.chunk_index}"
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            items[rid] = r

        for rank, r in enumerate(bm25_results):
            rid = f"{r.source_file}::{r.chunk_index}"
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            if rid not in items:
                items[rid] = r

        fused_rids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result = []
        for rid in fused_rids:
            item = items[rid]
            item.score = round(scores[rid], 4)
            result.append(item)

        return result

    async def _rerank(
        self, query: str, candidates: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """使用SiliconFlow API进行重排序"""
        if not candidates or not self.rerank_api_key:
            return candidates

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.rerank_api_key,
                base_url=self.rerank_base_url,
            )

            response = client.request(
                method="post",
                url="/rerank",
                json={
                    "model": self.rerank_model,
                    "query": query,
                    "documents": [c.content for c in candidates],
                    "top_n": len(candidates),
                    "return_documents": False,
                },
            )

            data = response.json() if hasattr(response, 'json') else response
            reranked_indices = [
                r.get("index", i)
                for r in data.get("results", [])
            ]

            reordered = [candidates[i] for i in reranked_indices if i < len(candidates)]

            for i, r in enumerate(reordered):
                result_item = next(
                    (res for res in data.get("results", []) if res.get("index") == candidates.index(r)),
                    None,
                )
                if result_item:
                    r.score = result_item.get("relevance_score", r.score)

            remaining = [
                c for c in candidates
                if c not in reordered
            ]
            return reordered + remaining

        except ImportError:
            logger.warning("openai package not available for reranking")
            return candidates
        except Exception as e:
            logger.warning(f"Reranking failed ({self.rerank_model}): {e}")
            return candidates
