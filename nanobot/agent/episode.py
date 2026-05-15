# nanobot/agent/episode.py
"""
Episode记忆系统 - 替代原有HISTORY.md

设计理念：
- 结构化存储：每个Episode是一个独立的JSON文件，包含完整的情境信息
- 分层检索：时间过滤 → 索引查找 → grep搜索 → 向量检索
- 延迟初始化：向量存储仅在Episode数量>100时启用
- grep兼容：summary字段支持原有的grep搜索方式
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re
import subprocess

from loguru import logger

from nanobot.utils.helpers import ensure_dir


@dataclass
class Episode:
    """
    Episode - 结构化的历史记忆单元
    
    每个Episode代表一段有意义的对话片段，包含：
    - 主题和摘要：便于快速理解内容
    - 关键要点和决策：提取核心信息
    - 实体信息：涉及的人物、项目、概念等
    - 时间和消息范围：便于定位原始对话
    - 重要性评分：用于检索排序
    """
    
    episode_id: str
    session_id: str
    created_at: str
    trigger: str
    topic: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    message_range: dict[str, int] = field(default_factory=dict)
    time_range: dict[str, str] = field(default_factory=dict)
    token_count: int = 0
    importance: int = 3
    tags: list[str] = field(default_factory=list)
    updated_at: str | None = None
    raw_messages: list[dict] | None = None
    
    def to_dict(self) -> dict:
        """转换为字典，过滤None值"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """从字典创建Episode实例"""
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)
    
    def to_grep_text(self) -> str:
        """
        生成grep兼容的文本格式
        
        保持与原HISTORY.md格式兼容，便于用户使用grep搜索
        格式：[时间] 主题摘要
        """
        time_str = self.time_range.get("start", self.created_at[:16])
        return f"[{time_str}] {self.topic}\n{self.summary}"
    
    def get_searchable_text(self) -> str:
        """获取用于向量检索的文本"""
        parts = [
            f"主题: {self.topic}",
            f"摘要: {self.summary}",
        ]
        if self.key_points:
            parts.append(f"要点: {', '.join(self.key_points)}")
        if self.decisions:
            parts.append(f"决策: {', '.join(self.decisions)}")
        if self.tags:
            parts.append(f"标签: {', '.join(self.tags)}")
        return "\n".join(parts)


class EpisodeStore:
    """
    Episode存储管理器
    
    负责：
    - Episode文件的CRUD操作
    - 索引文件的维护
    - 向量存储的延迟初始化
    """
    
    VECTOR_STORE_THRESHOLD = 100
    
    def __init__(self, memory_dir: Path):
        self.memory_dir = ensure_dir(memory_dir)
        self.episodes_dir = ensure_dir(memory_dir / "episodes")
        self.index_file = memory_dir / "episode_index.json"
        self.index: dict = self._load_index()
        self._vector_store = None
        self._vector_store_initialized = False
        
    def _load_index(self) -> dict:
        """加载索引文件，不存在则创建空索引"""
        if self.index_file.exists():
            try:
                return json.loads(self.index_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Episode index corrupted, creating new one")
        return {
            "episodes": [],
            "stats": {
                "total_count": 0,
                "last_updated": None
            }
        }
    
    def _save_index(self) -> None:
        """保存索引文件"""
        self.index["stats"]["last_updated"] = datetime.now().isoformat()
        self.index_file.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def _update_index(self, episode: Episode) -> None:
        """更新索引：添加新Episode的元信息"""
        episode_meta = {
            "episode_id": episode.episode_id,
            "session_id": episode.session_id,
            "created_at": episode.created_at,
            "topic": episode.topic,
            "importance": episode.importance,
            "tags": episode.tags,
            "time_start": episode.time_range.get("start", ""),
            "time_end": episode.time_range.get("end", ""),
        }
        
        self.index["episodes"].append(episode_meta)
        self.index["stats"]["total_count"] = len(self.index["episodes"])
        self._save_index()
        
        if self._should_init_vector_store():
            self._init_vector_store()
    
    def _should_init_vector_store(self) -> bool:
        """判断是否需要初始化向量存储"""
        return (
            not self._vector_store_initialized 
            and len(self.index["episodes"]) >= self.VECTOR_STORE_THRESHOLD
        )
    
    def _init_vector_store(self) -> None:
        """
        延迟初始化向量存储
        
        当Episode数量超过阈值时，自动启用向量检索
        使用ChromaDB作为向量数据库
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            chroma_dir = self.memory_dir / "chroma"
            chroma_dir.mkdir(parents=True, exist_ok=True)
            
            client = chromadb.PersistentClient(path=str(chroma_dir))
            self._vector_store = client.get_or_create_collection(
                name="episodes",
                metadata={"hnsw:space": "cosine"}
            )
            
            self._vector_store_initialized = True
            logger.info(
                f"Vector store initialized at {chroma_dir} "
                f"(episodes: {len(self.index['episodes'])})"
            )
            
            self._sync_vector_store()
            
        except ImportError:
            logger.warning(
                "ChromaDB not installed, vector search disabled. "
                "Install with: pip install chromadb"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    def _sync_vector_store(self) -> None:
        """同步所有Episode到向量存储"""
        if not self._vector_store:
            return
            
        for meta in self.index["episodes"]:
            episode = self.load(meta["episode_id"])
            if episode:
                self._add_to_vector_store(episode)
    
    def _add_to_vector_store(self, episode: Episode) -> None:
        """将单个Episode添加到向量存储"""
        if not self._vector_store:
            return
            
        try:
            self._vector_store.upsert(
                ids=[episode.episode_id],
                documents=[episode.get_searchable_text()],
                metadatas=[{
                    "session_id": episode.session_id,
                    "importance": episode.importance,
                    "created_at": episode.created_at,
                    "topic": episode.topic,
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to add episode to vector store: {e}")
    
    def _get_episode_path(self, episode_id: str) -> Path:
        """
        根据episode_id计算存储路径
        
        路径格式：episodes/YYYY-MM/ep_YYYYMMDD_HHMMSS_XXX.json
        按月份分目录，便于管理和清理
        """
        date_part = episode_id.split("_")[1] if "_" in episode_id else ""
        if len(date_part) >= 6:
            year_month = f"{date_part[:4]}-{date_part[4:6]}"
        else:
            year_month = datetime.now().strftime("%Y-%m")
        return self.episodes_dir / year_month / f"{episode_id}.json"
    
    def generate_episode_id(self) -> str:
        """生成唯一的Episode ID"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        seq = len(self.index.get("episodes", []))
        return f"ep_{ts}_{seq:04d}"
    
    def save(self, episode: Episode) -> None:
        """
        保存Episode到文件系统
        
        同时更新索引和向量存储（如果已启用）
        """
        path = self._get_episode_path(episode.episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = episode.to_dict()
        if save_data.get("raw_messages") and len(save_data["raw_messages"]) > 10:
            save_data["raw_messages"] = save_data["raw_messages"][-10:]
            save_data["raw_messages_truncated"] = True
        
        path.write_text(
            json.dumps(save_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        self._update_index(episode)
        
        if self._vector_store:
            self._add_to_vector_store(episode)
        
        logger.debug(f"Episode saved: {episode.episode_id}")
    
    def load(self, episode_id: str) -> Episode | None:
        """加载指定Episode"""
        path = self._get_episode_path(episode_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return Episode.from_dict(data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load episode {episode_id}: {e}")
        return None
    
    def delete(self, episode_id: str) -> bool:
        """删除指定Episode"""
        path = self._get_episode_path(episode_id)
        if path.exists():
            path.unlink()
            
            self.index["episodes"] = [
                ep for ep in self.index["episodes"] 
                if ep["episode_id"] != episode_id
            ]
            self.index["stats"]["total_count"] = len(self.index["episodes"])
            self._save_index()
            
            if self._vector_store:
                try:
                    self._vector_store.delete(ids=[episode_id])
                except Exception:
                    pass
            
            return True
        return False
    
    def append(self, entry: str, session_id: str = "unknown") -> Episode:
        """
        兼容原HISTORY.md的append_history接口
        
        用于平滑迁移，将简单的文本条目转换为Episode
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        episode = Episode(
            episode_id=self.generate_episode_id(),
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            trigger="consolidation",
            topic="",
            summary=entry,
            time_range={"start": ts, "end": ts},
        )
        self.save(episode)
        return episode
    
    def get_episodes_by_time(
        self, 
        start_time: str | None = None, 
        end_time: str | None = None
    ) -> list[Episode]:
        """
        按时间范围获取Episode
        
        时间格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM
        """
        results = []
        for meta in self.index.get("episodes", []):
            ep_time = meta.get("time_start", meta.get("created_at", ""))
            if start_time and ep_time < start_time:
                continue
            if end_time and ep_time > end_time:
                continue
            
            episode = self.load(meta["episode_id"])
            if episode:
                results.append(episode)
        
        results.sort(key=lambda e: e.created_at, reverse=True)
        return results
    
    def get_episodes_by_tags(self, tags: list[str]) -> list[Episode]:
        """按标签获取Episode"""
        results = []
        for meta in self.index.get("episodes", []):
            ep_tags = set(meta.get("tags", []))
            if any(tag in ep_tags for tag in tags):
                episode = self.load(meta["episode_id"])
                if episode:
                    results.append(episode)
        
        results.sort(key=lambda e: e.importance, reverse=True)
        return results
    
    def get_recent_episodes(self, limit: int = 10) -> list[Episode]:
        """获取最近的N个Episode"""
        results = []
        for meta in self.index.get("episodes", [])[-limit:]:
            episode = self.load(meta["episode_id"])
            if episode:
                results.append(episode)
        return results
    
    def get_stats(self) -> dict:
        """获取Episode统计信息"""
        return self.index.get("stats", {})


class EpisodeRetriever:
    """
    Episode检索器
    
    实现分层检索策略：
    1. 时间过滤：快速缩小范围
    2. 索引查找：利用索引元数据
    3. grep搜索：文本匹配（兼容原有方式）
    4. 向量检索：语义相似度（可选）
    """
    
    def __init__(self, store: EpisodeStore):
        self.store = store
    
    def search(
        self,
        query: str,
        time_start: str | None = None,
        time_end: str | None = None,
        tags: list[str] | None = None,
        min_importance: int = 1,
        limit: int = 10,
        use_vector: bool = True,
    ) -> list[Episode]:
        """
        综合检索Episode
        
        Args:
            query: 搜索查询文本
            time_start: 起始时间
            time_end: 结束时间
            tags: 标签过滤
            min_importance: 最小重要性
            limit: 返回数量限制
            use_vector: 是否使用向量检索
        
        Returns:
            匹配的Episode列表，按相关性排序
        """
        candidates = self._get_candidates(time_start, time_end, tags)
        
        if use_vector and self.store._vector_store:
            scored = self._vector_search(query, candidates, limit)
        else:
            scored = self._grep_search(query, candidates)
        
        results = [
            ep for ep, score in scored 
            if ep.importance >= min_importance
        ][:limit]
        
        return results
    
    def _get_candidates(
        self,
        time_start: str | None,
        time_end: str | None,
        tags: list[str] | None,
    ) -> list[Episode]:
        """获取候选Episode集合"""
        if time_start or time_end:
            return self.store.get_episodes_by_time(time_start, time_end)
        elif tags:
            return self.store.get_episodes_by_tags(tags)
        else:
            return self.store.get_recent_episodes(limit=100)
    
    def _grep_search(
        self, 
        query: str, 
        candidates: list[Episode]
    ) -> list[tuple[Episode, float]]:
        """
        grep风格的文本搜索
        
        使用正则匹配，支持大小写不敏感
        返回匹配的Episode及其简单评分
        """
        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        
        for episode in candidates:
            searchable = episode.get_searchable_text()
            matches = pattern.findall(searchable)
            
            if matches:
                score = len(matches) * episode.importance
                results.append((episode, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _vector_search(
        self,
        query: str,
        candidates: list[Episode],
        limit: int,
    ) -> list[tuple[Episode, float]]:
        """
        向量语义检索
        
        使用ChromaDB进行相似度搜索
        返回匹配的Episode及其相似度分数
        """
        if not self.store._vector_store:
            return self._grep_search(query, candidates)
        
        try:
            results = self.store._vector_store.query(
                query_texts=[query],
                n_results=min(limit * 2, 50),
            )
            
            episodes = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]
                
                for ep_id, distance in zip(ids, distances):
                    episode = self.store.load(ep_id)
                    if episode and episode in candidates:
                        similarity = 1 - distance
                        episodes.append((episode, similarity))
            
            return episodes
            
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to grep: {e}")
            return self._grep_search(query, candidates)
    
    def grep_files(
        self,
        pattern: str,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> list[str]:
        """
        兼容原有grep搜索方式
        
        直接对Episode文件进行grep搜索
        返回匹配的文本行，格式与原HISTORY.md兼容
        """
        candidates = self._get_candidates(time_start, time_end, None)
        
        results = []
        for episode in candidates:
            grep_text = episode.to_grep_text()
            for line in grep_text.split("\n"):
                if re.search(pattern, line, re.IGNORECASE):
                    results.append(line)
        
        return results
    
    def get_context_for_prompt(
        self,
        query: str,
        max_episodes: int = 5,
        max_tokens: int = 2000,
    ) -> str:
        """
        为LLM Prompt构建Episode上下文
        
        检索相关Episode并格式化为可读文本
        控制总token数在限制内
        """
        episodes = self.search(query, limit=max_episodes)
        
        if not episodes:
            return ""
        
        parts = ["## 相关历史记忆\n"]
        total_tokens = 0
        
        for episode in episodes:
            entry = f"### [{episode.time_range.get('start', '?')}] {episode.topic}\n"
            entry += f"{episode.summary}\n"
            
            if episode.key_points:
                entry += f"要点: {', '.join(episode.key_points)}\n"
            if episode.decisions:
                entry += f"决策: {', '.join(episode.decisions)}\n"
            
            entry += "\n"
            
            entry_tokens = len(entry) // 4
            if total_tokens + entry_tokens > max_tokens:
                break
            
            parts.append(entry)
            total_tokens += entry_tokens
        
        return "".join(parts)


def create_episode_from_consolidation(
    session_id: str,
    messages: list[dict],
    consolidation_result: dict,
) -> Episode:
    """
    从压缩结果创建Episode
    
    Args:
        session_id: 会话ID
        messages: 被压缩的消息列表
        consolidation_result: LLM压缩结果，包含topic, summary等
    
    Returns:
        新创建的Episode实例
    """
    now = datetime.now()
    
    time_start = messages[0].get("timestamp", now.isoformat()) if messages else now.isoformat()
    time_end = messages[-1].get("timestamp", now.isoformat()) if messages else now.isoformat()
    
    message_start = messages[0].get("index", 0) if messages else 0
    message_end = messages[-1].get("index", len(messages)) if messages else 0
    
    token_count = sum(
        len(str(m.get("content", ""))) // 4 
        for m in messages
    )
    
    return Episode(
        episode_id=f"ep_{now.strftime('%Y%m%d_%H%M%S')}_{hash(session_id) % 10000:04d}",
        session_id=session_id,
        created_at=now.isoformat(),
        trigger="consolidation",
        topic=consolidation_result.get("topic", ""),
        summary=consolidation_result.get("summary", ""),
        key_points=consolidation_result.get("key_points", []),
        decisions=consolidation_result.get("decisions", []),
        entities=consolidation_result.get("entities", {}),
        message_range={"start": message_start, "end": message_end},
        time_range={"start": time_start[:16], "end": time_end[:16]},
        token_count=token_count,
        importance=consolidation_result.get("importance", 3),
        tags=consolidation_result.get("tags", []),
        raw_messages=messages,
    )
