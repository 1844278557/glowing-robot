# nanobot/agent/tools/episode.py
"""
Episode检索工具 - 供LLM自主判断是否需要检索历史记忆

设计理念：
- 作为Tool注册到工具系统，LLM可自主调用
- 类似原有的grep工具，由LLM判断是否需要检索
- 支持多种检索方式：关键词、时间范围、标签、语义
"""

from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.episode import EpisodeStore, EpisodeRetriever


class SearchEpisodesTool(Tool):
    """
    Episode检索工具
    
    让LLM能够自主检索历史记忆，判断是否需要回顾过去的对话。
    支持多种检索维度：
    - query: 关键词/语义搜索
    - time_start/time_end: 时间范围
    - tags: 标签过滤
    - min_importance: 重要性过滤
    """
    
    def __init__(self, store: EpisodeStore):
        """
        初始化检索工具
        
        Args:
            store: Episode存储管理器实例
        """
        self._store = store
        self._retriever = EpisodeRetriever(store)
    
    @property
    def name(self) -> str:
        return "search_episodes"
    
    @property
    def description(self) -> str:
        return (
            "搜索历史记忆(Episodes)。"
            "当你需要回顾过去的对话、查找之前讨论过的内容、"
            "或者用户提到'之前'、'上次'、'记得吗'等关键词时，"
            "使用此工具检索相关历史记忆。"
            "支持关键词搜索、时间范围过滤、标签过滤。"
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索查询文本。可以是关键词、短语或问题描述。"
                        "例如：'用户偏好设置'、'上次讨论的项目架构'"
                    ),
                },
                "time_start": {
                    "type": "string",
                    "description": (
                        "起始时间，格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM。"
                        "例如：'2024-01-01'、'2024-01-15 10:00'"
                    ),
                },
                "time_end": {
                    "type": "string",
                    "description": (
                        "结束时间，格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM。"
                        "例如：'2024-01-31'、'2024-01-31 23:59'"
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "标签过滤列表。只返回包含这些标签的记忆。"
                        "例如：['项目', '决策']"
                    ),
                },
                "min_importance": {
                    "type": "integer",
                    "description": "最小重要性等级(1-5)，默认1。5最重要。",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制，默认5。",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
        }
    
    async def execute(
        self,
        query: str | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        tags: list[str] | None = None,
        min_importance: int = 1,
        limit: int = 5,
        **kwargs: Any,
    ) -> str:
        """
        执行Episode检索
        
        Args:
            query: 搜索查询
            time_start: 起始时间
            time_end: 结束时间
            tags: 标签列表
            min_importance: 最小重要性
            limit: 结果数量限制
        
        Returns:
            格式化的检索结果文本
        """
        if not query:
            return "错误：请提供搜索查询(query参数)"
        
        try:
            episodes = self._retriever.search(
                query=query,
                time_start=time_start,
                time_end=time_end,
                tags=tags,
                min_importance=min_importance,
                limit=limit,
                use_vector=True,
            )
            
            if not episodes:
                return self._format_no_result(query, time_start, time_end, tags)
            
            return self._format_results(episodes, query)
            
        except Exception as e:
            return f"搜索历史记忆时出错: {e}"
    
    def _format_no_result(
        self,
        query: str,
        time_start: str | None,
        time_end: str | None,
        tags: list[str] | None,
    ) -> str:
        """格式化无结果时的响应"""
        parts = [f"未找到与 '{query}' 相关的历史记忆。"]
        
        filters = []
        if time_start:
            filters.append(f"起始时间: {time_start}")
        if time_end:
            filters.append(f"结束时间: {time_end}")
        if tags:
            filters.append(f"标签: {', '.join(tags)}")
        
        if filters:
            parts.append("过滤条件: " + "; ".join(filters))
        
        parts.append("建议：尝试放宽搜索条件或使用不同的关键词。")
        return "\n".join(parts)
    
    def _format_results(self, episodes: list, query: str) -> str:
        """格式化检索结果"""
        lines = [
            f"找到 {len(episodes)} 条与 '{query}' 相关的历史记忆：",
            "",
            "---",
        ]
        
        for i, episode in enumerate(episodes, 1):
            lines.append(f"### 记忆 #{i}")
            lines.append(f"**时间**: {episode.time_range.get('start', '未知')}")
            lines.append(f"**主题**: {episode.topic}")
            lines.append(f"**摘要**: {episode.summary}")
            
            if episode.key_points:
                lines.append(f"**要点**: {', '.join(episode.key_points)}")
            
            if episode.decisions:
                lines.append(f"**决策**: {', '.join(episode.decisions)}")
            
            if episode.tags:
                lines.append(f"**标签**: {', '.join(episode.tags)}")
            
            lines.append(f"**重要性**: {'⭐' * episode.importance}")
            lines.append("---")
        
        stats = self._store.get_stats()
        lines.append("")
        lines.append(f"(共  {stats.get('total_count', 0)} 条记忆)")


        
        return "\n".join(lines)


class ListRecentEpisodesTool(Tool):
    """
    列出最近的Episode工具
    
    快速查看最近的记忆，无需搜索关键词
    """
    
    def __init__(self, store: EpisodeStore):
        self._store = store
    
    @property
    def name(self) -> str:
        return "list_recent_episodes"
    
    @property
    def description(self) -> str:
        return (
            "列出最近的历史记忆。"
            "当你想快速了解最近的对话内容，"
            "或者用户问'最近做了什么'时使用。"
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "返回数量，默认5",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": [],
        }
    
    async def execute(self, limit: int = 5, **kwargs: Any) -> str:
        """列出最近的Episode"""
        try:
            episodes = self._store.get_recent_episodes(limit=limit)
            
            if not episodes:
                return "暂无历史记忆。"
            
            lines = [f"最近 {len(episodes)} 条历史记忆：", ""]
            
            for i, episode in enumerate(episodes, 1):
                time_str = episode.time_range.get("start", "未知时间")
                lines.append(f"{i}. [{time_str}] {episode.topic}")
                lines.append(f"   {episode.summary[:100]}{'...' if len(episode.summary) > 100 else ''}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"获取最近记忆时出错: {e}"


class GetEpisodeStatsTool(Tool):
    """
    获取Episode统计信息工具
    
    查看记忆系统的整体状态
    """
    
    def __init__(self, store: EpisodeStore):
        self._store = store
    
    @property
    def name(self) -> str:
        return "get_episode_stats"
    
    @property
    def description(self) -> str:
        return (
            "获取历史记忆的统计信息。"
            "查看总记忆数量、最后更新时间等。"
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }
    
    async def execute(self, **kwargs: Any) -> str:
        """获取统计信息"""
        try:
            stats = self._store.get_stats()
            
            lines = ["## 历史记忆统计", ""]
            lines.append(f"- 总记忆数: {stats.get('total_count', 0)}")
            lines.append(f"- 最后更新: {stats.get('last_updated', '未知')}")
            
            vector_status = "已启用" if self._store._vector_store else "未启用"
            lines.append(f"- 向量检索: {vector_status}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"获取统计信息时出错: {e}"
