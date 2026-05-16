"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import inspect
import json
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.episode import Episode, EpisodeStore, create_episode_from_consolidation
from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Brief topic/title of this conversation segment (5-10 words).",
                    },
                    "summary": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Include detail useful for future retrieval.",
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key points discussed (3-5 items).",
                    },
                    "decisions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of decisions made or conclusions reached.",
                    },
                    "entities": {
                        "type": "object",
                        "description": "Key entities mentioned: {\"people\": [...], \"projects\": [...], \"concepts\": [...]}",
                    },
                    "importance": {
                        "type": "integer",
                        "description": "Importance level 1-5 (5=most important). Default 3.",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization (e.g., ['project', 'debugging']).",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["topic", "summary", "memory_update"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """
    Two-layer memory system:
    - MEMORY.md: Long-term facts (unchanged)
    - Episodes: Structured conversation history (replaces HISTORY.md)
    """

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.history_file = self.memory_dir / "HISTORY.md"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.episode_store = EpisodeStore(self.memory_dir)
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, content: Any) -> None:
        """为旧集成追加一条历史记录。"""
        text = _ensure_text(content).strip()
        if not text:
            return
        previous = self.history_file.read_text(encoding="utf-8") if self.history_file.exists() else ""
        separator = "\n" if previous and not previous.endswith("\n") else ""
        self.history_file.write_text(f"{previous}{separator}{text}\n", encoding="utf-8")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        session_id: str = "unknown",
    ) -> bool:
        """
        Consolidate the provided message chunk into MEMORY.md + Episode.

        Args:
            messages: Messages to consolidate
            provider: LLM provider for summarization
            model: Model to use
            session_id: Session identifier for the Episode
        """
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Analyze the conversation and call the save_memory tool with structured information including topic, summary, key points, decisions, entities, importance, and tags."},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages, session_id)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages, session_id)

            legacy_history = args.get("history_entry")
            has_structured_summary = "topic" in args and "summary" in args
            if (
                "memory_update" not in args
                or (legacy_history is None and not has_structured_summary)
            ):
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages, session_id)

            update = args.get("memory_update")
            if update is None:
                logger.warning("Memory consolidation: memory_update is empty")
                return self._fail_or_raw_archive(messages, session_id)

            if legacy_history is not None:
                history_text = _ensure_text(legacy_history).strip()
                if isinstance(legacy_history, dict):
                    summary_source = legacy_history.get("summary", legacy_history)
                    topic_source = legacy_history.get("topic", "Conversation history")
                else:
                    summary_source = legacy_history
                    topic_source = args.get("topic", "Conversation history")
                topic = _ensure_text(topic_source).strip()
                summary = _ensure_text(summary_source).strip()
            else:
                history_text = ""
                topic_source = args.get("topic")
                summary_source = args.get("summary")
                if topic_source is None or summary_source is None:
                    logger.warning("Memory consolidation: topic or summary is empty")
                    return self._fail_or_raw_archive(messages, session_id)
                topic = _ensure_text(topic_source).strip()
                summary = _ensure_text(summary_source).strip()

            if not topic or not summary or (legacy_history is not None and not history_text):
                logger.warning("Memory consolidation: required save_memory value is empty")
                return self._fail_or_raw_archive(messages, session_id)

            consolidation_result = {
                "topic": topic,
                "summary": summary,
                "key_points": args.get("key_points", []),
                "decisions": args.get("decisions", []),
                "entities": args.get("entities", {}),
                "importance": args.get("importance", 3),
                "tags": args.get("tags", []),
            }

            episode = create_episode_from_consolidation(
                session_id=session_id,
                messages=messages,
                consolidation_result=consolidation_result,
            )
            self.episode_store.save(episode)
            self.append_history(legacy_history if legacy_history is not None else consolidation_result)

            update = _ensure_text(update)
            if update != current_memory:
                self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info(
                "Memory consolidation done: episode {} created for {} messages",
                episode.episode_id,
                len(messages),
            )
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages, session_id)

    def _fail_or_raw_archive(self, messages: list[dict], session_id: str = "unknown") -> bool:
        """Increment failure count; after threshold, raw-archive messages as Episode."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages, session_id)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict], session_id: str = "unknown") -> None:
        """Fallback: create a raw Episode without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw_lines = [
            f"[RAW] {len(messages)} messages archived without summarization",
            self._format_messages(messages),
        ]
        self.append_history("\n".join(line for line in raw_lines if line))
        episode = Episode(
            episode_id=self.episode_store.generate_episode_id(),
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            trigger="raw_archive",
            topic="[RAW] Unprocessed messages",
            summary=f"{len(messages)} messages archived without summarization",
            key_points=[],
            decisions=[],
            entities={},
            message_range={"start": 0, "end": len(messages)},
            time_range={"start": ts, "end": ts},
            token_count=sum(len(str(m.get("content", ""))) // 4 for m in messages),
            importance=1,
            tags=["raw", "fallback"],
            raw_messages=messages,
        )
        self.episode_store.save(episode)
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages as episode {}",
            len(messages),
            episode.episode_id,
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(
        self,
        messages: list[dict[str, object]],
        session_id: str = "unknown",
    ) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.store.consolidate(messages, self.provider, self.model, session_id)

    async def _call_consolidate_messages(
        self,
        messages: list[dict[str, object]],
        session_id: str,
    ) -> bool:
        """调用记忆整理，同时兼容旧的一参数替身函数。"""
        try:
            parameters = inspect.signature(self.consolidate_messages).parameters
        except (TypeError, ValueError):
            parameters = {}
        if len(parameters) <= 1:
            return await self.consolidate_messages(messages)
        return await self.consolidate_messages(messages, session_id)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(
        self,
        messages: list[dict[str, object]],
        session_id: str = "unknown",
    ) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self._call_consolidate_messages(messages, session_id):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < budget:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self._call_consolidate_messages(chunk, session.key):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
