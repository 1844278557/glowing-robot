# tests/agent/test_episode.py
"""Tests for Episode memory system: Episode, EpisodeStore, EpisodeRetriever."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.agent.episode import (
    Episode,
    EpisodeStore,
    EpisodeRetriever,
    create_episode_from_consolidation,
)


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

class TestEpisode:
    """Test Episode dataclass functionality."""

    def test_episode_creation_minimal(self):
        """Create episode with minimal required fields."""
        episode = Episode(
            episode_id="ep-001",
            session_id="session-abc",
            created_at="2026-01-15T10:30:00",
            trigger="consolidation",
            topic="Test Topic",
            summary="A test summary",
        )
        assert episode.episode_id == "ep-001"
        assert episode.importance == 3  # default
        assert episode.key_points == []
        assert episode.decisions == []
        assert episode.entities == {}
        assert episode.tags == []

    def test_episode_creation_full(self):
        """Create episode with all fields."""
        episode = Episode(
            episode_id="ep-002",
            session_id="session-xyz",
            created_at="2026-01-15T10:30:00",
            trigger="manual",
            topic="Full Episode",
            summary="Complete test",
            key_points=["point1", "point2"],
            decisions=["decision1"],
            entities={"person": ["Alice"], "project": ["nanobot"]},
            importance=5,
            tags=["important", "feature"],
            raw_messages=[{"role": "user", "content": "hello"}],
            message_range={"start": 0, "end": 5},
            time_range={"start": "2026-01-15 10:00", "end": "2026-01-15 10:30"},
            token_count=100,
        )
        assert episode.importance == 5
        assert len(episode.key_points) == 2
        assert episode.entities["person"] == ["Alice"]

    def test_episode_to_dict(self):
        """Episode can be serialized to dict."""
        episode = Episode(
            episode_id="ep-003",
            session_id="s1",
            created_at="2026-01-15T10:30:00",
            trigger="consolidation",
            topic="Serialization Test",
            summary="Testing to_dict",
        )
        data = episode.to_dict()
        assert data["episode_id"] == "ep-003"
        assert data["topic"] == "Serialization Test"
        assert "created_at" in data

    def test_episode_from_dict(self):
        """Episode can be deserialized from dict."""
        data = {
            "episode_id": "ep-004",
            "session_id": "s2",
            "created_at": "2026-01-15T10:30:00",
            "trigger": "consolidation",
            "topic": "Deserialization Test",
            "summary": "Testing from_dict",
            "importance": 4,
            "tags": ["test"],
        }
        episode = Episode.from_dict(data)
        assert episode.episode_id == "ep-004"
        assert episode.importance == 4
        assert episode.tags == ["test"]

    def test_episode_to_grep_text(self):
        """Episode generates grep-compatible text format."""
        episode = Episode(
            episode_id="ep-005",
            session_id="s3",
            created_at="2026-01-15T10:30:00",
            trigger="consolidation",
            topic="Grep Format Test",
            summary="Testing grep output",
            key_points=["point A", "point B"],
            time_range={"start": "2026-01-15 10:30", "end": "2026-01-15 11:00"},
        )
        text = episode.to_grep_text()
        assert "2026-01-15 10:30" in text
        assert "Grep Format Test" in text

    def test_episode_get_searchable_text(self):
        """Episode generates searchable text for vector search."""
        episode = Episode(
            episode_id="ep-006",
            session_id="s4",
            created_at="2026-01-15T10:30:00",
            trigger="consolidation",
            topic="Python Programming",
            summary="Discussion about coding",
            key_points=["Use type hints"],
            tags=["python"],
        )
        text = episode.get_searchable_text()
        assert "Python Programming" in text
        assert "Discussion about coding" in text
        assert "Use type hints" in text


# ---------------------------------------------------------------------------
# EpisodeStore
# ---------------------------------------------------------------------------

class TestEpisodeStore:
    """Test EpisodeStore CRUD operations and indexing."""

    @pytest.fixture()
    def store(self, tmp_path: Path):
        """Create an EpisodeStore with temporary directory."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        return EpisodeStore(memory_dir)

    @pytest.fixture()
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            episode_id="ep-001",
            session_id="session-1",
            created_at="2026-01-15T10:30:00",
            trigger="consolidation",
            topic="Sample Episode",
            summary="A sample for testing",
            key_points=["point1", "point2"],
            importance=4,
            tags=["test"],
            time_range={"start": "2026-01-15 10:00", "end": "2026-01-15 10:30"},
        )

    def test_save_episode(self, store: EpisodeStore, sample_episode: Episode):
        """Save episode creates JSON file."""
        store.save(sample_episode)

        ep_file = store._get_episode_path("ep-001")     
        assert ep_file.exists()
        data = json.loads(ep_file.read_text(encoding="utf-8"))

        assert data["episode_id"] == "ep-001"
        assert data["topic"] == "Sample Episode"

    def test_load_episode(self, store: EpisodeStore, sample_episode: Episode):
        """Load episode from file."""
        store.save(sample_episode)

        loaded = store.load("ep-001")
        assert loaded is not None
        assert loaded.episode_id == "ep-001"
        assert loaded.topic == "Sample Episode"

    def test_load_nonexistent_episode(self, store: EpisodeStore):
        """Load returns None for nonexistent episode."""
        loaded = store.load("nonexistent")
        assert loaded is None

    def test_list_episodes(self, store: EpisodeStore):
        """List all episodes."""
        for i in range(3):
            ep = Episode(
                episode_id=f"ep-{i:03d}",
                session_id="s1",
                created_at=f"2026-01-{15+i:02d}T10:00:00",
                trigger="consolidation",
                topic=f"Episode {i}",
                summary=f"Summary {i}",
                time_range={"start": f"2026-01-{15+i:02d} 10:00", "end": f"2026-01-{15+i:02d} 10:30"},
            )
            store.save(ep)

        episodes = store.get_recent_episodes(limit=10)
        assert len(episodes) == 3

    def test_delete_episode(self, store: EpisodeStore, sample_episode: Episode):
        """Delete episode removes file."""
        store.save(sample_episode)
        assert store.load("ep-001") is not None

        store.delete("ep-001")
        assert store.load("ep-001") is None

    def test_get_stats(self, store: EpisodeStore):
        """Get episode statistics."""
        for i in range(5):
            ep = Episode(
                episode_id=f"ep-{i}",
                session_id="s1",
                created_at=datetime.now().isoformat(),
                trigger="consolidation",
                topic=f"Topic {i}",
                summary=f"Summary {i}",
                importance=i + 1,
                time_range={"start": "2026-01-15 10:00", "end": "2026-01-15 10:30"},
            )
            store.save(ep)

        stats = store.get_stats()
        assert stats["total_count"] == 5

    def test_append_compatibility(self, store: EpisodeStore):
        """Append method creates episode from simple text (HISTORY.md compatibility)."""
        episode = store.append(
            entry="[2026-01-15 10:30] User discussed testing",
            session_id="session-compat"
        )

        assert episode.episode_id.startswith("ep_")
        assert episode.session_id == "session-compat"
        assert "testing" in episode.summary

    def test_update_index(self, store: EpisodeStore, sample_episode: Episode):
        """Index is updated when episode is saved."""
        store.save(sample_episode)

        assert len(store.index["episodes"]) == 1
        assert store.index["episodes"][0]["episode_id"] == "ep-001"

    def test_generate_episode_id(self, store: EpisodeStore):
        """Generate unique episode ID."""
        id1 = store.generate_episode_id()
        assert id1.startswith("ep_")
        assert len(id1) > 10  # Should have timestamp and sequence


# ---------------------------------------------------------------------------
# EpisodeRetriever
# ---------------------------------------------------------------------------

class TestEpisodeRetriever:
    """Test EpisodeRetriever search functionality."""

    @pytest.fixture()
    def store(self, tmp_path: Path):
        """Create a store with sample episodes."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        store = EpisodeStore(memory_dir)

        episodes = [
            Episode(
                episode_id="ep-python",
                session_id="s1",
                created_at="2026-01-10T10:00:00",
                trigger="consolidation",
                topic="Python Best Practices",
                summary="Discussion about Python coding standards",
                key_points=["Use type hints", "Follow PEP 8"],
                tags=["python", "coding"],
                importance=4,
                time_range={"start": "2026-01-10 10:00", "end": "2026-01-10 11:00"},
            ),
            Episode(
                episode_id="ep-database",
                session_id="s1",
                created_at="2026-01-12T14:00:00",
                trigger="consolidation",
                topic="Database Optimization",
                summary="How to optimize SQL queries",
                key_points=["Add indexes", "Use EXPLAIN"],
                tags=["database", "sql"],
                importance=5,
                time_range={"start": "2026-01-12 14:00", "end": "2026-01-12 15:00"},
            ),
            Episode(
                episode_id="ep-bug",
                session_id="s2",
                created_at="2026-01-15T09:00:00",
                trigger="consolidation",
                topic="Bug Fix: Login Issue",
                summary="Fixed authentication bug in login flow",
                key_points=["Session timeout was too short"],
                tags=["bug", "auth"],
                importance=3,
                time_range={"start": "2026-01-15 09:00", "end": "2026-01-15 09:30"},
            ),
        ]
        for ep in episodes:
            store.save(ep)

        return store

    @pytest.fixture()
    def retriever(self, store: EpisodeStore):
        """Create a retriever for the store."""
        return EpisodeRetriever(store)

    def test_search_by_keyword(self, retriever: EpisodeRetriever):
        """Search finds episodes by keyword."""
        results = retriever.search(query="python")
        assert len(results) == 1
        assert results[0].episode_id == "ep-python"

    def test_search_case_insensitive(self, retriever: EpisodeRetriever):
        """Search is case insensitive."""
        results = retriever.search(query="PYTHON")
        assert len(results) == 1
        assert results[0].episode_id == "ep-python"

    def test_search_by_tag(self, retriever: EpisodeRetriever):
        """Search filters by tag."""
        results = retriever.search(query="", tags=["bug"])
        assert len(results) == 1
        assert results[0].episode_id == "ep-bug"

    def test_search_by_importance(self, retriever: EpisodeRetriever):
        """Search filters by minimum importance."""
        results = retriever.search(query="", min_importance=4)
        assert len(results) == 2
        ids = {ep.episode_id for ep in results}
        assert "ep-python" in ids
        assert "ep-database" in ids
        assert "ep-bug" not in ids

    def test_search_limit(self, retriever: EpisodeRetriever):
        """Search respects limit."""
        results = retriever.search(query="", limit=2)
        assert len(results) == 2

    def test_search_no_results(self, retriever: EpisodeRetriever):
        """Search returns empty list when no matches."""
        results = retriever.search(query="nonexistent_topic_xyz")
        assert results == []

    def test_grep_files(self, retriever: EpisodeRetriever):
        """Grep files finds text in episode files."""
        results = retriever.grep_files("SQL")
        assert len(results) >= 1

    def test_get_context_for_prompt(self, retriever: EpisodeRetriever):
        """Get context for prompt returns formatted text."""
        context = retriever.get_context_for_prompt("python")
        assert "Python Best Practices" in context


# ---------------------------------------------------------------------------
# create_episode_from_consolidation
# ---------------------------------------------------------------------------

class TestCreateEpisodeFromConsolidation:
    """Test episode creation from consolidation result."""

    def test_create_from_consolidation(self):
        """Create episode from consolidation result."""
        messages = [
            {"role": "user", "content": "How do I optimize SQL?", "timestamp": "2026-01-15T10:00:00"},
            {"role": "assistant", "content": "You can add indexes...", "timestamp": "2026-01-15T10:30:00"},
        ]
        consolidation_result = {
            "topic": "SQL Optimization Discussion",
            "summary": "User asked about SQL optimization",
            "key_points": ["Add indexes", "Use EXPLAIN"],
            "decisions": ["Will add index on user_id"],
            "entities": {"database": ["PostgreSQL"]},
            "importance": 4,
            "tags": ["database", "optimization"],
        }

        episode = create_episode_from_consolidation(
            session_id="session-test",
            messages=messages,
            consolidation_result=consolidation_result,
        )

        assert episode.session_id == "session-test"
        assert episode.topic == "SQL Optimization Discussion"
        assert episode.summary == "User asked about SQL optimization"
        assert len(episode.key_points) == 2
        assert episode.importance == 4
        assert episode.tags == ["database", "optimization"]
        assert len(episode.raw_messages) == 2

    def test_create_with_minimal_fields(self):
        """Create episode with only required fields."""
        messages = [{"role": "user", "content": "Hello"}]
        consolidation_result = {
            "topic": "Greeting",
            "summary": "User said hello",
        }

        episode = create_episode_from_consolidation(
            session_id="s1",
            messages=messages,
            consolidation_result=consolidation_result,
        )

        assert episode.topic == "Greeting"
        assert episode.summary == "User said hello"
        assert episode.importance == 3  # default
        assert episode.key_points == []

    def test_create_with_dict_key_points(self):
        """Handle dict values in key_points (serialize to JSON)."""
        messages = []
        consolidation_result = {
            "topic": "Complex Data",
            "summary": "Test",
            "key_points": [{"detail": "nested data"}],
        }

        episode = create_episode_from_consolidation(
            session_id="s1",
            messages=messages,
            consolidation_result=consolidation_result,
        )

        assert len(episode.key_points) == 1
        assert isinstance(episode.key_points[0], dict)


# ---------------------------------------------------------------------------
# Episode Tools
# ---------------------------------------------------------------------------

class TestSearchEpisodesTool:
    """Test SearchEpisodesTool."""

    @pytest.fixture()
    def tool(self, tmp_path: Path):
        """Create a SearchEpisodesTool with sample data."""
        from nanobot.agent.tools.episode import SearchEpisodesTool

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        store = EpisodeStore(memory_dir)

        ep = Episode(
            episode_id="ep-test",
            session_id="s1",
            created_at=datetime.now().isoformat(),
            trigger="consolidation",
            topic="Test Topic",
            summary="Test summary with keyword",
            importance=3,
            time_range={"start": "2026-01-15 10:00", "end": "2026-01-15 10:30"},
        )
        store.save(ep)

        return SearchEpisodesTool(store)

    @pytest.mark.asyncio
    async def test_execute_with_query(self, tool):
        """Execute search with query returns formatted results."""
        result = await tool.execute(query="keyword")
        assert "Test Topic" in result
        assert "keyword" in result
        assert "1 条" in result

    @pytest.mark.asyncio
    async def test_execute_no_results(self, tool):
        """Execute search with no matches returns message."""
        result = await tool.execute(query="nonexistent_xyz")
        assert "未找到" in result or "No episodes" in result.lower()


class TestListRecentEpisodesTool:
    """Test ListRecentEpisodesTool."""

    @pytest.fixture()
    def tool(self, tmp_path: Path):
        """Create a ListRecentEpisodesTool with sample data."""
        from nanobot.agent.tools.episode import ListRecentEpisodesTool

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        store = EpisodeStore(memory_dir)

        for i in range(5):
            ep = Episode(
                episode_id=f"ep-{i}",
                session_id="s1",
                created_at=f"2026-01-15T{10+i:02d}:00:00",
                trigger="consolidation",
                topic=f"Episode {i}",
                summary=f"Summary {i}",
                importance=i + 1,
                time_range={"start": f"2026-01-15 {10+i:02d}:00", "end": f"2026-01-15 {10+i:02d}:30"},
            )
            store.save(ep)

        return ListRecentEpisodesTool(store)

    @pytest.mark.asyncio
    async def test_execute_default_limit(self, tool):
        """Execute lists recent episodes with default limit."""
        result = await tool.execute()
        assert "Episode" in result

    @pytest.mark.asyncio
    async def test_execute_custom_limit(self, tool):
        """Execute with custom limit."""
        result = await tool.execute(limit=2)
        assert result.count("Episode") <= 2


class TestGetEpisodeStatsTool:
    """Test GetEpisodeStatsTool."""

    @pytest.fixture()
    def tool(self, tmp_path: Path):
        """Create a GetEpisodeStatsTool with sample data."""
        from nanobot.agent.tools.episode import GetEpisodeStatsTool

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        store = EpisodeStore(memory_dir)

        for i in range(3):
            ep = Episode(
                episode_id=f"ep-{i}",
                session_id="s1",
                created_at=datetime.now().isoformat(),
                trigger="consolidation",
                topic=f"Topic {i}",
                summary=f"Summary {i}",
                importance=i + 2,  # 2, 3, 4
                time_range={"start": "2026-01-15 10:00", "end": "2026-01-15 10:30"},
            )
            store.save(ep)

        return GetEpisodeStatsTool(store)

    @pytest.mark.asyncio
    async def test_execute_returns_stats(self, tool):
        """Execute returns episode statistics."""
        result = await tool.execute()
        assert "3" in result  # total count
