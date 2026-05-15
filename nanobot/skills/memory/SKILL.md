---
name: memory
description: Two-layer memory system with Episode-based recall.
always: true
---

# Memory

## Structure

- `memory/MEMORY.md` — Long-term facts (preferences, project context, relationships). Always loaded into your context.
- `memory/episodes/` — Structured conversation history (Episodes). NOT loaded into context. Use the `search_episodes` tool to search past conversations.

## Search Past Episodes

Use the `search_episodes` tool when you need to recall past conversations:

```
search_episodes(query="keyword or topic", time_start="2024-01-01", time_end="2024-12-31")
```

The tool supports:
- **Keyword search**: Find episodes by topic, summary, or key points
- **Time range filter**: Narrow down to a specific date range
- **Tag filter**: Search by tags like "bug", "feature", "discussion"

## Episode Structure

Each Episode contains:
- **topic**: Main topic of the conversation
- **summary**: Brief summary of what was discussed
- **key_points**: Important points from the conversation
- **decisions**: Any decisions made
- **entities**: People, projects, or concepts mentioned
- **importance**: 1-5 rating (higher = more important)
- **tags**: Categorical labels

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`:
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")

## Auto-consolidation

Old conversations are automatically summarized and stored as Episodes when the session grows large. Long-term facts are extracted to MEMORY.md. You don't need to manage this.
