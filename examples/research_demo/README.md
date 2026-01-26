# Research Demo

A demonstration CLI application showcasing all P0/P1 features of the agentic-cli framework.

## Overview

The Research Demo is a research assistant that helps you explore topics while demonstrating:

- **Memory System**: Session-scoped working memory + persistent long-term memory
- **Task Planning**: Hierarchical task graphs with dependencies
- **File Operations**: Save, read, and compare research findings
- **Shell Commands**: Safe command execution with blocking of dangerous patterns
- **Human-in-the-Loop (HITL)**: Approval gates and review checkpoints

## Running the Demo

```bash
# From the project root
conda run -n agenticcli python -m examples.research_demo
```

## Features

### Memory

The agent can store and recall information across two tiers:

**Working Memory (Session)**
- Stores context for the current session
- Cleared when the app exits
- Use `/memory` to view current state

**Long-term Memory (Persistent)**
- Stores learnings, facts, preferences, and references
- Persists to `~/.research_demo/memory/longterm.json`
- Survives across sessions

### Planning

The agent can create structured research plans:

- Tasks with descriptions and dependencies
- Status tracking (pending, in_progress, completed, failed)
- Visual progress display via `/plan` command

### File Operations

- Save research findings to `~/.research_demo/findings/`
- Compare document versions with diff
- List and read saved files

### Shell Commands

- Run safe shell commands (ls, cat, grep, etc.)
- Dangerous commands are blocked (rm -rf, etc.)
- Timeout protection (30 seconds default)

### HITL

- Approval rules for sensitive operations
- Auto-approve patterns for safe commands
- Checkpoints for reviewing agent outputs

## Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/memory` | `/mem`, `/m` | Show working and long-term memory |
| `/plan` | `/tasks`, `/p` | Show current task graph |
| `/files` | `/ls`, `/f` | List files in workspace |
| `/approvals` | `/approve`, `/a` | Show pending approvals |
| `/checkpoints` | `/cp`, `/review` | Show checkpoints awaiting review |
| `/clear-memory` | `/clearmem` | Clear working memory |
| `/clear-plan` | `/clearplan` | Clear task plan |
| `/help` | `/h`, `/?` | Show all commands |
| `/settings` | `/s` | View/change settings |
| `/exit` | `/quit`, `/q` | Exit the application |

## Example Session

```
>>> Research the history of Python programming language

[Agent creates a task plan, stores context, and begins researching]

>>> /plan

Progress: 2/5 completed, 1 in progress

✓ Gather initial sources
✓ Review key events (depends on: Gather initial)
◐ Document milestones (depends on: Review key events)
☐ Write summary (depends on: Document milestones)
☐ Create final report (depends on: Write summary)

>>> /memory

Working Memory (Session)
┌─────────────────┬───────────────────────┬─────────┐
│ Key             │ Value                 │ Tags    │
├─────────────────┼───────────────────────┼─────────┤
│ research_topic  │ Python history        │ planning│
│ current_task    │ Document milestones   │         │
└─────────────────┴───────────────────────┴─────────┘

Long-term Memory (Persistent)
┌──────────┬──────────┬────────────────────────────────────────────┬──────┐
│ ID       │ Type     │ Content                                    │ Tags │
├──────────┼──────────┼────────────────────────────────────────────┼──────┤
│ a1b2c3d4 │ fact     │ Python was created by Guido van Rossum...  │      │
│ e5f6g7h8 │ learning │ Python 2.0 introduced list comprehensions  │      │
└──────────┴──────────┴────────────────────────────────────────────┴──────┘

>>> /files

Files in findings/
┌─────────────────────────┬──────┬─────────────┐
│ Name                    │ Type │ Size        │
├─────────────────────────┼──────┼─────────────┤
│ python_history_draft.md │ file │ 2,456 bytes │
│ key_milestones.txt      │ file │ 1,234 bytes │
└─────────────────────────┴──────┴─────────────┘
Total: 2 items
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearchDemoApp                          │
│  (extends BaseCLIApp)                                       │
├─────────────────────────────────────────────────────────────┤
│  Commands:                   │  Agent Tools:                │
│  /memory - show memory state │  remember_context()          │
│  /plan - show task graph     │  recall_info()               │
│  /approvals - pending list   │  create_research_plan()      │
│  /checkpoints - review list  │  update_task_status()        │
│  /files - list artifacts     │  save_finding()              │
│  /clear-memory               │  compare_versions()          │
│  /clear-plan                 │  run_safe_command()          │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Settings are loaded from environment variables or `~/.research_demo/.env`:

- `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY` - LLM provider API key
- `RESEARCH_DEMO_WORKSPACE_DIR` - Workspace directory (default: `~/.research_demo`)

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main CLI application class |
| `settings.py` | Demo app settings |
| `agents.py` | Agent configuration with tools |
| `tools.py` | Tools wrapping new features |
| `commands.py` | Status commands |
| `__main__.py` | Entry point |
