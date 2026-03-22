"""Generate a synthetic 50-turn agentic coding session for benchmarking.

Simulates a realistic Claude Code session: system setup, file reads, code edits,
test runs, and tool outputs with realistic token distributions.
"""

from __future__ import annotations

import json
from pathlib import Path

# Realistic content templates for each turn type
SYSTEM_PROMPT = (
    "You are Claude, an AI assistant by Anthropic. You help users with software engineering tasks. "
    "You have access to tools: Read, Write, Edit, Bash, Glob, Grep. "
    "Always read files before editing. Use immutable patterns. Follow PEP 8. "
    "Run tests after changes. Commit with conventional commit messages.\n"
) * 4  # ~2000 chars = ~500 tokens

CLAUDE_MD = (
    "# Project: web-dashboard\n\n"
    "## Architecture\n"
    "- Frontend: React + TypeScript + Tailwind\n"
    "- Backend: FastAPI + SQLAlchemy + PostgreSQL\n"
    "- Testing: pytest + vitest\n\n"
    "## Key Paths\n"
    "- src/api/ — FastAPI routes\n"
    "- src/models/ — SQLAlchemy models\n"
    "- src/services/ — Business logic\n"
    "- frontend/src/ — React components\n\n"
    "## Conventions\n"
    "- All endpoints return JSON with status field\n"
    "- Use dependency injection for services\n"
    "- Write tests first (TDD)\n"
    "- Conventional commits: feat/fix/refactor/test/docs\n"
) * 3  # ~1500 chars = ~375 tokens

PROGRESS_MD = (
    "## Current Status\n\n"
    "- Completed: User auth (JWT + refresh tokens)\n"
    "- Completed: Dashboard layout with sidebar\n"
    "- In progress: Task management CRUD\n\n"
    "## Key Decisions\n"
    "- Using SQLAlchemy 2.0 async sessions\n"
    "- Alembic for migrations\n"
    "- React Query for data fetching\n\n"
    "## Next Steps\n"
    "1. Create Task model\n"
    "2. Add CRUD endpoints\n"
    "3. Write frontend components\n"
    "4. Add tests\n"
) * 2  # ~800 chars = ~200 tokens


def _make_file_content(name: str, lines: int) -> str:
    """Generate fake file content of a specific size."""
    code_lines = [f"# {name}", ""]
    for i in range(lines):
        code_lines.append(f"    def method_{i}(self, arg_{i}: str) -> dict:")
        code_lines.append(f'        """Process {name} item {i}."""')
        code_lines.append(f"        result = self.repo.find(arg_{i})")
        code_lines.append(f"        return {{'id': arg_{i}, 'data': result}}")
        code_lines.append("")
    return "\n".join(code_lines)


def _make_tool_output(tool: str, size: int) -> str:
    """Generate fake tool output."""
    if tool == "test":
        lines = [f"PASSED tests/test_item_{i}.py::test_case_{i}" for i in range(size)]
        lines.append(f"\n{size} passed in 1.23s")
        return "\n".join(lines)
    if tool == "lint":
        return f"All checks passed! {size} files scanned, 0 errors."
    if tool == "grep":
        return "\n".join(
            [f"src/services/task.py:{i*10}: def process_task_{i}():" for i in range(size)]
        )
    return f"Command output ({size} lines)\n" + "\n".join(
        [f"line {i}: output data here" for i in range(size)]
    )


def generate_session() -> list[dict]:
    """Generate a 50-turn agentic session with realistic token distributions."""
    turns: list[dict] = []

    # Phase 1: Setup (turns 1-5) — system prompt, CLAUDE.md, initial task
    turns.append({
        "turn": 1,
        "role": "system",
        "type": "system_prompt",
        "layer": "persistent",
        "content": SYSTEM_PROMPT,
        "description": "System prompt with tool definitions",
    })
    turns.append({
        "turn": 2,
        "role": "system",
        "type": "context_file",
        "layer": "persistent",
        "content": CLAUDE_MD,
        "description": "CLAUDE.md project instructions",
    })
    turns.append({
        "turn": 3,
        "role": "system",
        "type": "context_file",
        "layer": "session",
        "content": PROGRESS_MD,
        "description": "progress.md session state",
    })
    turns.append({
        "turn": 4,
        "role": "user",
        "type": "message",
        "layer": "ephemeral",
        "content": "Add a Task model with CRUD endpoints. Use SQLAlchemy 2.0 async. "
        "Include title, description, status, created_at, updated_at fields. "
        "Write tests first.",
        "description": "User request",
    })
    turns.append({
        "turn": 5,
        "role": "assistant",
        "type": "message",
        "layer": "ephemeral",
        "content": "I'll implement the Task model with CRUD endpoints using TDD. Let me start by "
        "reading the existing models to understand the patterns, then write tests first.\n\n"
        "Plan:\n1. Read existing model structure\n2. Write test_task.py\n"
        "3. Create Task model\n4. Add CRUD endpoints\n5. Run tests\n6. Commit",
        "description": "Assistant plan",
    })

    # Phase 2: File reads and exploration (turns 6-20) — ephemeral, growing
    file_reads = [
        ("src/models/user.py", 40),
        ("src/models/base.py", 20),
        ("src/api/auth.py", 60),
        ("src/services/auth_service.py", 45),
        ("tests/test_auth.py", 50),
        ("src/api/router.py", 30),
        ("alembic/versions/001_users.py", 25),
        ("frontend/src/hooks/useAuth.ts", 35),
        ("src/config.py", 15),
        ("pyproject.toml", 20),
    ]

    for i, (filename, lines) in enumerate(file_reads):
        turn_num = 6 + i
        # Tool call
        turns.append({
            "turn": turn_num,
            "role": "assistant",
            "type": "tool_call",
            "layer": "ephemeral",
            "content": f"Reading {filename} to understand existing patterns...",
            "description": f"Tool call: Read {filename}",
        })
        # Tool result (file content)
        turns.append({
            "turn": turn_num,
            "role": "tool",
            "type": "tool_result",
            "layer": "ephemeral",
            "content": _make_file_content(filename, lines),
            "description": f"File content: {filename} ({lines} methods)",
        })

    # Phase 3: Code edits and test runs (turns 21-35) — ephemeral churn
    edits = [
        ("tests/test_task.py", 80, "Write Task model tests"),
        ("src/models/task.py", 50, "Create Task model"),
        ("src/api/tasks.py", 70, "Add CRUD endpoints"),
        ("src/services/task_service.py", 55, "Implement task service"),
        ("alembic/versions/002_tasks.py", 30, "Create migration"),
    ]

    turn_num = 21
    for filename, lines, desc in edits:
        # Write file
        turns.append({
            "turn": turn_num,
            "role": "assistant",
            "type": "tool_call",
            "layer": "ephemeral",
            "content": f"Creating {filename}: {desc}",
            "description": f"Write: {filename}",
        })
        turns.append({
            "turn": turn_num,
            "role": "tool",
            "type": "tool_result",
            "layer": "ephemeral",
            "content": _make_file_content(filename, lines),
            "description": f"Written: {filename}",
        })
        turn_num += 1

        # Run tests after each edit
        turns.append({
            "turn": turn_num,
            "role": "assistant",
            "type": "tool_call",
            "layer": "ephemeral",
            "content": "Running pytest to verify changes...",
            "description": "Bash: pytest",
        })
        turns.append({
            "turn": turn_num,
            "role": "tool",
            "type": "tool_result",
            "layer": "ephemeral",
            "content": _make_tool_output("test", 15 + turn_num),
            "description": "Test results",
        })
        turn_num += 1

    # Phase 4: More tool outputs and session wind-down (turns 36-50)
    turn_num = 36
    # Lint check
    turns.append({
        "turn": turn_num,
        "role": "assistant",
        "type": "tool_call",
        "layer": "ephemeral",
        "content": "Running ruff check to verify code quality...",
        "description": "Bash: ruff check",
    })
    turns.append({
        "turn": turn_num,
        "role": "tool",
        "type": "tool_result",
        "layer": "ephemeral",
        "content": _make_tool_output("lint", 25),
        "description": "Lint results",
    })
    turn_num += 1

    # Frontend work
    frontend_files = [
        ("frontend/src/components/TaskList.tsx", 60),
        ("frontend/src/components/TaskForm.tsx", 45),
        ("frontend/src/hooks/useTasks.ts", 30),
        ("frontend/src/types/task.ts", 15),
    ]

    for filename, lines in frontend_files:
        turns.append({
            "turn": turn_num,
            "role": "assistant",
            "type": "tool_call",
            "layer": "ephemeral",
            "content": f"Creating {filename}...",
            "description": f"Write: {filename}",
        })
        turns.append({
            "turn": turn_num,
            "role": "tool",
            "type": "tool_result",
            "layer": "ephemeral",
            "content": _make_file_content(filename, lines),
            "description": f"Written: {filename}",
        })
        turn_num += 1

    # More test runs
    for i in range(turn_num, 48):
        turns.append({
            "turn": i,
            "role": "assistant",
            "type": "tool_call",
            "layer": "ephemeral",
            "content": "Running full test suite...",
            "description": "Bash: pytest",
        })
        turns.append({
            "turn": i,
            "role": "tool",
            "type": "tool_result",
            "layer": "ephemeral",
            "content": _make_tool_output("test", 30 + i),
            "description": "Full test results",
        })

    # Final commit messages
    turns.append({
        "turn": 48,
        "role": "assistant",
        "type": "message",
        "layer": "ephemeral",
        "content": "All tests pass. Let me commit the changes with conventional commit messages.\n"
        "feat: add Task model with CRUD endpoints\n\n"
        "- Task model with title, description, status, timestamps\n"
        "- CRUD API endpoints at /api/tasks\n"
        "- Task service with business logic\n"
        "- Alembic migration\n"
        "- Frontend components: TaskList, TaskForm\n"
        "- Full test coverage",
        "description": "Commit message",
    })
    turns.append({
        "turn": 49,
        "role": "tool",
        "type": "tool_result",
        "layer": "ephemeral",
        "content": "[main abc1234] feat: add Task model with CRUD endpoints\n"
        " 8 files changed, 450 insertions(+)",
        "description": "Git commit result",
    })
    turns.append({
        "turn": 50,
        "role": "assistant",
        "type": "message",
        "layer": "ephemeral",
        "content": "Done! Task management is complete with full CRUD endpoints, "
        "frontend components, and test coverage.",
        "description": "Session conclusion",
    })

    return turns


def main() -> None:
    """Generate and save the synthetic dataset."""
    turns = generate_session()
    output_path = Path(__file__).parent / "data" / "agentic_session.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")

    total_chars = sum(len(t["content"]) for t in turns)
    print(f"Generated {len(turns)} messages across {turns[-1]['turn']} turns")
    print(f"Total content: {total_chars:,} chars (~{total_chars // 4:,} tokens)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
