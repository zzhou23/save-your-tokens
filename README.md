<p align="center">
  <h1 align="center">save-your-tokens</h1>
  <p align="center">
    <strong>Application-layer context budget management for LLM applications</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
    <a href="https://github.com/zzhou23/save-your-tokens"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zzhou23/save-your-tokens?style=social"></a>
  </p>
</p>

---

**LLMs are stateless.** Every turn resends the full conversation history. In a typical 50-turn agentic coding session, **60%+ of the context window is redundant** — wasting money and degrading output quality through the ["lost in the middle"](https://arxiv.org/abs/2307.03172) effect.

`save-your-tokens` is an open-source framework that brings **virtual memory management** to LLM context windows. It works across models (Claude, OpenAI, DeepSeek, Gemini), across frameworks (LangChain, LlamaIndex, raw SDK), and across interfaces (CLI, Web, IDE plugins).

> Think of it this way: your OS doesn't load every file into RAM. Why should your LLM application load every message into the context window?

## The Problem

| Pain Point | Impact |
|------------|--------|
| **No budget awareness** | Context fills up silently until you hit the limit, then everything breaks |
| **Redundant context** | System prompts, tool definitions, and stale file contents resent every turn |
| **Framework lock-in** | LangChain has `SummarizationMiddleware`, but nothing works outside LangChain |
| **Model lock-in** | Claude Code has `/compact`, but it only works with Claude in Claude Code |
| **No lifecycle management** | Context grows monotonically — nothing gets unloaded when it's no longer needed |
| **Quality degradation** | Long contexts cause models to miss instructions, repeat themselves, or contradict earlier outputs |

**Existing solutions** are either framework-locked, single-point (compression only or counting only), or proprietary. No open-source project provides a unified, cross-model, cross-framework context management protocol.

## The Solution

`save-your-tokens` manages context through four strategies:

1. **Three-Layer Budget Protocol** — Every piece of context is classified into Persistent, Session, or Ephemeral layers, each with its own token budget
2. **Skills System** — On-demand loading and unloading of context modules with budget awareness
3. **Active Compaction** — Rule-based triggers that proactively compact context before hitting limits
4. **Tiered Overage Response** — Graduated response from warnings to auto-compaction to rejection

### Three-Layer Context Model

```
┌─────────────────────────────────────────────────────┐
│                   Context Window                     │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Persistent  │  │   Session    │  │ Ephemeral │  │
│  │   5-15%      │  │   20-40%     │  │ Remainder │  │
│  │             │  │              │  │           │  │
│  │ • CLAUDE.md │  │ • progress   │  │ • tool    │  │
│  │ • system    │  │ • todo state │  │   output  │  │
│  │   prompt    │  │ • key facts  │  │ • messages│  │
│  │ • tool defs │  │              │  │ • file    │  │
│  │ • user prefs│  │              │  │   contents│  │
│  └─────────────┘  └──────────────┘  └───────────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────────┐│
│  │             Output Reserve (20-30%)              ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### Escalating Compaction

When context pressure builds, `syt` responds with graduated actions:

```
Context usage:  ████████░░  80%  →  Drop stale ephemeral blocks
Context usage:  █████████░  90%  →  Summarize ephemeral content
Context usage:  ██████████  95%  →  Compact session layer
Context usage:  ██████████▓ 100% →  Force trim persistent (last resort)
```

### Tiered Budget Enforcement

| Overage | Response | Action |
|---------|----------|--------|
| Within budget | None | Continue normally |
| < 5% over | **Warn** | Log warning, continue |
| 5-20% over | **Auto-compact** | Trigger compaction strategy |
| > 20% over | **Reject** | Block new content until budget is freed |

## Quick Start

### Installation

```bash
# Core (no model dependencies)
pip install save-your-tokens

# With Claude support
pip install save-your-tokens[claude]

# With OpenAI support
pip install save-your-tokens[openai]

# Everything
pip install save-your-tokens[all]
```

### CLI Usage

```bash
# Initialize syt in your project
syt init --profile agentic

# Analyze context usage in current directory
syt analyze

# Output:
# === Context Budget Analysis ===
# Profile: agentic
# Context files found: 3
#
#   persistent:    1,250 /  24,000 tokens (5%) [OK]
#   session:       8,400 /  56,000 tokens (15%) [OK]
#   ephemeral:         0 /  80,000 tokens (0%) [OK]
#
#   Total: 9,650 / 160,000 (6%)

# Analyze a conversation log
syt analyze --log conversation.jsonl

# Compact context files in current directory
syt compact

# Compact a specific file
syt compact progress.md

# Auto-compact (for use in hooks, no confirmation prompt)
syt compact --auto
```

### Python API

```python
from save_your_tokens.core.spec import BUILTIN_PROFILES, ContextBlock, ContextLayer
from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.strategy import StrategyEngine

# Set up the budget engine
profile = BUILTIN_PROFILES["agentic"]  # persistent=15%, session=35%, output=20%
engine = BudgetEngine(context_window=200_000, profile=profile)

# Add context blocks
engine.add_block(ContextBlock(
    id="system-prompt",
    layer=ContextLayer.PERSISTENT,
    content="You are a helpful assistant...",
    token_count=500,
))

# Check budget status
usage = engine.compute_budgets()
print(f"Utilization: {usage.utilization:.0%}")

# Lifecycle management
lifecycle = LifecycleManager(engine)
lifecycle.start_session()

# After each turn, evaluate and auto-compact
result = lifecycle.post_turn(referenced_block_ids=["system-prompt"])
if result.needs_compaction:
    strategy = StrategyEngine(engine)
    strategy.execute_actions(result.recommended_actions)
```

### Claude Code Integration

`save-your-tokens` integrates with Claude Code via hooks and context file management:

```python
from save_your_tokens.integrations.claude_code import ClaudeCodeIntegration
from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.core.spec import BUILTIN_PROFILES

# Set up
engine = BudgetEngine(200_000, BUILTIN_PROFILES["agentic"])
lifecycle = LifecycleManager(engine)
strategy = StrategyEngine(engine)
integration = ClaudeCodeIntegration(engine, lifecycle, strategy, project_dir=".")

# Auto-scan CLAUDE.md, progress.md, etc.
integration.setup({})

# Generate hooks config for settings.json
hooks = integration.generate_hooks_config()
# Or write directly:
integration.write_hooks_config()
```

### Budget-Aware Skill Loading

```python
from save_your_tokens.skills.loader import SkillLoader, SkillMetadata
from save_your_tokens.skills.registry import SkillRegistry

# Discover available skills
registry = SkillRegistry()
registry.add_scan_dir("./skills")
registry.scan()

# Load skills with budget awareness
loader = SkillLoader(engine)

# Load a skill — returns overage level
overage = loader.load(
    name="debugging",
    content="When debugging, follow these steps...",
    token_count=1200,
)

# If budget is tight, syt will reject the load
if overage.value == "reject":
    print("Not enough budget to load this skill")

# Unload when done
loader.unload("debugging")

# Check what's loaded
print(loader.get_budget_summary())
```

## Architecture

```
save-your-tokens/
├── src/save_your_tokens/
│   ├── core/                    # The unique value — the protocol engine
│   │   ├── spec.py              # Pydantic data models for the three-layer protocol
│   │   ├── budget.py            # Budget engine: allocate, check, recommend
│   │   ├── lifecycle.py         # Lifecycle manager: session start/end, post-turn eval
│   │   └── strategy.py          # Strategy engine: compact triggers & actions
│   ├── skills/                  # On-demand skill loading with budget awareness
│   │   ├── loader.py            # Load/unload skills, budget-aware
│   │   └── registry.py          # Discover skills from directories
│   ├── adapters/                # Model-specific plugins
│   │   ├── base.py              # Abstract ModelAdapter interface
│   │   ├── claude.py            # Claude adapter (Anthropic SDK)
│   │   └── openai.py            # OpenAI adapter (tiktoken)
│   ├── integrations/            # Framework-specific glue
│   │   ├── base.py              # Abstract FrameworkIntegration interface
│   │   └── claude_code.py       # Claude Code hooks + file management
│   ├── reuse/                   # Thin wrappers — don't reinvent, wrap
│   │   ├── tokenizers.py        # Unified tokenizer interface
│   │   ├── compression.py       # Extractive summarization
│   │   └── observability.py     # Optional Langfuse integration
│   └── cli/
│       └── main.py              # `syt` CLI commands
├── docs/
│   ├── SPEC.md                  # Context Budget Specification
│   ├── DESIGN.md                # Architecture decisions
│   └── ADAPTERS.md              # Guide for writing adapters
└── tests/
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Cross-model** | `ModelAdapter` ABC — implement `count_tokens()` + `format_context()` for any model |
| **Cross-framework** | Two-layer integration: unified inner logic + framework-specific outer glue |
| **Minimal core deps** | stdlib + pydantic only. Model SDKs via optional extras |
| **Don't reinvent** | Wrap tiktoken for counting, wrap summarizers for compression, wrap Langfuse for observability |
| **Immutability** | Create new objects, never mutate existing ones |

### Plugin Interfaces

**ModelAdapter** — Add support for any model:

```python
class ModelAdapter(ABC):
    model_name: str                                    # e.g., "gpt-4o"
    context_window: int                                # e.g., 128000
    def count_tokens(self, text: str) -> int: ...      # Required
    def format_context(self, ...) -> list[dict]: ...   # Required
    def model_compact(self, ...) -> str | None: ...    # Optional: native summarization
```

**FrameworkIntegration** — Add support for any framework:

```python
class FrameworkIntegration(ABC):
    def setup(self, config: dict) -> None: ...                            # Install hooks/middleware
    def intercept_context(self, messages: list[dict]) -> list[dict]: ...  # Transform context
    def on_response(self, response: dict) -> None: ...                    # Post-response tracking
```

## Predefined Budget Profiles

| Profile | Persistent | Session | Output Reserve | Best For |
|---------|-----------|---------|---------------|----------|
| `chat` | 5% | 20% | 25% | Conversational chatbots |
| `agentic` | 15% | 35% | 20% | Agentic coding sessions (Claude Code, Cursor, etc.) |
| `rag` | 5% | 10% | 25% | RAG-based Q&A systems |

```bash
syt init --profile agentic  # Choose a profile
```

## Competitive Landscape

| Project | Scope | Limitation |
|---------|-------|------------|
| LangChain `SummarizationMiddleware` | Compression | LangChain only |
| Claude Code `/compact` | Compaction | Claude only, single interface |
| Various CLI proxy tools | Specific commands | Narrow scope |
| `context-engineering-toolkit` | Compression + priority assembly | No lifecycle, no skills, no cross-framework protocol |
| Langfuse | Observability | Complementary, not competitive |
| **save-your-tokens** | **Full lifecycle management** | **Cross-model, cross-framework, open protocol** |

Reference benchmark: JetBrains Research (March 2026) showed hybrid context management achieves 7-11% cost reduction with 2.6% accuracy improvement on SWE-bench.

## Benchmark Results

On a synthetic 50-turn agentic coding session (72 messages, ~45k tokens):

| Metric | Baseline | syt-Managed | Delta |
|--------|----------|-------------|-------|
| Cumulative tokens | 701,387 | 592,714 | **-15.5%** |
| Compaction events | 0 | 2 | automatic |
| Peak utilization | unbounded | 18.1% | budget-aware |

syt's staleness-based eviction automatically drops unreferenced ephemeral blocks, keeping context bounded while preserving important persistent and session context.

See [full benchmark details](docs/benchmark.md) for methodology, per-turn utilization chart, and reproduction steps.

## Roadmap

### Phase 1 — Core Protocol & CLI (Current)

- [x] Context Budget Specification (three-layer model)
- [x] Core engine: budget allocation, lifecycle management, strategy execution
- [x] Skills system: budget-aware loading/unloading
- [x] Model adapters: Claude, OpenAI
- [x] Framework integration: Claude Code (hooks + file management)
- [x] CLI: `syt init`, `syt analyze`, `syt compact`, `syt report`
- [x] Unit tests for core modules (98 tests, 97-100% coverage)
- [x] Integration tests for Claude Code workflow (16 integration + 19 CLI e2e)
- [x] Benchmark: [15.5% token savings](docs/benchmark.md) vs baseline (exceeds JetBrains 7-11%)
- [ ] Publish to PyPI as `0.1.0-alpha`

### Phase 2 — Ecosystem Expansion

- [ ] DeepSeek / Gemini model adapters
- [ ] LangChain + raw SDK framework integrations
- [ ] Web/GUI mode support
- [ ] Langfuse observability integration
- [ ] Independent `Compactor` interface (for local models)
- [ ] Additional benchmarks across model families

### Phase 3 — Community-Driven Growth

- [ ] More adapters: Qwen, Mistral, Llama, etc.
- [ ] More integrations: LlamaIndex, AutoGen, CrewAI, Dify, etc.
- [ ] IDE plugins (VS Code, JetBrains)
- [ ] Smart strategy engine with heuristics
- [ ] Multi-agent context sharing protocol

## Contributing

We welcome contributions! The easiest ways to get started:

1. **Add a model adapter** — Implement `ModelAdapter` for your favorite model ([guide](docs/ADAPTERS.md))
2. **Add a framework integration** — Implement `FrameworkIntegration` for your framework ([guide](docs/ADAPTERS.md))
3. **Improve compaction strategies** — Better summarization, smarter staleness detection
4. **Run benchmarks** — Test with your real-world workloads and share results

```bash
# Development setup
git clone https://github.com/zzhou23/save-your-tokens.git
cd save-your-tokens
pip install -e ".[dev]"

# Run tests
pytest

# Lint & format
ruff format . && ruff check .

# Type check
mypy src/
```

## License

[MIT](LICENSE)
