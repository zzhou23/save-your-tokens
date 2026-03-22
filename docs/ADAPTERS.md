# Writing Adapters

> Guide for contributing new model adapters and framework integrations.

## ModelAdapter

Implement `save_your_tokens.adapters.base.ModelAdapter`:

- `model_name` (property): Human-readable model name
- `context_window` (property): Max context window in tokens
- `count_tokens(text)`: Accurate token counting
- `format_context(persistent, session, ephemeral)`: Assemble into model-native messages
- `model_compact(content, target_tokens)` (optional): Native model compaction

## FrameworkIntegration

Implement `save_your_tokens.integrations.base.FrameworkIntegration`:

- `setup(config)`: Install hooks/middleware
- `teardown()`: Clean up
- `intercept_context(messages)`: Transform context before sending
- `on_response(response)`: Handle model response

Inner layer logic (budget check, compact orchestration) is inherited from the base class.
