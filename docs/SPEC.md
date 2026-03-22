# Context Budget Specification

> The formal protocol for three-layer context management. To be written in Phase 1.

## Three-Layer Context Model

| Layer | Lifetime | Budget |
|-------|----------|--------|
| Persistent | Cross-session | 5-15% of window |
| Session | Single session | 20-40% of window |
| Ephemeral | Single/few turns | Remainder |

## Classification Rules

Hybrid mode: convention-based defaults + user override via metadata.

## Budget Formula

Reserve 20-30% for output first. Then allocate: persistent → session → ephemeral gets rest.

## Compact Triggers

- Token threshold (>80% of window)
- Turn interval (every N turns)
- Staleness (unreferenced content older than M turns)
- Quality degradation (repetition/contradiction detected)

## Compact Actions (Escalating)

1. Drop stale ephemeral
2. Summarize ephemeral
3. Compact session
4. Force trim persistent (last resort)

## Overage Response (Tiered)

- Within budget: no action
- <5% over: warn
- 5-20% over: auto-compact
- >20% over: reject
