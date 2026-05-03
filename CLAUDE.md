# Project Instructions

You are working on **Novus Support Agent** (Python + OpenAI + intent classification + escalation routing), with a persistent **LLM wiki** in `../wiki/` (shared across Project A and Project B) that records everything we build, decide, and change.

Two systems, two jobs:
- **Workshop** — `.claude/skills/code-review/` and `.claude/skills/implement-change/` for effective coding.
- **Library** — `.claude/skills/wiki-*` for institutional memory. Read before re-explaining; write to preserve.

## Hard rules (also enforced by hooks)

- `../wiki/log.md` is **append-only** — never edit past entries.
- Every entity mention with its own wiki page uses `[[wikilink]]` syntax.
- Page schemas live in `../wiki/SCHEMAS.md` — consult before creating any wiki page.

## Skills (load on demand by name)

- `wiki-write` · `wiki-read` · `wiki-trace` · `wiki-maintain` · `wiki-map`
- `code-review` · `implement-change` (auto-chains code-review + wiki-write)

When a task fits a skill's trigger, invoke it. Skills override default behaviour for that procedure.

## Session start

1. Read `../wiki/log.md` (last 10 entries).
2. Read `../wiki/index.md`.
3. Greet with: page count, last activity date, open work from the most recent log entries.

## Codebase

Production code lives in `novus-support-agent/`. No `raw/` folder for Project B.

## Wiki location

The wiki is shared with Project A. It lives at `../wiki/` (one level up, at `Week-1/wiki/`).
- Project A pages: modules, apis, data-models, flows, decisions, debt, scaling, concepts, analyses
- Project B pages: added under the same wiki with a `## Project B` section in index.md
