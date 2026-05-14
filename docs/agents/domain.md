# Domain Docs

How the engineering skills should consume this repo's domain documentation.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root
- **`docs/adr/`** — read ADRs that touch the area you're about to work in

If any of these files don't exist, proceed silently.

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── docs/adr/
│   └── ...
└── src/
```

## Use the glossary's vocabulary

When naming a domain concept (in an issue title, refactor proposal, hypothesis, or test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly:

> _Contradicts ADR-0001 (…) — but worth reopening because…_
