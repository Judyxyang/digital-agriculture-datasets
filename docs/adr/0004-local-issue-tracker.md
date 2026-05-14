# ADR-0004 — Local markdown issue tracker

**Date:** 2026-05-14  
**Status:** Accepted

## Context

The project needs an issue tracker to manage the 8 implementation issues derived from the PRD. Options include GitHub Issues, Jira/Linear, and local markdown files.

## Decision

Use **local markdown files** under `.scratch/banana-ripeness/issues/`. Each issue is a `.md` file with a `Status:` frontmatter field that tracks triage state using the canonical label vocabulary.

## Rationale

- The project is developed by a single agent + one human — a full issue tracker adds friction without benefit
- Local markdown files are version-controlled alongside the code, keeping issues and implementation in the same commit history
- The mattpocock agent skills (`triage`, `to-issues`, `diagnose`) read and write `.scratch/` directly
- No external service dependency; works offline

## Conventions

```
.scratch/banana-ripeness/
├── PRD.md
└── issues/
    ├── 01-dataset-loader.md    # Status: done
    ├── 02-image-preprocessor.md
    └── ...
```

Each issue file follows the template:
```markdown
---
Status: ready-for-agent | done | needs-triage | ...
---
## What to build
## Acceptance criteria
- [ ] criterion
## Blocked by
```

## Consequences

- Issues are visible in GitHub diffs — progress is transparent in the PR
- No GitHub Issues API calls required during development
- If the project grows to a team, migrate by running `/setup-matt-pocock-skills` and switching to GitHub Issues
