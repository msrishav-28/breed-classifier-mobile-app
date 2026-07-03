# Contributing

## Workflow

1. Branch from `main`.
2. Make the change with tests. Bug fixes come with a regression test.
3. `./gradlew build` locally (or at minimum `:core:test`,
   `:app:testDebugUnitTest`, `:app:lintDebug`).
4. Open a pull request; CI must be green.

## Ground rules

- **`:core` stays Android-free.** Domain logic, parsing, policies and report
  content belong there with tests; only platform glue goes in `:app`.
- **One source of truth per policy.** Confidence thresholds, quality
  thresholds and name normalisation live in `:core` — never re-declare them
  in UI code.
- **Every user-visible string** goes in `res/values/strings.xml`.
- **Dependency changes** go through `gradle/libs.versions.toml`, with a
  sentence in the PR about why the dependency is worth its weight. Licenses
  must be permissive (no GPL/AGPL — an AGPL PDF library has already been
  removed from this codebase once).
- **The model contract is frozen** (docs/MODEL.md). Changing tensors, label
  matching or asset paths requires updating the contract doc, the loader
  validation, and `training/train.py` in the same PR.
- **No speculative code.** Features land wired to UI and tested, or not at
  all. Git history preserves anything deleted.

## Style

Standard Kotlin style (`kotlin.code.style=official`), meaningful names, KDoc
on public API where the signature alone doesn't explain the behavior.
Comments explain *constraints and whys*, not what the next line does.

## Tests

Prefer property-based tests (Kotest) for algebraic guarantees — round-trips,
ordering, idempotency, ranges. See docs/TESTING.md for the layer map and the
manual QA checklist for device-facing changes.
