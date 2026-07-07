# Testing

## Strategy

| Layer | Where | What | How it runs |
|-------|-------|------|-------------|
| Domain logic | `core/src/test` | Catalog parsing, name normalisation, prediction post-processing, confidence policy, quality metrics/policy, report content | JUnit 5 + Kotest, including property-based tests; pure JVM, no SDK |
| App logic | `app/src/test` | Entity↔domain mapping, alternatives codec, subsampling math, validation of the real bundled CSV asset | JUnit 5 on the JVM |
| On-device E2E | `app/src/androidTest` | Home screen smoke; real TFLite inference through `TfLiteBreedClassifier` against the committed fixture model and images, incl. catalog resolution; the degraded "model unavailable" results screen | Espresso/JUnit4 on an emulator; CI runs the full suite on an Android 11 emulator every push |

Property-based tests carry real weight here: they already caught a
normalisation idempotency bug during development (`BreedNames.normalize("_")`).
Prefer a property over an example when a function has an algebraic guarantee
(sums to one, ordering, round-trips, idempotency).

## Commands

```bash
./gradlew :core:test
./gradlew :app:testDebugUnitTest
./gradlew :app:lintDebug
./gradlew :app:connectedDebugAndroidTest   # device required
```

CI runs all of the above on every push and pull request: the `build` job
covers unit tests, lint and debug/release assembly; the `e2e` job boots a
hardware-accelerated emulator and runs the instrumented suite, exercising
real TFLite inference end to end.

### The fixture model

`app/src/androidTest/assets/models/` contains a tiny (~100 KB) TFLite model
that implements the exact production model contract but was trained on
procedurally generated textures (`training/export_test_model.py`). It gives
the E2E tests deterministic predictions without shipping any model in the
app itself. It recognises textures, not animals — never move it into
`app/src/main/assets/`.

## Manual QA checklist (device)

Run before any release; requires a bundled model.

1. **Fresh install, no model** — home shows the "model not installed" card;
   camera/gallery still open; results screen explains the missing model.
2. **Camera flow** — capture in portrait; photo appears upright in results;
   result arrives in < 3 s; entry appears in history.
3. **Gallery flow** — pick an existing photo; classification matches the
   camera flow; a picked photo remains available after relaunch (it is
   copied into app storage).
4. **Quality warnings** — photograph a dark scene and a deliberately shaken
   shot; expect dark/blur warnings alongside a result.
5. **Low confidence** — photograph a non-animal; expect a low-confidence
   warning, not a confident wrong answer.
6. **History** — reopen an entry (no re-classification delay), delete one
   entry, clear all; empty state appears.
7. **Report** — share a PDF; verify photo, breed, confidence, breed details
   and "performed on device" line render correctly in a PDF viewer.
8. **Dark mode** — toggle system dark theme; every screen remains legible.
9. **Permission denial** — deny camera permission; the app explains and
   returns home rather than crashing; gallery flow still works.
10. **Process death** — background the app on results, kill it, reopen from
    history; the record renders from the database.
