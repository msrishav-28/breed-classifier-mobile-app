# Testing

## Strategy

| Layer | Where | What | How it runs |
|-------|-------|------|-------------|
| Domain logic | `core/src/test` | Catalog parsing, name normalisation, prediction post-processing, confidence policy, quality metrics/policy, report content | JUnit 5 + Kotest, including property-based tests; pure JVM, no SDK |
| App logic | `app/src/test` | Entity-domain mapping, alternatives codec, subsampling math, validation of the bundled CSV asset | JUnit 5 on the JVM |
| On-device E2E | `app/src/androidTest` | Home screen smoke; real TFLite inference through the committed fixture model; bundled production model sanity test against held-out real photos; release-flow PDF/history proof; degraded "model unavailable" flow when no model is bundled | Espresso/JUnit4 on an emulator; CI runs the full suite on an Android 11 emulator every push |

Property-based tests carry real weight here: they already caught a
normalisation idempotency bug during development (`BreedNames.normalize("_")`).
Prefer a property over an example when a function has an algebraic guarantee
(sums to one, ordering, round-trips, idempotency).

## Commands

```bash
./gradlew :core:test
./gradlew :app:testDebugUnitTest
./gradlew :app:lintDebug
./gradlew :app:connectedDebugAndroidTest   # device or emulator required
```

CI runs all of the above on every push and pull request: the `build` job
covers unit tests, lint and debug/release assembly; the `e2e` job boots a
hardware-accelerated emulator and runs the instrumented suite, exercising real
TFLite inference end to end.

## Model Tests

`app/src/androidTest/assets/models/` contains a tiny TFLite fixture model that
implements the exact production model contract but was trained on procedural
textures (`training/export_test_model.py`). It gives deterministic predictions
for low-cost contract tests. It recognises textures, not animals; never move it
into `app/src/main/assets/`.

`BundledModelSanityTest` runs only when a main-assets production model is
bundled. It loads the app model through `ClassifierProvider`, classifies five
held-out real photos from `app/src/androidTest/assets/real_samples/`, asserts
the true breed appears in top-3, and checks inference remains below 3 seconds.

`ReleaseManualFlowTest` exercises the release-candidate result path with a
held-out Gir photo, verifies catalog-backed UI content, generates a PDF report,
checks the PDF header, opens history, and exports QA artifacts under
`docs/qa/` when run locally.

`ModelUnavailableFlowTest` covers the degraded path and automatically skips
when a production model is bundled.

## Manual QA Checklist

Run before any store release; a physical device is required for the true camera
capture checks.

1. **Fresh install** - home opens without crash; camera/gallery entry points
   are visible; no network prompt or storage permission appears.
2. **Camera flow** - capture in portrait; photo appears upright in results;
   result arrives in under 3 seconds; entry appears in history.
3. **Gallery flow** - pick an existing photo; classification matches the
   camera flow; a picked photo remains available after relaunch because it is
   copied into app storage.
4. **Quality warnings** - photograph a dark scene and a deliberately shaken
   shot; expect dark/blur warnings alongside a result.
5. **Low confidence** - photograph a non-animal; expect a low-confidence
   warning, not a confident wrong answer.
6. **History** - reopen an entry with no re-classification delay, delete one
   entry, clear all; empty state appears.
7. **Report** - share/export a PDF; verify photo, breed, confidence, breed
   details, and "performed on device" language render correctly in a PDF
   viewer.
8. **Dark mode** - toggle system dark theme; every screen remains legible.
9. **Permission denial** - deny camera permission; the app explains and
   returns home rather than crashing; gallery flow still works.
10. **Process death** - background the app on results, kill it, reopen from
    history; the record renders from the database.

Optional degraded-path check for builds with the model intentionally removed:
results should show the "model unavailable" message instead of crashing or
spinning.
