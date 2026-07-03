# PROJECT AUDIT

**Date:** 2026-07-03
**Repository:** msrishav-28/breed-classifier-mobile-app
**Scope:** Full repository — build system, application code, ML pipeline, tests, docs, release readiness.

## Executive summary

**Current maturity score: 18 / 100**

This repository is a spec-scaffolded hackathon prototype (Smart India Hackathon 2025), not a
shippable application. The Kotlin codebase (~8,900 lines of main source, ~5,700 lines of tests)
**does not compile**, the Gradle build **cannot run** (the wrapper script and JAR are missing),
the app's entry screen is a dead end that navigates nowhere, no ML model is present, and roughly
70% of the source implements speculative subsystems (three-model ensembles, on-device dataset
acquisition and augmentation, metric-learning fallbacks, sync queues) that nothing invokes and
that reference infrastructure which does not exist.

The salvageable assets are: a reasonable CameraX capture activity, a breed-catalog CSV with 24
Indian cattle/buffalo breeds, a sensible Room schema core, image-quality heuristics, a results
screen layout, and the product concept itself.

## Findings by severity

### Critical (release-blocking, broken today)

| # | Finding | Detail | Effort | Impact |
|---|---------|--------|--------|--------|
| C1 | Build cannot run | `gradlew` (Unix) and `gradle-wrapper.jar` are missing; only `gradlew.bat` exists. `./gradlew` fails on every platform CI would use. | S | Blocks everything |
| C2 | Code does not compile | `ImageCaptureService.kt:49` declares `override preprocessImage(...)` — missing `fun`. `ModelConversionService.kt` is a 0-byte file. | S | Blocks everything |
| C3 | Phantom dependency injection | 9 classes use `javax.inject` `@Inject`/`@Singleton`, but no DI framework (Hilt/Dagger) nor `javax.inject` is declared in Gradle. Nothing constructs these classes. | M | Blocks compile & runtime |
| C4 | App does nothing | `MainActivity` renders a title and a button that is never wired. Camera, results, database, and all services are unreachable from the launcher. | M | No product |
| C5 | AGPL license contamination | `com.itextpdf:itext7-core` is AGPL-licensed. Shipping a closed-source store build with it is a license violation. | S | Legal |
| C6 | Tests never run | Property tests use Kotest/JUnit 5 but the module never configures `useJUnitPlatform()`; most tests also reference the non-compiling services. | M | False confidence |
| C7 | No CI | No `.github/workflows`. Nothing gates regressions. | S | Quality |

### High

| # | Finding | Detail | Effort | Impact |
|---|---------|--------|--------|--------|
| H1 | Duplicate inference engines | `OfflineMLInferenceEngine` (507 lines) and `AdvancedMLInferenceEngine` (755 lines) implement the same responsibility with diverging model filenames and thresholds. | M | Maintainability |
| H2 | Breed-name mismatch bug | Model labels use underscores (`Red_Sindhi`); the catalog CSV uses spaces (`Red Sindhi`). `TypeClassificationService` looks up by exact lowercase match, so breed info silently fails for every multi-word breed. | S | Wrong results |
| H3 | Training code inside the app | `training/*.kt` (1,200+ lines) performs dataset downloading/augmentation *on the phone*; `app/src/main/python/` (5,300 lines, 10 scripts, 6 configs) ships training pipelines inside the app source tree. | M | Bloat, confusion |
| H4 | No model, no degradation story | All engines assume `.tflite` assets exist (they are gitignored); there is no user-facing state for "model missing". | M | Crashes/blank UX |
| H5 | Unused heavyweight dependencies | Jetpack Compose (enabled, zero composables), Glide (zero usages), Navigation component (zero usages), camera-video/extensions, GPU delegate plugin, Gson (only used by an unneeded converter). | S | APK size, build time |
| H6 | EXIF rotation ignored | Captured/gallery images are decoded with `BitmapFactory.decodeFile` without EXIF handling — portrait photos will be classified and displayed sideways. | S | Accuracy, UX |
| H7 | Over-broad permissions | `INTERNET`, `ACCESS_NETWORK_STATE`, `READ_MEDIA_IMAGES`, legacy storage permissions — none needed for an offline classifier using the photo picker. | S | Privacy, store review |

### Medium

| # | Finding | Detail | Effort | Impact |
|---|---------|--------|--------|--------|
| M1 | Stale toolchain | AGP 8.2.0 / Kotlin 1.9.10 / compileSdk 34 / Java 8 targets; kapt where KSP is standard. | M | Security, velocity |
| M2 | Release build unhardened | `isMinifyEnabled = false`, no resource shrinking, no ProGuard rules for TFLite. | S | Size, IP |
| M3 | Fake domain data | `typeConfidence` is presented as an ML output but animal type is a deterministic CSV lookup from breed; `getMetricLearningEmbedding` fabricates "embeddings" from milk-yield numbers. | S | Integrity |
| M4 | Launcher icon misconfigured | Adaptive-icon XMLs live only in `mipmap-hdpi/`; no `anydpi-v26` qualifier, no legacy rasters for API 24–25. | S | Install UX |
| M5 | Speculative persistence | `CacheEntity`/`CacheDao`, `BreedMappingEntity`/`BreedMappingDao`, `synced` flags and sync queries model a backend that doesn't exist. | S | Complexity |
| M6 | Dead planning artifacts | `.kiro/specs/` (1,000 lines), `plan.md` (899), `improvement.md` (572), per-package `README.md`s describing unbuilt features. | S | Confusion |
| M7 | Theme only styles light mode | `Theme.Material3.DayNight` parent but every color is hardcoded to the light palette; no `values-night`. | S | Dark mode broken |
| M8 | God-object services | `ErrorHandlingService`, `PerformanceMetricsService`, `OfflineProcessingService`, `NetworkStateMonitor` (~1,400 lines) — cross-cutting frameworks with zero call sites. | M | Maintainability |

### Low

| # | Finding | Detail |
|---|---------|--------|
| L1 | `README.md` claims "95%+ accuracy", sub-3s inference, and features that do not exist. |
| L2 | Non-exhaustive `when` over sealed `CameraState` (missing `Initializing`). |
| L3 | Hardcoded fallback label list duplicated across engines and drifting from the CSV. |
| L4 | `rootProject.name` contains spaces ("Cattle Buffalo Recognition"). |
| L5 | No LICENSE file despite "public release" intent. |
| L6 | `screenOrientation="portrait"` locked without rationale; no landscape/tablet story. |

## Dimension review

- **Architecture:** Layered names (data/ml/services/ui) exist, but there is no composition root, no
  dependency direction discipline, and two parallel implementations of the ML path. Score: 2/10.
- **ML pipeline:** Aspirational three-model ensemble (ViT + YOLOv8-CBAM + EfficientNetV2) with
  test-time augmentation — none of it runnable; Python training scripts target three different
  frameworks with inconsistent export contracts and live inside the app source set. No committed
  model. Score: 2/10.
- **UX:** Launcher screen is a dead end; no history surface despite a history database; no error/
  empty/loading states wired; dark mode broken. Score: 2/10.
- **Security/Privacy:** No secrets committed (good). Over-broad permissions; AGPL dependency;
  backup rules exclude a database name (`classifications.db`) that doesn't match the actual
  database (`livestock_database`). Score: 4/10.
- **Testing:** 5,700 lines of tests that cannot execute; no CI. Effective coverage: 0%. Score: 1/10.
- **Docs:** README describes a fictional finished product; specs describe plans, not the code.
  Score: 2/10.
- **Store readiness:** Not signable, not buildable, no privacy policy, no store assets. Score: 0/10.

## Strategy

The honest engineering call is a **rebuild around the working core**, not incremental patching:
keep the CameraX flow, the breed catalog, the quality heuristics, and the results UX; delete every
speculative subsystem; make the ML path a single, well-defined TFLite contract that the training
pipeline provably produces; and make every remaining line verifiable by build + tests + CI.

See the roadmap in this document's companion (`docs/ARCHITECTURE.md` describes the target design)
and the phase breakdown below.

## Roadmap

### Phase 1 — Build resurrection (Critical)
*Objectives:* repository builds with one command everywhere.
*Work:* commit a real Gradle wrapper; upgrade to AGP 8.7 / Kotlin 2.0 / SDK 35 / JDK 17; introduce
a version catalog; split a pure-JVM `:core` module for domain logic; remove unused build features
(Compose, kapt) and dependencies (Glide, Gson, Navigation, iText, GPU delegate); enable R8.
*Acceptance:* `./gradlew build` succeeds; no analyzer errors.

### Phase 2 — Architecture rebuild (Critical/High)
*Objectives:* a single coherent app: Home → (Camera | Photo picker) → Results → History.
*Work:* delete non-compiling/speculative code (both engines, on-device training, metric learning,
ensemble coordination, offline sync framework, cache entities); implement `TfLiteBreedClassifier`
with an explicit model contract and a first-class "model unavailable" state; manual DI via an
application-scoped container; slim Room schema; EXIF-aware image loading; breed-name
normalization shared between labels and catalog.
*Acceptance:* every class in the repo is reachable from the app; zero `javax.inject` references.

### Phase 3 — UX & platform polish (High/Medium)
*Objectives:* production feel.
*Work:* Material 3 light/dark palettes; adaptive + legacy launcher icons; history screen with
empty state; loading/error states on results; photo-picker (no storage permission); drop INTERNET
permission; accessibility labels; localization-ready strings.
*Acceptance:* lint clean; manual flows documented in TESTING.md.

### Phase 4 — ML contract & training pipeline (High)
*Objectives:* one training pipeline that emits exactly what the app consumes.
*Work:* move Python out of `app/src/main/`; single transfer-learning pipeline (EfficientNetV2-B0)
exporting float16-quantized TFLite + labels; document dataset layout and the preprocessing
contract; delete the four divergent pipelines.
*Acceptance:* `training/train.py --export` produces `breed_classifier.tflite` + `labels.txt`
consumed as-is by the app.

### Phase 5 — Tests (Critical)
*Objectives:* tests that actually run.
*Work:* JUnit 5 + Kotest property tests in `:core` (catalog parsing, normalization, post-
processing, confidence policy, quality policy, report content); pragmatic unit tests in `:app`;
compile-gated instrumentation smoke test.
*Acceptance:* `./gradlew test` green locally (core) and in CI (all).

### Phase 6 — CI/CD & docs (High)
*Objectives:* regressions gated; contributors onboarded.
*Work:* GitHub Actions (build, lint, unit tests, androidTest compile, APK artifact); rewrite
README; add ARCHITECTURE, SETUP, TESTING, MODEL, CONTRIBUTING, RELEASE, PRIVACY docs; delete
`.kiro/`, `plan.md`, `improvement.md`.
*Acceptance:* Actions run green on the branch; docs match reality.

### Phase 7 — Release readiness (tracked, partially external)
Remaining for the owner: train and commit a real model, add signing config + store listing,
host a privacy policy, on-device performance validation. Documented in `docs/RELEASE.md`.

## Risk assessment

- **No Android device/emulator in this environment** — runtime behavior is verified by unit
  tests, lint, and CI builds; a manual QA checklist is provided for device validation.
- **Model absence** — the app is fully functional as software, but classification requires the
  owner to train and bundle a model (documented; the UI degrades gracefully).
- **Dataset availability** — training accuracy targets depend on a dataset the repository does
  not (and should not) contain.
