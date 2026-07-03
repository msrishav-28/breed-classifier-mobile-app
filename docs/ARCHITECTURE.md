# Architecture

## Modules

```
┌─────────────────────────────── app (Android) ────────────────────────────────┐
│  ui/        main · camera · results · history  (Activities + ViewModels)    │
│  ml/        BreedClassifier (interface) · TfLiteBreedClassifier · Provider  │
│  data/      Room (entity/dao/db) · HistoryRepository · BreedCatalogProvider │
│  image/     BitmapLoader (EXIF) · ImageQualityAnalyzer · ImageFiles         │
│  report/    PdfReportGenerator (platform PdfDocument) · ReportSharer        │
│  di/        AppContainer (manual composition root, owned by LivestockApp)   │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │ depends on
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                            core (pure JVM Kotlin)                            │
│  model/     AnimalType · BreedInfo · Prediction · ClassificationRecord      │
│  catalog/   BreedCatalog (CSV parsing) · BreedNames (normalisation)         │
│  classify/  PredictionPostProcessor (softmax/top-k) · ConfidencePolicy      │
│  quality/   LuminanceStatistics · ImageQualityPolicy                        │
│  report/    ReportContent · ReportContentBuilder                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

`:core` has **no Android dependency**. Everything that can be expressed as
pure logic lives there, which is what makes the domain exhaustively testable
(JUnit 5 + Kotest property tests) on any JVM. `:app` contains only what needs
the Android platform: UI, camera, bitmap handling, TFLite runtime, Room, PDF
rendering.

## The classification flow

```
MainActivity ── camera ──▶ CameraActivity ──photo path──┐
     │                                                  ▼
     └── photo picker ──copy to app storage──▶ ResultsActivity
                                                    │ ResultsViewModel
                                                    ▼
      BitmapLoader.decode (subsampled + EXIF-rotated)
                                                    │
      ImageQualityAnalyzer ──▶ quality warnings (advisory, non-blocking)
                                                    │
      ClassifierProvider.get() ── model missing ──▶ "model unavailable" state
                                                    │ Ready
      TfLiteBreedClassifier.classify ──▶ top-3 predictions
                                                    │
      BreedCatalog.find(label) ──▶ animal type + breed details
                                                    │
      HistoryRepository.save ──▶ Room               │
                                                    ▼
                        Result UI (photo, match, confidence, warnings, details)
```

The history flow re-enters the same screen with a record id instead of an
image path; nothing is re-classified.

## Key decisions

**Single model, explicit contract.** The app consumes exactly one TFLite
model with a documented tensor contract (docs/MODEL.md), validated at load
time. The previous three-model ensemble design had no trained models, tripled
memory and latency budgets, and had no measurable accuracy story. An ensemble
can return later behind the same `BreedClassifier` interface if a trained one
proves its worth.

**Model absence is a first-class state.** Model binaries are gitignored, so
`ClassifierProvider` exposes `Ready`/`Unavailable`, the home screen shows a
warning, and the results screen explains rather than crashes.

**Manual DI.** One `AppContainer` on the `Application` class. At this app's
size a DI framework (Hilt) buys annotation processing time and indirection
without solving a problem this codebase has.

**Breed names are normalised.** Model labels (`Red_Sindhi`) and catalog names
(`Red Sindhi`, `Nili-Ravi`) are matched through `BreedNames.normalize`, fixing
the silent lookup failures the original code had for every multi-word breed.

**Animal type is a lookup, not a prediction.** Type (dairy/draught/dual) is
deterministic from breed via the catalog. The old code fabricated a
"type confidence" number for it; that fiction is gone.

**Platform PDF instead of iText.** `android.graphics.pdf.PdfDocument` removes
an AGPL dependency (a store-release blocker) and ~2 MB of APK for a report
this simple. Report *content* is built in `:core` and unit-tested; only the
drawing lives in `:app`.

**Photos in internal storage.** `filesDir/images` needs no permissions, is
excluded from backups (see `res/xml/backup_rules.xml`), and is deleted with
the app. The photo picker (`PickVisualMedia`) means no storage permission on
any Android version; the app's entire permission surface is `CAMERA`.

**Views, not Compose.** The screens are simple and the original codebase
declared Compose without using it. ViewBinding + Material 3 keeps the
toolchain small; migrating to Compose later is a screen-by-screen decision.

## Threading

- ViewModels own coroutines (`viewModelScope`); IO on `Dispatchers.IO`,
  pixel/tensor work on `Dispatchers.Default`.
- The TFLite interpreter is not thread-safe; `TfLiteBreedClassifier` guards
  it with a `Mutex` and is created once per process by `ClassifierProvider`.

## Error handling policy

- Domain parsing (`BreedCatalog.parse`) returns a result type with warnings;
  a bad CSV row degrades that row, not the app.
- Classifier failures throw `ClassificationException`, mapped to user-facing
  states in the ViewModel — never raw exceptions in UI code.
- Quality issues and low confidence are advisory: the user always sees the
  best available answer plus an honest warning.
