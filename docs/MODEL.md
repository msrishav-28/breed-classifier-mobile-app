# Model contract

The app consumes one TensorFlow Lite image classifier. Anything that
satisfies this contract can be dropped in without code changes; the app
validates it at load time and refuses (gracefully) to run a model that
doesn't match.

## Files

| Asset path | Description |
|------------|-------------|
| `app/src/main/assets/models/breed_classifier.tflite` | The model |
| `app/src/main/assets/models/labels.txt` | One label per line, in model output order |

Both are **gitignored** — distribute them via releases or CI artifacts, not
git history.

## Tensors

| | Shape | Type | Semantics |
|--|-------|------|-----------|
| Input | `[1, 224, 224, 3]` | float32 | RGB pixels, **raw 0–255 values**. All normalisation must be baked into the model graph (`include_preprocessing=True` in the Keras export). |
| Output | `[1, N]` | float32 | Class probabilities (softmax). `N` must equal the number of lines in `labels.txt`. |

The app center-crops the photo to a square and bilinearly scales it to
224×224 before packing the buffer (`TfLiteBreedClassifier.preprocess`).
Post-processing (`PredictionPostProcessor`) is defensive: scores are clamped
non-negative and renormalised, so a slightly miscalibrated model cannot
produce out-of-range confidences.

## Labels

- One label per line, e.g. `Red_Sindhi`.
- Labels are matched to the breed catalog (`app/src/main/assets/data/
  breed_mapping.csv`) case-insensitively with `_`, `-` and spaces treated as
  equivalent, so `Red_Sindhi` finds "Red Sindhi".
- A label without a catalog entry still classifies; the app simply cannot
  show breed details for it. Keep the two files in sync.

## Load-time validation

`TfLiteBreedClassifier.create` fails with a precise message when:

- either asset is missing or the label file is empty,
- the input tensor is not float32 `[1, 224, 224, 3]`,
- the output class count differs from the label count.

Failures surface as the "model unavailable" state, never a crash.

## Confidence policy

Defined once in `core` (`ConfidencePolicy`) and used by UI and reports:

| Confidence | Level | Behaviour |
|-----------|-------|-----------|
| ≥ 0.75 | High | Shown without reservations |
| 0.50 – 0.75 | Medium | Shown with a retake suggestion |
| < 0.50 | Low | Shown with a prominent warning |

## Producing a model

`training/train.py` produces contract-compliant artifacts (EfficientNetV2-B0
transfer learning, float16 quantisation ≈ 12 MB) and verifies them against
this contract before finishing. See `training/README.md`.

## Performance expectations

Float16 EfficientNetV2-B0 at 224×224 runs in roughly 40–150 ms on 2020+
mid-range devices with XNNPACK (4 threads), comfortably within the 3-second
product budget. Verify on target hardware as part of the release checklist.
