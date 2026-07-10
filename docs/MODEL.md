# Model Contract

The app consumes one TensorFlow Lite image classifier. The app validates the
contract at load time and refuses, gracefully, to run a model that does not
match.

## Files

| Asset path | Description |
|------------|-------------|
| `app/src/main/assets/models/breed_classifier.tflite` | The bundled production model |
| `app/src/main/assets/models/labels.txt` | One label per line, in model output order |

The production model is committed because it is 11.3 MB, below the documented
roughly 15 MB repository cap. Future models above that cap should be attached
to releases and fetched by release automation instead.

## Tensors

| | Shape | Type | Semantics |
|--|-------|------|-----------|
| Input | `[1, 224, 224, 3]` | float32 | RGB pixels, raw 0-255 values. All normalisation is baked into the model graph (`include_preprocessing=True` in the Keras export). |
| Output | `[1, N]` | float32 | Class probabilities (softmax). `N` must equal the number of lines in `labels.txt`. |

The app center-crops the photo to a square and bilinearly scales it to 224 x
224 before packing the buffer (`TfLiteBreedClassifier.preprocess`).
Post-processing (`PredictionPostProcessor`) is defensive: scores are clamped
non-negative and renormalised, so a slightly miscalibrated model cannot produce
out-of-range confidences.

## Bundled Production Model

Architecture and training:

- EfficientNetV2-B0 transfer learning
- ImageNet initialization with `include_preprocessing=True`
- tf.data augmentation outside the exported graph
- 15 frozen-head epochs, then 8 fine-tuning epochs
- class weighting for imbalance
- label smoothing 0.1
- float16 TFLite weight quantisation with float32 input/output contract

Supported labels, in output order:

1. Gir
2. Hallikar
3. Murrah
4. Sahiwal
5. Tharparkar

Held-out TEST metrics, using app-equivalent TFLite preprocessing over 30
untouched images per class:

| Metric | Value |
|---|---:|
| Top-1 accuracy | 88.00% |
| Top-3 accuracy | 100.00% |
| Minimum class recall | 63.33% |
| Mean Python TFLite latency | 65.0 ms |
| Calibration ECE | 0.0983 |

Per-class metrics:

| Breed | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Gir | 86.36% | 63.33% | 73.08% | 30 |
| Hallikar | 90.62% | 96.67% | 93.55% | 30 |
| Murrah | 96.77% | 100.00% | 98.36% | 30 |
| Sahiwal | 72.97% | 90.00% | 80.60% | 30 |
| Tharparkar | 96.43% | 90.00% | 93.10% | 30 |

Full reports live under
`training/reports/combined_kaggle_v2_5class/`. The rejected 6-class candidate
that included Ongole is preserved under
`training/reports/rejected_6class_with_ongole/`; it was not shipped because
Ongole recall was 53.33%, below the release floor.

## Labels

- One label per line, e.g. `Red_Sindhi`.
- Labels are matched to the breed catalog
  (`app/src/main/assets/data/breed_mapping.csv`) case-insensitively with `_`,
  `-`, and spaces treated as equivalent, so `Red_Sindhi` finds "Red Sindhi".
- Every shipped production label must have a catalog row.

## Load-Time Validation

`TfLiteBreedClassifier.create` fails with a precise message when:

- either asset is missing or the label file is empty,
- the input tensor is not float32 `[1, 224, 224, 3]`,
- the output class count differs from the label count.

Failures surface as the "model unavailable" state, never a crash.

## Confidence Policy

Defined once in `core` (`ConfidencePolicy`) and used by UI and reports:

| Confidence | Level | Behaviour |
|-----------|-------|-----------|
| >= 0.75 | High | Shown without reservations |
| 0.50 to 0.75 | Medium | Shown with a retake suggestion |
| < 0.50 | Low | Shown with a prominent warning |

The final model calibration did not justify changing these thresholds.

## Producing a Model

`training/train.py` produces contract-compliant artifacts and verifies them
before finishing. `training/prepare_dataset.py` prepares cleaned train/test
splits, and `training/evaluate_tflite.py` evaluates the exported model using
the same center-crop, bilinear-resize, raw-float preprocessing used by the
Android app. See `training/README.md`.

## Performance Expectations

Float16 EfficientNetV2-B0 at 224 x 224 should run well under the 3-second
product budget with XNNPACK and 4 threads. The Python TFLite harness measured
65.0 ms mean inference on this development machine. On the API 30 ATD emulator,
`BundledModelSanityTest` measured a 1,583 ms cold first inference and 178-457 ms
warm inferences across the remaining held-out samples.
