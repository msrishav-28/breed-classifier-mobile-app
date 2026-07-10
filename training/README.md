# Training Pipeline

The training pipeline prepares real-photo datasets, trains an EfficientNetV2-B0
transfer-learning classifier, exports a contract-compliant TensorFlow Lite
model, and evaluates that model with preprocessing that mirrors the Android
app.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- A GPU is strongly recommended. Native Windows TensorFlow 2.11+ does not use
  CUDA GPUs; use WSL2, Kaggle, Colab, or another GPU Linux environment for
  faster iteration.

## Dataset Preparation

Use `prepare_dataset.py` for directory-per-class source datasets:

```bash
python prepare_dataset.py \
  --source-dir /path/to/source1 \
  --source-dir /path/to/source2 \
  --output-dir ../datasets/processed/my_run \
  --source-name "dataset name" \
  --source-url "https://example.invalid/dataset" \
  --license "license name" \
  --hash-distance 0 \
  --include Gir \
  --include Hallikar \
  --include Murrah \
  --include Sahiwal \
  --include Tharparkar
```

The script:

- validates class folder names against `app/src/main/assets/data/breed_mapping.csv`,
- applies known public-dataset aliases,
- drops corrupt images,
- drops images whose shorter side is below 224 px,
- deduplicates with dHash,
- writes `trainval/` plus an untouched `test/` split,
- writes `summary.json`, `manifest.csv`, and `provenance.md`.

Only ship labels that resolve to the app catalog.

## Train and Export

```bash
python train.py \
  --data-dir ../datasets/processed/my_run/trainval \
  --output-dir ../build/model \
  --epochs 15 \
  --fine-tune-epochs 8 \
  --batch-size 16 \
  --label-smoothing 0.1
```

Outputs:

| File | Purpose |
|------|---------|
| `breed_classifier.tflite` | Float16-quantised model with float32 input `[1,224,224,3]`, raw 0-255 RGB, and softmax output `[1,N]` |
| `labels.txt` | Class labels in output order |
| `metrics.json` | Best validation accuracy / top-3 / loss, class weights, labels |
| `history.json` | Training curves |

The script verifies the exported model against the Android contract and prints
`Contract OK: float32 [1, 224, 224, 3] -> [1, N]` before finishing.

## Evaluate TFLite

Evaluate the exported model on the untouched test split:

```bash
python evaluate_tflite.py \
  --model ../build/model/breed_classifier.tflite \
  --labels ../build/model/labels.txt \
  --test-dir ../datasets/processed/my_run/test \
  --output-dir reports/my_run
```

The evaluator center-crops, bilinearly resizes to 224 x 224, and feeds raw
float32 RGB pixels exactly like `TfLiteBreedClassifier`. It writes overall
metrics, per-class precision/recall/F1, predictions, a confusion matrix CSV,
a confusion matrix PNG, calibration bins, and Python-side TFLite latency.

## Current Production Run

The bundled production model is documented in
`training/reports/combined_kaggle_v2_5class/` and supports:

- Gir
- Hallikar
- Murrah
- Sahiwal
- Tharparkar

Held-out TEST metrics:

- top-1 accuracy: 88.00%
- top-3 accuracy: 100.00%
- minimum class recall: 63.33%
- mean Python TFLite latency: 65.0 ms

The rejected 6-class candidate with Ongole is preserved in
`training/reports/rejected_6class_with_ongole/`; it missed the release floor
because Ongole recall was 53.33%.

## Bundle Into The App

The current production model is committed because it is under the documented
roughly 15 MB cap:

```bash
cp ../build/model/breed_classifier.tflite ../app/src/main/assets/models/breed_classifier.tflite
cp ../build/model/labels.txt ../app/src/main/assets/models/labels.txt
```

If a future model exceeds the cap, attach it to a release and wire CI/release
builds to fetch it instead of committing the binary.

## Test Fixture Tooling

Two auxiliary scripts support the app's low-cost instrumented tests:

- `make_fixture_dataset.py --output-dir <dir>` generates a procedural texture
  dataset named after real breeds so catalog integration can be tested.
- `export_test_model.py --data-dir <dir> --output-dir <dir>` trains a tiny
  CNN and exports a model implementing the exact production contract; the
  result is committed under `app/src/androidTest/assets/`.

The fixture model recognises procedural textures, not animals. It must never
be copied into `app/src/main/assets/`.
