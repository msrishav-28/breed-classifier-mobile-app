# Training pipeline

One script trains the breed classifier and exports exactly what the Android
app consumes. The previous multi-framework pipelines (ViT, YOLOv8-CBAM,
metric learning, ensembles) were removed: they never produced a deployable
artifact, and a single well-trained EfficientNetV2 transfer-learning model is
simpler to train, evaluate, and ship. Extend from a working baseline instead.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- A GPU is strongly recommended (Kaggle/Colab free tiers work).

## Dataset

Directory-per-class layout; directory names become model labels:

```
dataset/
├── Gir/
├── Red_Sindhi/
├── Murrah/
└── ...
```

Use the breed names from `app/src/main/assets/data/breed_mapping.csv` so the
app can show catalog details; underscores and spaces are interchangeable.
Aim for at least a few hundred images per breed. An 80/20 train/validation
split is applied automatically.

## Train and export

```bash
python train.py --data-dir /path/to/dataset --output-dir build
```

Outputs in `build/`:

| File | Purpose |
|------|---------|
| `breed_classifier.tflite` | float16-quantised model (app contract: float32 input `[1,224,224,3]`, raw 0–255 RGB; softmax output `[1,N]`) |
| `labels.txt` | class labels in output order |
| `metrics.json` | validation accuracy / top-3 / loss |

The script verifies the exported model against the contract and fails loudly
if it drifts. See `docs/MODEL.md` at the repository root for the full
contract description.

## Bundle into the app

```bash
cp build/breed_classifier.tflite ../app/src/main/assets/models/
cp build/labels.txt              ../app/src/main/assets/models/
```

Model binaries are intentionally gitignored; distribute them through releases
or CI artifacts, not the repository.
