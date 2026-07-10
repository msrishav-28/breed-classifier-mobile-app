# Production Model Report: combined_kaggle_v2_5class

This is the production candidate bundled in the Android app.

## Scope

Supported labels, in model output order:

1. Gir
2. Hallikar
3. Murrah
4. Sahiwal
5. Tharparkar

Ongole was evaluated in an earlier 6-class candidate but dropped because its
held-out recall was 53.33%, below the 60% release floor. That rejected run is
kept in `training/reports/rejected_6class_with_ongole`.

## Dataset

Sources:

- Kaggle `lukex9442/indian-bovine-breeds`, CC0: Public Domain
- Kaggle `atharvadarpude/indian-buffalo-dataset`, CC0: Public Domain
- Kaggle `algsoch/breed-cattle-buffalo`, MIT

Cleaning and split details are in `dataset_summary.json` and
`dataset_provenance.md`. The final held-out test split contains 30 images per
supported class.

## Metrics

Held-out TEST, app-equivalent TFLite preprocessing:

| Metric | Value |
|---|---:|
| Top-1 accuracy | 88.00% |
| Top-3 accuracy | 100.00% |
| Minimum class recall | 63.33% |
| Mean Python TFLite latency | 65.0 ms |
| Calibration ECE | 0.0983 |

Per-class precision/recall is in `per_class_metrics.csv`; confusion matrix is
available as `confusion_matrix.csv` and `confusion_matrix.png`.

## Calibration Note

The model is not wildly overconfident on this 150-image hold-out set. Most
predictions above 0.9 confidence were correct, while lower-confidence bins are
mixed. The existing app thresholds remain appropriate:

- high confidence: 0.75 and above
- medium confidence: 0.50 to 0.75
- low confidence: below 0.50

The hardest remaining confusion is Gir versus Sahiwal. Gir recall is 63.33%,
just above the release floor; future data collection should prioritize more
clean Gir photos across age, lighting, angle, and body-frame variation.
