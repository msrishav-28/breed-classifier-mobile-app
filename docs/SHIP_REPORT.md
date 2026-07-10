# Breed Classifier Ship Report

Date: 2026-07-10

## 1. Executive Verdict

Engineering, ML, and local QA are release-candidate ready for a 5-breed offline Android app. The app bundles a real TensorFlow Lite model, runs without network/storage permissions, classifies on device, stores history locally, and generates PDF reports.

The code/model release-candidate commit is on `main` and GitHub Actions is green. The single release-blocking operator step is signing and Play ownership: create the release keystore, set the CI signing secrets, choose a license, provide a hosted privacy-policy URL/contact, and run one physical-device camera check before tagging `v1.0.0`.

Unsigned local artifacts built successfully:

| Artifact | Path | Size |
|---|---|---:|
| Release APK | `app/build/outputs/apk/release/app-release-unsigned.apk` | 28,956,455 bytes |
| Release AAB | `app/build/outputs/bundle/release/app-release.aab` | 20,492,575 bytes |

## 2. Dataset

Production training used only openly licensed real-image datasets:

| Source | URL | License |
|---|---|---|
| Kaggle `lukex9442/indian-bovine-breeds`, version 5 | https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds | CC0: Public Domain |
| Kaggle `atharvadarpude/indian-buffalo-dataset`, version 1 | https://www.kaggle.com/datasets/atharvadarpude/indian-buffalo-dataset | CC0: Public Domain |
| Kaggle `algsoch/breed-cattle-buffalo`, version 1 | https://www.kaggle.com/datasets/algsoch/breed-cattle-buffalo | MIT |

Data was prepared by `training/prepare_dataset.py` into `datasets/processed/combined_kaggle_v2_5class`. Cleaning dropped corrupt files, images with shorter side below 224 px, and exact dHash duplicates (`--hash-distance 0`). The hold-out test split is 30 untouched images per shipped class.

| Breed | Raw | Cleaned | Train/Val | Test | Decision |
|---|---:|---:|---:|---:|---|
| Gir | 1,140 | 396 | 366 | 30 | Included |
| Hallikar | 488 | 191 | 161 | 30 | Included, limited data |
| Murrah | 682 | 208 | 178 | 30 | Included, limited data |
| Sahiwal | 894 | 374 | 344 | 30 | Included |
| Tharparkar | 447 | 180 | 150 | 30 | Included, limited data |

Ongole was tried in a 6-class run and rejected: held-out top-1 was 84.44%, top-3 was 99.44%, but Ongole recall was only 53.33%, below the 60% minimum class-recall gate. That rejected run is retained under `training/reports/rejected_6class_with_ongole/`.

Primary dataset evidence:

- `training/reports/combined_kaggle_v2_5class/dataset_summary.json`
- `training/reports/combined_kaggle_v2_5class/dataset_provenance.md`

## 3. Model

Architecture: EfficientNetV2-B0 transfer learning with Keras preprocessing baked into the graph (`include_preprocessing=True`), input `float32 [1,224,224,3]` raw RGB 0-255, output `float32 [1,5]` softmax. Training used a frozen-head phase followed by fine-tuning, tf.data augmentation outside the exported graph, label smoothing 0.1, and inverse-frequency class weights.

Exported model:

| File | Size | Distribution |
|---|---:|---|
| `app/src/main/assets/models/breed_classifier.tflite` | 11,846,652 bytes | Committed, under the ~15 MB policy cap |
| `app/src/main/assets/models/labels.txt` | 5 labels | Committed |

Export verifier printed the expected contract: `Contract OK: float32 [1, 224, 224, 3] -> [1, 5]`.

Held-out TEST evaluation, using app-equivalent preprocessing (center crop, 224 bilinear resize, raw float32 RGB):

| Metric | Value |
|---|---:|
| Test images | 150 |
| Top-1 accuracy | 88.00% |
| Top-3 accuracy | 100.00% |
| Minimum class recall | 63.33% |
| Python TFLite mean latency | 65.03 ms |
| Expected calibration error | 0.0983 |

Per-class metrics:

| Breed | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Gir | 86.36% | 63.33% | 73.08% | 30 |
| Hallikar | 90.62% | 96.67% | 93.55% | 30 |
| Murrah | 96.77% | 100.00% | 98.36% | 30 |
| Sahiwal | 72.97% | 90.00% | 80.60% | 30 |
| Tharparkar | 96.43% | 90.00% | 93.10% | 30 |

Confusion matrix reference: `training/reports/combined_kaggle_v2_5class/confusion_matrix.csv` and `.png`. The main remaining confusion is Gir vs Sahiwal: 10 of 30 Gir hold-out images were predicted as Sahiwal.

Measured emulator latency from `BundledModelSanityTest` on API 30 ATD:

| Sample | Top-3 | Latency |
|---|---|---:|
| Gir | Gir, Sahiwal, Tharparkar | 1,583 ms |
| Hallikar | Hallikar, Sahiwal, Tharparkar | 200 ms |
| Murrah | Murrah, Gir, Hallikar | 178 ms |
| Sahiwal | Sahiwal, Gir, Murrah | 457 ms |
| Tharparkar | Tharparkar, Hallikar, Gir | 187 ms |

Worst observed emulator latency was 1,583 ms, under the 3,000 ms product budget.

## 4. Verification Evidence

Final CI on `main` for the code/model release-candidate commit `3cfd7b960a19b44f2f9943aa24107032d0394d5c` is green:

- GitHub Actions run: https://github.com/msrishav-28/breed-classifier-mobile-app/actions/runs/29111578929
- Build job: `86424910176`
- E2E job: `86424910245`

Baseline CI on `main` before model integration was also green for both build and e2e:

- GitHub Actions run: https://github.com/msrishav-28/breed-classifier-mobile-app/actions/runs/28833844616
- Build job: `85513369784`
- E2E job: `85513369804`

Final local verification on the release-candidate tree:

| Check | Result |
|---|---|
| `./gradlew build` | Passed |
| `./gradlew :core:test :app:testDebugUnitTest :app:lintDebug` | Passed |
| `./gradlew :app:bundleRelease` | Passed |
| `./gradlew :app:connectedDebugAndroidTest` on `breed_api30` API 30 ATD | Passed: 8 tests, 1 expected skip |

Instrumented tests exercised:

- `BundledModelSanityTest`: production model loaded from main assets, 5 held-out real samples true breed in top-3, latency under 3,000 ms.
- `ReleaseManualFlowTest`: launched Results with a held-out Gir photo, verified result UI/catalog details, generated a PDF report via the app report generator, verified `%PDF`, opened History, and exported QA evidence.
- `ClassifierEndToEndTest`: fixture model contract and fixture predictions.
- `MainActivitySmokeTest`: home screen launches and primary actions are visible.
- `ModelUnavailableFlowTest`: skipped as expected because the production model is bundled.

QA artifacts:

- Results screenshot: `docs/qa/results_manual.png`
- History screenshot: `docs/qa/history_manual.png`
- Generated PDF: `docs/qa/manual_report.pdf`

Manual checklist status:

| Area | Status |
|---|---|
| Launch/home | Passed on emulator |
| Model availability | Passed; no missing-model card in bundled build |
| Result screen | Passed via held-out Gir sample |
| History | Passed via saved Gir record |
| PDF generation | Passed; generated PDF opens as `%PDF` artifact |
| System photo picker | Not completed on API 30 ATD because the fake system picker rendered blank; app path is covered by copy-to-app-storage code and result-flow instrumentation |
| Physical camera capture | Operator TODO on a real device |

## 5. Autonomous Decisions

- Shipped 5 breeds instead of 24 because only these classes met the measured quality gate. Fewer classes with honest performance is safer than shipping a broad but weak model.
- Dropped Ongole after a measured failed 6-class evaluation; recall was below gate.
- Committed the production TFLite model because it is 11.3 MB, below the documented ~15 MB policy cap. `.gitignore` was deliberately updated so this model can live in `app/src/main/assets/models/`.
- Kept existing confidence thresholds unchanged. Calibration was acceptable for a first release candidate, and the hardest issue is class data/visual overlap rather than threshold policy.
- Added `ReleaseManualFlowTest` instead of relying on the system share sheet or fake picker UI in CI. It verifies the app-owned result/history/PDF behavior without making CI depend on external Android UI.

## 6. Operator TODOs

Create the release keystore outside the repo:

```bash
keytool -genkeypair -v \
  -keystore breed-classifier-release.jks \
  -alias breed-classifier \
  -keyalg RSA \
  -keysize 4096 \
  -validity 10000
```

Set these GitHub Actions secrets before tagging:

- `RELEASE_KEYSTORE_BASE64`
- `RELEASE_KEYSTORE_PASSWORD`
- `RELEASE_KEY_ALIAS`

Then create `keystore.properties` locally only when signing on a trusted machine:

```properties
storeFile=../breed-classifier-release.jks
storePassword=...
keyAlias=breed-classifier
keyPassword=...
```

Remaining release-owner actions:

- Run one physical-device QA pass for camera capture.
- Host `store/privacy_policy.md` and insert the final URL/contact/effective date.
- Choose a repository license. MIT is shorter and permissive; Apache-2.0 is also permissive and adds explicit patent language. Do not add the file until the owner chooses.
- After signing secrets, privacy-policy hosting, license choice, and physical camera QA are complete, tag `v1.0.0` to publish the release workflow.

## 7. Risks And Next Iterations

- Gir recall is the weakest class at 63.33%; collect more clean side-profile Gir photos, especially examples that do not resemble Sahiwal.
- Hallikar, Murrah, and Tharparkar shipped below the 300-clean-image target, even though measured recall passed. Prioritize additional data before expanding class count.
- Source labels are public dataset labels and may contain noise. A veterinary or breed-expert review set would reduce risk.
- Add more buffalo breeds only after enough real images are available; do not rely on synthetic/augmented images for production claims.
- Consider INT8 quantization only after measuring accuracy loss; float16 is currently under the size and latency budgets.
- Future calibration work can tune confidence thresholds if larger test sets show overconfidence on field photos.
