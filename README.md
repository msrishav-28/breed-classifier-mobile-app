# Breed Classifier

An offline Android app that identifies supported Indian cattle and buffalo
breeds from a photo using an on-device TensorFlow Lite model, and classifies
each breed's primary use (dairy, draught, dual-purpose). Built for field use
in support of livestock programmes such as the Rashtriya Gokul Mission: no
connectivity required, no data leaves the device.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

## What the app does

1. **Capture or pick a photo** - CameraX capture screen, or the system photo
   picker (no storage permission needed).
2. **Quality check** - brightness, contrast, sharpness and resolution
   heuristics warn about photos the model will struggle with.
3. **Classify on device** - a single TFLite model returns the top-3 breeds
   with confidences; low-confidence results are clearly flagged.
4. **Breed details** - species, origin, primary use, typical milk yield and
   traits from a bundled catalog of 24 Indian breeds.
5. **History & reports** - every classification is stored locally (Room) and
   can be reopened or exported as a PDF report and shared.

The app requires exactly one permission: the camera.

## Repository layout

```text
app/       Android application (UI, camera, TFLite, Room, PDF reports)
core/      Pure-JVM Kotlin module: domain models and logic (fully unit-tested)
training/  Python pipeline that trains and exports the TFLite model
docs/      Architecture, setup, model contract, testing, release, privacy
```

## Quick start

```bash
git clone <this repo>
cd breed-classifier-mobile-app
./gradlew :app:assembleDebug     # requires JDK 17 + Android SDK 35
```

Open in Android Studio (Ladybug or newer) and run on a device or emulator
(minSdk 24).

**Note on the model:** the app bundles a real 5-breed production TFLite model
under `app/src/main/assets/models/`. It supports Gir, Hallikar, Murrah,
Sahiwal, and Tharparkar offline. See [docs/MODEL.md](docs/MODEL.md) and
[training/reports/combined_kaggle_v2_5class](training/reports/combined_kaggle_v2_5class)
for measured metrics and dataset provenance.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Modules, layers, data flow, key decisions |
| [docs/SETUP.md](docs/SETUP.md) | Development environment setup |
| [docs/MODEL.md](docs/MODEL.md) | The TFLite model contract the app enforces |
| [docs/TESTING.md](docs/TESTING.md) | Test strategy, running tests, manual QA checklist |
| [docs/RELEASE.md](docs/RELEASE.md) | Release checklist, signing, versioning |
| [docs/PRIVACY.md](docs/PRIVACY.md) | Data handling facts for a privacy policy |
| [docs/AUDIT.md](docs/AUDIT.md) | The audit that motivated the 2026 production rebuild |
| [docs/SHIP_REPORT.md](docs/SHIP_REPORT.md) | Production release-candidate evidence and remaining operator tasks |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Workflow and standards for contributors |

## Status

Software and model are release-candidate ready: the bundled model is
contract-verified and honestly evaluated at 88.00% held-out top-1 accuracy,
100.00% top-3 accuracy, and 63.33% minimum class recall across the five
supported breeds. Store release still requires operator-owned signing keys,
Play Console upload, a hosted privacy policy URL, physical-device camera QA,
and a repository license choice; see [docs/SHIP_REPORT.md](docs/SHIP_REPORT.md)
and [docs/RELEASE.md](docs/RELEASE.md).
