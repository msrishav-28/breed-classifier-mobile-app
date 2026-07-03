# Development setup

## Prerequisites

- **JDK 17** (Temurin recommended)
- **Android SDK 35** with build-tools (installed automatically by Android
  Studio Ladybug+, or via `sdkmanager "platforms;android-35"`)
- Android Studio Ladybug (2024.2) or newer, or plain Gradle from the CLI

## Build

```bash
./gradlew :app:assembleDebug        # debug APK
./gradlew :app:assembleRelease      # R8-minified release (unsigned)
./gradlew build                     # everything incl. lint and unit tests
```

Point Gradle at your SDK with a `local.properties` containing
`sdk.dir=/path/to/Android/sdk` if you build outside Android Studio.

## Run tests

```bash
./gradlew :core:test                # domain unit + property tests (no SDK needed)
./gradlew :app:testDebugUnitTest    # app-layer unit tests
./gradlew :app:lintDebug            # Android lint
./gradlew :app:connectedDebugAndroidTest   # smoke test, device required
```

`:core` is a pure JVM module — it builds and tests without any Android SDK,
which keeps the domain-logic feedback loop fast.

## Bundling a model (optional but needed for classification)

```bash
mkdir -p app/src/main/assets/models
cp <your>/breed_classifier.tflite app/src/main/assets/models/
cp <your>/labels.txt              app/src/main/assets/models/
```

Without these files the app still builds and runs, showing a "model not
installed" notice. See `docs/MODEL.md` for the contract and
`training/README.md` for producing a model.

## Project conventions

- Dependency versions live in `gradle/libs.versions.toml` (version catalog).
- `:core` must never gain an Android dependency.
- All user-visible text goes through `res/values/strings.xml`.
- CI (`.github/workflows/ci.yml`) must be green before merging.
