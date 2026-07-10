# Release Guide

## Versioning

Semantic versioning lives in `app/build.gradle.kts`: bump `versionName`
(`MAJOR.MINOR.PATCH`) and increment `versionCode` monotonically for every
store upload. Tag releases `vX.Y.Z`.

The current release candidate is `versionName "1.0.0"` / `versionCode 1`.

## Release Checklist

Software gates:

- [ ] CI green: core tests, app tests, lint, debug + release assembly
- [ ] CI e2e green: Android emulator instrumented tests
- [ ] No new lint warnings introduced

Model gates:

- [x] Real TFLite model bundled in `app/src/main/assets/models/`
- [x] Model contract verified at export and app load time
- [x] Held-out TEST top-1 >= 85%: measured 88.00%
- [x] Held-out TEST top-3 >= 95%: measured 100.00%
- [x] Minimum class recall >= 60%: measured 63.33%
- [x] Model binary is under the repository cap: 11.3 MB

Release-specific steps:

- [ ] Run the manual QA checklist (`docs/TESTING.md`) on at least one low-end
      2-3 GB RAM device and one recent device
- [ ] Measure cold start and inference latency on the low-end device;
      inference must stay under 3 seconds
- [ ] Bump `versionCode` / `versionName` if needed; tag the final commit
      `vX.Y.Z`
- [ ] Build a signed release: create a keystore outside the repo and provide
      it via `keystore.properties` locally or the CI secrets below
- [ ] Store listing: screenshots, feature graphic, description, camera
      permission declaration
- [ ] Host a privacy policy and link it in the listing
- [ ] Complete Play Console data-safety form: no data collected, no data
      shared, all processing on device
- [ ] Decide and add a repository LICENSE before publishing source or
      accepting external contributions

## Signing Configuration

Signing is already wired in `app/build.gradle.kts` and activates only when a
`keystore.properties` file exists at the repository root:

```properties
storeFile=release.keystore
storePassword=...
keyAlias=...
keyPassword=...
```

`keystore.properties` and keystore files are gitignored. Without the file,
release builds succeed unsigned (CI relies on this).

Create an operator-owned upload keystore outside the repository:

```bash
keytool -genkeypair \
  -v \
  -keystore release.keystore \
  -alias livestock-breed-release \
  -keyalg RSA \
  -keysize 4096 \
  -validity 10000
```

The tag-triggered Release workflow (`.github/workflows/release.yml`) signs
automatically when these repository secrets are configured:

| Secret | Contents |
|--------|----------|
| `RELEASE_KEYSTORE_BASE64` | Base64-encoded keystore file |
| `RELEASE_KEYSTORE_PASSWORD` | Store and key password |
| `RELEASE_KEY_ALIAS` | Key alias |

On macOS/Linux, encode with:

```bash
base64 -w0 release.keystore
```

On Windows PowerShell, encode with:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("release.keystore"))
```

Push a `vX.Y.Z` tag and the workflow runs checks, builds APK + AAB, and
attaches them to a GitHub release. It signs when the secrets exist and builds
unsigned artifacts otherwise.

## Model Distribution

The APK/AAB embeds the production model at build time. The current model is
committed because it is 11.3 MB, under the documented roughly 15 MB cap:

- `app/src/main/assets/models/breed_classifier.tflite`
- `app/src/main/assets/models/labels.txt`

If a future production model exceeds the cap, do not commit it. Attach the
model and `labels.txt` to a release artifact instead, wire CI/release builds
to fetch them, and update this section plus `docs/MODEL.md`.
