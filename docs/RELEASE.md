# Release guide

## Versioning

Semantic versioning in `app/build.gradle.kts`: bump `versionName`
(`MAJOR.MINOR.PATCH`) and increment `versionCode` monotonically for every
store upload. Tag releases `vX.Y.Z`.

## Release checklist

Software gates (enforced by CI):

- [ ] CI green: core tests, app tests, lint, debug + release assembly
- [ ] No new lint warnings introduced

Release-specific steps:

- [ ] **Bundle a trained model** (`app/src/main/assets/models/`, see
      docs/MODEL.md) and record its validation metrics from
      `training/build/metrics.json`
- [ ] Run the manual QA checklist (docs/TESTING.md) on at least one low-end
      (2–3 GB RAM) and one recent device
- [ ] Measure cold start and inference latency on the low-end device;
      inference must stay under 3 s
- [ ] Bump `versionCode` / `versionName`; tag the commit `vX.Y.Z` — the
      Release workflow builds the APK + AAB and attaches them to a GitHub
      release automatically
- [ ] Build a **signed** release: create a keystore (never commit it) and
      provide it via `keystore.properties` locally or the CI secrets below;
      the Gradle signing config activates automatically when present
- [ ] Store listing: screenshots (light + dark), feature graphic, 
      description; declare the camera permission usage
- [ ] Host a privacy policy (docs/PRIVACY.md has the facts; the app collects
      nothing, which makes this short) and link it in the listing
- [ ] Complete Play Console data-safety form: no data collected, no data
      shared, all processing on device
- [ ] Decide and add a repository LICENSE before publishing source

## Signing configuration

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

The tag-triggered Release workflow (`.github/workflows/release.yml`) signs
automatically when these repository secrets are configured:

| Secret | Contents |
|--------|----------|
| `RELEASE_KEYSTORE_BASE64` | `base64 -w0 release.keystore` |
| `RELEASE_KEYSTORE_PASSWORD` | store and key password |
| `RELEASE_KEY_ALIAS` | key alias |

Push a `vX.Y.Z` tag and the workflow runs checks, builds APK + AAB, and
attaches them to a GitHub release (unsigned when no secrets are set).

## Distribution of the model

The APK/AAB embeds the model at build time. Because model binaries are not
in git, the release pipeline (or the engineer cutting the release) must fetch
the blessed model artifact — attach it to the GitHub release of the training
run that produced it, alongside its `metrics.json`.
