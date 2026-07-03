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
- [ ] Bump `versionCode` / `versionName`; tag the commit
- [ ] Build a **signed** release: create a keystore (never commit it), add a
      `signingConfig` pointing at it via environment variables or a local
      `keystore.properties`, then `./gradlew :app:bundleRelease` for the
      Play-Store `.aab`
- [ ] Store listing: screenshots (light + dark), feature graphic, 
      description; declare the camera permission usage
- [ ] Host a privacy policy (docs/PRIVACY.md has the facts; the app collects
      nothing, which makes this short) and link it in the listing
- [ ] Complete Play Console data-safety form: no data collected, no data
      shared, all processing on device
- [ ] Decide and add a repository LICENSE before publishing source

## Signing configuration sketch

`app/build.gradle.kts` deliberately ships without a signing config. Add:

```kotlin
signingConfigs {
    create("release") {
        val props = java.util.Properties().apply {
            val f = rootProject.file("keystore.properties")
            if (f.exists()) f.inputStream().use { load(it) }
        }
        storeFile = props.getProperty("storeFile")?.let { file(it) }
        storePassword = props.getProperty("storePassword")
        keyAlias = props.getProperty("keyAlias")
        keyPassword = props.getProperty("keyPassword")
    }
}
```

and set `signingConfig = signingConfigs.getByName("release")` in the release
build type. `keystore.properties` and `*.jks` are gitignored.

## Distribution of the model

The APK/AAB embeds the model at build time. Because model binaries are not
in git, the release pipeline (or the engineer cutting the release) must fetch
the blessed model artifact — attach it to the GitHub release of the training
run that produced it, alongside its `metrics.json`.
