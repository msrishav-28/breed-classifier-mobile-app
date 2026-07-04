// Root build script. Plugin versions are declared in gradle/libs.versions.toml
// and applied by the individual modules; nothing is applied at the root so that
// the pure-JVM :core module can be built without the Android toolchain present.
tasks.register("clean", Delete::class) {
    delete(rootProject.layout.buildDirectory)
}
