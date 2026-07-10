package com.livestock.recognition

import android.content.ContentValues
import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.os.Build
import android.os.Environment
import android.os.SystemClock
import android.provider.MediaStore
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.livestock.recognition.core.report.ReportContentBuilder
import com.livestock.recognition.ml.ClassifierProvider
import com.livestock.recognition.ui.common.formatDateTime
import com.livestock.recognition.ui.history.HistoryActivity
import com.livestock.recognition.ui.results.ResultsActivity
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Release-candidate flow evidence for the bundled model build. It mirrors the
 * manual photo-result-history-report check and writes screenshots/PDF under
 * app files so they can be pulled with `adb run-as` after the test.
 */
@RunWith(AndroidJUnit4::class)
class ReleaseManualFlowTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val appContext = instrumentation.targetContext

    @Test
    fun resultHistoryAndPdfReportWorkForHeldOutSample() {
        val app = appContext.applicationContext as LivestockApp
        assumeTrue(
            "No bundled production model in app assets",
            app.container.classifierProvider.isModelBundled(),
        )

        val qaDir = File(appContext.filesDir, "qa").apply {
            deleteRecursively()
            mkdirs()
        }
        val reportsDir = File(appContext.filesDir, "reports").apply {
            deleteRecursively()
            mkdirs()
        }
        val photo = File(qaDir, "gir_manual.jpg")
        testContext.assets.open(SAMPLE_ASSET).use { input ->
            photo.outputStream().use { output -> input.copyTo(output) }
        }

        ActivityScenario.launch<ResultsActivity>(
            ResultsActivity.newClassificationIntent(appContext, photo.absolutePath)
        ).use { scenario ->
            waitForText("Gir", RESULT_TIMEOUT_MS)
            onView(withId(R.id.breedNameText)).check(matches(withText("Gir")))
            onView(withId(R.id.breedInfoCard)).check(matches(isDisplayed()))
            saveScreenshot(scenario, qaDir, "results_manual.png")
        }

        val report = generateReportFromSavedHistory(app, reportsDir)
        assertPdfHeader(report)
        val reportEvidence = report.copyTo(File(qaDir, "manual_report.pdf"), overwrite = true)
        exportForPull(reportEvidence, MIME_PDF)

        ActivityScenario.launch(HistoryActivity::class.java).use { scenario ->
            waitForText("Gir", HISTORY_TIMEOUT_MS)
            SystemClock.sleep(UI_SETTLE_MS)
            saveScreenshot(scenario, qaDir, "history_manual.png")
        }
    }

    private fun waitForText(text: String, timeoutMs: Long) {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        var lastFailure: Throwable? = null
        while (SystemClock.elapsedRealtime() < deadline) {
            try {
                onView(withText(text)).check(matches(isDisplayed()))
                return
            } catch (failure: Throwable) {
                lastFailure = failure
                SystemClock.sleep(POLL_INTERVAL_MS)
            }
        }
        throw AssertionError("Timed out waiting for visible text: $text", lastFailure)
    }

    private fun generateReportFromSavedHistory(app: LivestockApp, reportsDir: File): File =
        runBlocking {
            val saved = app.container.historyRepository.observeAll().first().firstOrNull()
                ?: throw AssertionError("No saved history entry after classification")
            val catalog = app.container.breedCatalogProvider.catalog
            val photo = BitmapFactory.decodeFile(saved.imagePath)
            try {
                val content = ReportContentBuilder.build(
                    record = saved.record,
                    breedInfo = catalog?.find(saved.record.breedLabel),
                    generatedAt = formatDateTime(System.currentTimeMillis()),
                    capturedAt = formatDateTime(saved.record.capturedAtEpochMillis),
                )
                val report = app.container.reportGenerator.generate(content, photo)
                assertTrue(
                    "Expected report under ${reportsDir.absolutePath}, got ${report.absolutePath}",
                    report.parentFile?.absolutePath == reportsDir.absolutePath,
                )
                report
            } finally {
                photo?.recycle()
            }
        }

    private fun assertPdfHeader(report: File) {
        val header = ByteArray(PDF_HEADER.length)
        val count = report.inputStream().use { it.read(header) }
        assertEquals(PDF_HEADER.length, count)
        assertEquals(PDF_HEADER, String(header, Charsets.US_ASCII))
    }

    private fun <A : Activity> saveScreenshot(
        scenario: ActivityScenario<A>,
        dir: File,
        name: String,
    ) {
        instrumentation.waitForIdleSync()
        scenario.onActivity { activity ->
            val root = activity.window.decorView.rootView
            val screenshot = Bitmap.createBitmap(
                root.width.coerceAtLeast(1),
                root.height.coerceAtLeast(1),
                Bitmap.Config.ARGB_8888,
            )
            root.draw(Canvas(screenshot))
            File(dir, name).outputStream().use { output ->
                assertTrue(screenshot.compress(Bitmap.CompressFormat.PNG, 100, output))
            }
            screenshot.recycle()
            exportForPull(File(dir, name), MIME_PNG)
        }
    }

    private fun exportForPull(file: File, mimeType: String) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return

        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, file.name)
            put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
            put(MediaStore.MediaColumns.RELATIVE_PATH, PULL_DIR)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val resolver = appContext.contentResolver
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: throw AssertionError("Could not create MediaStore entry for ${file.name}")
        resolver.openOutputStream(uri)?.use { output ->
            file.inputStream().use { input -> input.copyTo(output) }
        } ?: throw AssertionError("Could not open MediaStore output for ${file.name}")
        values.clear()
        values.put(MediaStore.MediaColumns.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
    }

    private companion object {
        const val SAMPLE_ASSET = "real_samples/gir_sample.jpg"
        const val RESULT_TIMEOUT_MS = 20_000L
        const val HISTORY_TIMEOUT_MS = 5_000L
        const val POLL_INTERVAL_MS = 250L
        const val UI_SETTLE_MS = 1_000L
        const val PDF_HEADER = "%PDF"
        const val MIME_PDF = "application/pdf"
        const val MIME_PNG = "image/png"
        val PULL_DIR = "${Environment.DIRECTORY_DOWNLOADS}/breed_classifier_qa"
    }
}
