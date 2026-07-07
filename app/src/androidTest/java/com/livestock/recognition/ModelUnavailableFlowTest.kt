package com.livestock.recognition

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.livestock.recognition.ui.results.ResultsActivity
import org.junit.Assume.assumeFalse
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * UI-level proof of the degraded path: builds that ship without a model
 * (the default — model binaries are not committed) must explain the
 * situation on the results screen instead of crashing or spinning.
 */
@RunWith(AndroidJUnit4::class)
class ModelUnavailableFlowTest {

    @Test
    fun resultsScreenExplainsMissingModel() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val app = appContext.applicationContext as LivestockApp

        // Only meaningful when no model is bundled in the app under test.
        assumeFalse(
            "Model is bundled; degraded-path test does not apply",
            app.container.classifierProvider.isModelBundled(),
        )

        val photo = File(appContext.filesDir, "e2e_test_photo.jpg")
        Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.rgb(150, 100, 60))
            photo.outputStream().use { compress(Bitmap.CompressFormat.JPEG, 90, it) }
            recycle()
        }

        val intent = ResultsActivity.newClassificationIntent(appContext, photo.absolutePath)
        try {
            ActivityScenario.launch<ResultsActivity>(intent).use {
                onView(withId(R.id.errorText)).check(matches(isDisplayed()))
                onView(withText(R.string.model_unavailable_message)).check(matches(isDisplayed()))
            }
        } finally {
            photo.delete()
        }
    }
}
