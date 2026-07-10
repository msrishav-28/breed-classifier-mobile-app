package com.livestock.recognition

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.livestock.recognition.ml.ClassifierProvider
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Production-model smoke test: runs the bundled main-assets model against
 * held-out real photos and checks the correct breed is present in top-3.
 */
@RunWith(AndroidJUnit4::class)
class BundledModelSanityTest {

    private val testContext = InstrumentationRegistry.getInstrumentation().context
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun bundledModelRecognizesHeldOutRealSamplesInTop3() = runBlocking {
        val provider = ClassifierProvider(appContext)
        assumeTrue("No bundled production model in app assets", provider.isModelBundled())

        val state = provider.get()
        assertTrue("Bundled model failed to load: $state", state is ClassifierProvider.State.Ready)
        val classifier = (state as ClassifierProvider.State.Ready).classifier

        try {
            samples.forEach { sample ->
                val bitmap = testContext.assets.open(sample.assetPath).use {
                    BitmapFactory.decodeStream(it)
                }
                assertNotNull("Sample ${sample.assetPath} failed to decode", bitmap)

                val output = classifier.classify(bitmap!!)
                val top3 = output.predictions.map { it.label }
                Log.i(TAG, "${sample.expectedBreed}: top3=$top3 latency=${output.processingTimeMillis}ms")

                assertTrue(
                    "${sample.expectedBreed} was not in top-3 for ${sample.assetPath}: $top3",
                    sample.expectedBreed in top3,
                )
                assertTrue(
                    "Inference took ${output.processingTimeMillis}ms, expected under ${MAX_LATENCY_MS}ms",
                    output.processingTimeMillis < MAX_LATENCY_MS,
                )
            }
        } finally {
            classifier.close()
        }
    }

    private data class Sample(val assetPath: String, val expectedBreed: String)

    private companion object {
        const val TAG = "BundledModelSanityTest"
        const val MAX_LATENCY_MS = 3_000L

        val samples = listOf(
            Sample("real_samples/gir_sample.jpg", "Gir"),
            Sample("real_samples/hallikar_sample.jpg", "Hallikar"),
            Sample("real_samples/murrah_sample.jpg", "Murrah"),
            Sample("real_samples/sahiwal_sample.jpg", "Sahiwal"),
            Sample("real_samples/tharparkar_sample.jpg", "Tharparkar"),
        )
    }
}
