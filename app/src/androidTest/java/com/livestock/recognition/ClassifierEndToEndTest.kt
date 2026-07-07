package com.livestock.recognition

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.livestock.recognition.core.catalog.BreedCatalog
import com.livestock.recognition.ml.TfLiteBreedClassifier
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device end-to-end proof of the ML path: loads the committed fixture
 * model (production contract, tiny CNN trained on procedural textures —
 * see training/export_test_model.py) from the *test* APK's assets, runs
 * real TFLite inference on fixture images, and checks the results resolve
 * against the app's bundled breed catalog.
 */
@RunWith(AndroidJUnit4::class)
class ClassifierEndToEndTest {

    // Test-APK context: fixture model + images live in androidTest assets.
    private val testContext = InstrumentationRegistry.getInstrumentation().context

    // App context: the real bundled breed catalog.
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext

    private fun classify(assetImage: String) = runBlocking {
        val classifier = TfLiteBreedClassifier.create(testContext)
        try {
            val bitmap = testContext.assets.open(assetImage).use {
                BitmapFactory.decodeStream(it)
            }
            assertNotNull("Fixture image $assetImage failed to decode", bitmap)
            classifier.classify(bitmap!!)
        } finally {
            classifier.close()
        }
    }

    @Test
    fun fixtureModelSatisfiesContractAndClassifiesFixtures() {
        val output = classify("fixtures/gir_sample.jpg")

        assertEquals(3, output.predictions.size)
        val top = output.predictions.first()
        assertEquals("Gir", top.label)
        assertTrue("Expected confident prediction, got ${top.confidence}",
            top.confidence > 0.8f)
        assertTrue(output.predictions.zipWithNext().all { (a, b) ->
            a.confidence >= b.confidence
        })
        assertTrue(output.processingTimeMillis >= 0)
    }

    @Test
    fun secondClassDistinguishedFromFirst() {
        val output = classify("fixtures/murrah_sample.jpg")
        assertEquals("Murrah", output.predictions.first().label)
    }

    @Test
    fun predictionsResolveAgainstBundledCatalog() {
        val catalogResult = appContext.assets.open("data/breed_mapping.csv")
            .bufferedReader().use { BreedCatalog.parse(it) }
        assertTrue(catalogResult is BreedCatalog.ParseResult.Success)
        val catalog = (catalogResult as BreedCatalog.ParseResult.Success).catalog

        val output = classify("fixtures/gir_sample.jpg")
        val info = catalog.find(output.predictions.first().label)
        assertNotNull("Top prediction has no catalog entry", info)
        assertEquals("Gir", info!!.name)
    }
}
