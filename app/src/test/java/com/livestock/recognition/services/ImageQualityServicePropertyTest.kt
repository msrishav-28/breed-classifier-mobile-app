package com.livestock.recognition.services

import android.graphics.Bitmap
import android.graphics.Color
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Property-based tests for ImageQualityService
 * **Feature: cattle-buffalo-recognition, Property 2: Image quality validation feedback**
 * **Validates: Requirements 1.3**
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ImageQualityServicePropertyTest {
    
    private val imageQualityService = ImageQualityService()
    
    /**
     * Property: For any captured image with quality issues (poor lighting, blur, low resolution), 
     * the system should provide appropriate feedback messages
     */
    @Test
    fun `property test - images with quality issues should provide appropriate feedback`() {
        checkAll(100, imageWithQualityIssuesGenerator()) { testImage ->
            val report = imageQualityService.validateImageQuality(testImage.bitmap)
            
            // Verify that quality issues are detected and feedback is provided
            when (testImage.expectedIssue) {
                QualityIssue.LOW_RESOLUTION -> {
                    assert(report.issues.contains(QualityIssue.LOW_RESOLUTION)) {
                        "Low resolution image should be detected as having resolution issues"
                    }
                    assert(report.feedback.any { it.contains("resolution", ignoreCase = true) }) {
                        "Feedback should mention resolution issues"
                    }
                }
                QualityIssue.TOO_DARK -> {
                    assert(report.issues.contains(QualityIssue.TOO_DARK)) {
                        "Dark image should be detected as too dark"
                    }
                    assert(report.feedback.any { it.contains("dark", ignoreCase = true) || it.contains("lighting", ignoreCase = true) }) {
                        "Feedback should mention darkness or lighting issues"
                    }
                }
                QualityIssue.TOO_BRIGHT -> {
                    assert(report.issues.contains(QualityIssue.TOO_BRIGHT)) {
                        "Bright image should be detected as too bright"
                    }
                    assert(report.feedback.any { it.contains("bright", ignoreCase = true) || it.contains("lighting", ignoreCase = true) }) {
                        "Feedback should mention brightness or lighting issues"
                    }
                }
                QualityIssue.LOW_CONTRAST -> {
                    assert(report.issues.contains(QualityIssue.LOW_CONTRAST)) {
                        "Low contrast image should be detected as having contrast issues"
                    }
                    assert(report.feedback.any { it.contains("contrast", ignoreCase = true) }) {
                        "Feedback should mention contrast issues"
                    }
                }
                QualityIssue.TOO_BLURRY -> {
                    assert(report.issues.contains(QualityIssue.TOO_BLURRY)) {
                        "Blurry image should be detected as too blurry"
                    }
                    assert(report.feedback.any { it.contains("blur", ignoreCase = true) || it.contains("focus", ignoreCase = true) }) {
                        "Feedback should mention blur or focus issues"
                    }
                }
            }
            
            // Verify that feedback is not empty for problematic images
            assert(report.feedback.isNotEmpty()) {
                "Images with quality issues should always provide feedback"
            }
            
            // Verify that quality score reflects the issues
            assert(report.qualityScore < 100.0) {
                "Images with quality issues should have quality score less than 100"
            }
            
            // Verify that isAcceptable is false for images with issues
            assert(!report.isAcceptable) {
                "Images with quality issues should not be marked as acceptable"
            }
        }
    }
    
    /**
     * Property: For any good quality image, the system should not report false positive issues
     */
    @Test
    fun `property test - good quality images should not have false positive issues`() {
        checkAll(100, goodQualityImageGenerator()) { bitmap ->
            val report = imageQualityService.validateImageQuality(bitmap)
            
            // Good quality images should have minimal issues
            assert(report.qualityScore >= 70.0) {
                "Good quality images should have quality score >= 70"
            }
            
            // Should not have resolution issues
            assert(!report.issues.contains(QualityIssue.LOW_RESOLUTION)) {
                "Good quality images should not have resolution issues"
            }
            
            // Metrics should be within reasonable ranges
            val brightness = report.metrics["brightness"] ?: 0.0
            assert(brightness in 30.0..225.0) {
                "Good quality images should have reasonable brightness: $brightness"
            }
            
            val contrast = report.metrics["contrast"] ?: 0.0
            assert(contrast >= 20.0) {
                "Good quality images should have adequate contrast: $contrast"
            }
        }
    }
    
    /**
     * Property: Quality validation should be consistent - same image should produce same results
     */
    @Test
    fun `property test - quality validation should be deterministic`() {
        checkAll(50, anyImageGenerator()) { bitmap ->
            val report1 = imageQualityService.validateImageQuality(bitmap)
            val report2 = imageQualityService.validateImageQuality(bitmap)
            
            // Results should be identical
            assert(report1.isAcceptable == report2.isAcceptable) {
                "Quality validation should be deterministic for acceptability"
            }
            
            assert(report1.qualityScore == report2.qualityScore) {
                "Quality validation should be deterministic for quality score"
            }
            
            assert(report1.issues == report2.issues) {
                "Quality validation should be deterministic for issues list"
            }
            
            assert(report1.feedback == report2.feedback) {
                "Quality validation should be deterministic for feedback"
            }
        }
    }
    
    // Generators for property-based testing
    
    private fun imageWithQualityIssuesGenerator(): Arb<TestImageWithIssue> {
        return Arb.choice(
            lowResolutionImageGenerator().map { TestImageWithIssue(it, QualityIssue.LOW_RESOLUTION) },
            darkImageGenerator().map { TestImageWithIssue(it, QualityIssue.TOO_DARK) },
            brightImageGenerator().map { TestImageWithIssue(it, QualityIssue.TOO_BRIGHT) },
            lowContrastImageGenerator().map { TestImageWithIssue(it, QualityIssue.LOW_CONTRAST) }
        )
    }
    
    private fun lowResolutionImageGenerator(): Arb<Bitmap> {
        return Arb.bind(
            Arb.int(50, 200), // width
            Arb.int(50, 200)  // height
        ) { width, height ->
            createSolidColorBitmap(width, height, Color.GRAY)
        }
    }
    
    private fun darkImageGenerator(): Arb<Bitmap> {
        return Arb.bind(
            Arb.int(300, 500), // width
            Arb.int(300, 500), // height
            Arb.int(0, 25)     // dark color value
        ) { width, height, colorValue ->
            val darkColor = Color.rgb(colorValue, colorValue, colorValue)
            createSolidColorBitmap(width, height, darkColor)
        }
    }
    
    private fun brightImageGenerator(): Arb<Bitmap> {
        return Arb.bind(
            Arb.int(300, 500), // width
            Arb.int(300, 500), // height
            Arb.int(230, 255)  // bright color value
        ) { width, height, colorValue ->
            val brightColor = Color.rgb(colorValue, colorValue, colorValue)
            createSolidColorBitmap(width, height, brightColor)
        }
    }
    
    private fun lowContrastImageGenerator(): Arb<Bitmap> {
        return Arb.bind(
            Arb.int(300, 500), // width
            Arb.int(300, 500), // height
            Arb.int(100, 120)  // similar color values for low contrast
        ) { width, height, baseColor ->
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(bitmap)
            
            // Fill with very similar colors to create low contrast
            val color1 = Color.rgb(baseColor, baseColor, baseColor)
            val color2 = Color.rgb(baseColor + 5, baseColor + 5, baseColor + 5)
            
            val paint = android.graphics.Paint()
            paint.color = color1
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)
            
            paint.color = color2
            canvas.drawRect(width * 0.3f, height * 0.3f, width * 0.7f, height * 0.7f, paint)
            
            bitmap
        }
    }
    
    private fun goodQualityImageGenerator(): Arb<Bitmap> {
        return Arb.bind(
            Arb.int(300, 800), // width
            Arb.int(300, 800), // height
            Arb.int(50, 200)   // base color for good contrast
        ) { width, height, baseColor ->
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(bitmap)
            
            // Create a pattern with good contrast
            val paint = android.graphics.Paint()
            paint.color = Color.rgb(baseColor, baseColor, baseColor)
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)
            
            paint.color = Color.rgb(255 - baseColor, 255 - baseColor, 255 - baseColor)
            canvas.drawCircle(width / 2f, height / 2f, minOf(width, height) / 4f, paint)
            
            bitmap
        }
    }
    
    private fun anyImageGenerator(): Arb<Bitmap> {
        return Arb.choice(
            goodQualityImageGenerator(),
            lowResolutionImageGenerator(),
            darkImageGenerator(),
            brightImageGenerator(),
            lowContrastImageGenerator()
        )
    }
    
    private fun createSolidColorBitmap(width: Int, height: Int, color: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(color)
        return bitmap
    }
    
    private data class TestImageWithIssue(
        val bitmap: Bitmap,
        val expectedIssue: QualityIssue
    )
}