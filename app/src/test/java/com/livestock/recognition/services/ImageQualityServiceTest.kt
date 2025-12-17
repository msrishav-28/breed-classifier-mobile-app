package com.livestock.recognition.services

import android.graphics.Bitmap
import android.graphics.Color
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Unit tests for ImageQualityService
 * Tests specific quality validation scenarios and edge cases
 * 
 * Requirements: 1.1, 1.3, 1.5
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ImageQualityServiceTest {
    
    private lateinit var imageQualityService: ImageQualityService
    
    @Before
    fun setUp() {
        imageQualityService = ImageQualityService()
    }
    
    @Test
    fun `validateImageQuality should detect low resolution images`() {
        val lowResBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        lowResBitmap.eraseColor(Color.GRAY)
        
        val report = imageQualityService.validateImageQuality(lowResBitmap)
        
        assertTrue("Should detect low resolution", report.issues.contains(QualityIssue.LOW_RESOLUTION))
        assertFalse("Low resolution image should not be acceptable", report.isAcceptable)
        assertTrue("Should provide feedback", report.feedback.isNotEmpty())
        assertTrue("Feedback should mention resolution", 
            report.feedback.any { it.contains("resolution", ignoreCase = true) })
    }
    
    @Test
    fun `validateImageQuality should detect dark images`() {
        val darkBitmap = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        darkBitmap.eraseColor(Color.rgb(10, 10, 10)) // Very dark
        
        val report = imageQualityService.validateImageQuality(darkBitmap)
        
        assertTrue("Should detect dark image", report.issues.contains(QualityIssue.TOO_DARK))
        assertFalse("Dark image should not be acceptable", report.isAcceptable)
        assertTrue("Should provide feedback about darkness", 
            report.feedback.any { it.contains("dark", ignoreCase = true) })
        
        val brightness = report.metrics["brightness"] ?: 0.0
        assertTrue("Brightness should be low", brightness < 30.0)
    }
    
    @Test
    fun `validateImageQuality should detect bright images`() {
        val brightBitmap = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        brightBitmap.eraseColor(Color.rgb(250, 250, 250)) // Very bright
        
        val report = imageQualityService.validateImageQuality(brightBitmap)
        
        assertTrue("Should detect bright image", report.issues.contains(QualityIssue.TOO_BRIGHT))
        assertFalse("Bright image should not be acceptable", report.isAcceptable)
        assertTrue("Should provide feedback about brightness", 
            report.feedback.any { it.contains("bright", ignoreCase = true) })
        
        val brightness = report.metrics["brightness"] ?: 0.0
        assertTrue("Brightness should be high", brightness > 225.0)
    }
    
    @Test
    fun `validateImageQuality should accept good quality images`() {
        val goodBitmap = createGoodQualityTestImage(400, 400)
        
        val report = imageQualityService.validateImageQuality(goodBitmap)
        
        assertTrue("Good quality image should be acceptable", report.isAcceptable)
        assertTrue("Should have high quality score", report.qualityScore >= 70.0)
        assertTrue("Should have minimal issues", report.issues.size <= 1)
        
        val brightness = report.metrics["brightness"] ?: 0.0
        assertTrue("Brightness should be in good range", brightness in 30.0..225.0)
    }
    
    @Test
    fun `preprocessForInference should resize image correctly`() {
        val originalBitmap = Bitmap.createBitmap(800, 600, Bitmap.Config.ARGB_8888)
        originalBitmap.eraseColor(Color.GRAY)
        
        val processed = imageQualityService.preprocessForInference(originalBitmap)
        
        assertEquals("Should preserve original width", 800, processed.originalWidth)
        assertEquals("Should preserve original height", 600, processed.originalHeight)
        assertEquals("Should resize to 224x224", 224, processed.processedWidth)
        assertEquals("Should resize to 224x224", 224, processed.processedHeight)
        assertTrue("Should record processing time", processed.processingTimeMs >= 0)
        assertNotNull("Processed bitmap should not be null", processed.bitmap)
    }
    
    @Test
    fun `validateImageQuality should handle edge case dimensions`() {
        // Test very small image
        val tinyBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        tinyBitmap.eraseColor(Color.GRAY)
        
        val report = imageQualityService.validateImageQuality(tinyBitmap)
        
        assertTrue("Should detect low resolution", report.issues.contains(QualityIssue.LOW_RESOLUTION))
        assertFalse("Tiny image should not be acceptable", report.isAcceptable)
    }
    
    @Test
    fun `validateImageQuality should calculate quality score correctly`() {
        val perfectBitmap = createGoodQualityTestImage(500, 500)
        val report = imageQualityService.validateImageQuality(perfectBitmap)
        
        assertTrue("Perfect image should have high score", report.qualityScore >= 80.0)
        assertTrue("Perfect image should be acceptable", report.isAcceptable)
        
        // Test image with multiple issues
        val problematicBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        problematicBitmap.eraseColor(Color.rgb(5, 5, 5)) // Low res + dark
        
        val problematicReport = imageQualityService.validateImageQuality(problematicBitmap)
        assertTrue("Problematic image should have low score", problematicReport.qualityScore < 50.0)
        assertFalse("Problematic image should not be acceptable", problematicReport.isAcceptable)
    }
    
    @Test
    fun `validateImageQuality should provide appropriate feedback messages`() {
        val lowResBitmap = Bitmap.createBitmap(50, 50, Bitmap.Config.ARGB_8888)
        val report = imageQualityService.validateImageQuality(lowResBitmap)
        
        assertTrue("Should have feedback", report.feedback.isNotEmpty())
        assertTrue("Feedback should be descriptive", 
            report.feedback.all { it.length > 10 })
        assertTrue("Should provide actionable feedback", 
            report.feedback.any { it.contains("Please", ignoreCase = true) })
    }
    
    @Test
    fun `validateImageQuality should be deterministic`() {
        val testBitmap = createGoodQualityTestImage(300, 300)
        
        val report1 = imageQualityService.validateImageQuality(testBitmap)
        val report2 = imageQualityService.validateImageQuality(testBitmap)
        
        assertEquals("Results should be identical", report1.isAcceptable, report2.isAcceptable)
        assertEquals("Quality scores should be identical", report1.qualityScore, report2.qualityScore, 0.001)
        assertEquals("Issues should be identical", report1.issues, report2.issues)
        assertEquals("Feedback should be identical", report1.feedback, report2.feedback)
    }
    
    @Test
    fun `preprocessForInference should handle various input sizes`() {
        val sizes = listOf(
            Pair(100, 100),
            Pair(500, 300),
            Pair(1000, 800),
            Pair(224, 224)
        )
        
        sizes.forEach { (width, height) ->
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmap.eraseColor(Color.GRAY)
            
            val processed = imageQualityService.preprocessForInference(bitmap)
            
            assertEquals("Should preserve original width for $width x $height", width, processed.originalWidth)
            assertEquals("Should preserve original height for $width x $height", height, processed.originalHeight)
            assertEquals("Should resize to 224 width for $width x $height", 224, processed.processedWidth)
            assertEquals("Should resize to 224 height for $width x $height", 224, processed.processedHeight)
        }
    }
    
    private fun createGoodQualityTestImage(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(bitmap)
        
        // Create a pattern with good contrast and brightness
        val paint = android.graphics.Paint()
        paint.color = Color.rgb(100, 100, 100)
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)
        
        paint.color = Color.rgb(200, 200, 200)
        canvas.drawCircle(width / 2f, height / 2f, minOf(width, height) / 4f, paint)
        
        return bitmap
    }
}