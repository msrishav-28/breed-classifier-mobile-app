package com.livestock.recognition.services

import android.graphics.Bitmap
import android.graphics.Color
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

/**
 * Unit tests for ImageCaptureService
 * Tests image loading, quality validation, and preprocessing
 * 
 * Requirements: 1.1, 1.3, 1.5
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ImageCaptureServiceTest {
    
    @Mock
    private lateinit var mockQualityService: ImageQualityService
    
    private lateinit var imageCaptureService: DefaultImageCaptureService
    private lateinit var testBitmap: Bitmap
    
    @Before
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        imageCaptureService = DefaultImageCaptureService(mockQualityService)
        
        // Create a test bitmap
        testBitmap = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        testBitmap.eraseColor(Color.GRAY)
    }
    
    @Test
    fun `loadImageFromPath should return Error for non-existent file`() {
        val nonExistentPath = "/path/to/nonexistent/image.jpg"
        
        val result = imageCaptureService.loadImageFromPath(nonExistentPath)
        
        assertTrue("Should return Error for non-existent file", result is ImageResult.Error)
        val errorMessage = (result as ImageResult.Error).message
        assertTrue("Error message should mention file not found", 
            errorMessage.contains("not found", ignoreCase = true))
    }
    
    @Test
    fun `validateImageQuality should delegate to quality service`() {
        val mockReport = ImageQualityReport(
            isAcceptable = true,
            qualityScore = 85.0,
            issues = emptyList(),
            metrics = mapOf("brightness" to 120.0),
            feedback = emptyList()
        )
        `when`(mockQualityService.validateImageQuality(testBitmap)).thenReturn(mockReport)
        
        val result = imageCaptureService.validateImageQuality(testBitmap)
        
        assertEquals("Should return quality service result", mockReport, result)
        verify(mockQualityService).validateImageQuality(testBitmap)
    }
    
    @Test
    fun `preprocessImage should delegate to quality service`() {
        val mockProcessedImage = ProcessedImage(
            bitmap = testBitmap,
            originalWidth = 300,
            originalHeight = 300,
            processedWidth = 224,
            processedHeight = 224,
            processingTimeMs = 50L
        )
        `when`(mockQualityService.preprocessForInference(testBitmap)).thenReturn(mockProcessedImage)
        
        val result = imageCaptureService.preprocessImage(testBitmap)
        
        assertEquals("Should return processed image from quality service", mockProcessedImage, result)
        verify(mockQualityService).preprocessForInference(testBitmap)
    }
    
    @Test
    fun `validateImageQuality should handle quality issues correctly`() {
        val issuesReport = ImageQualityReport(
            isAcceptable = false,
            qualityScore = 45.0,
            issues = listOf(QualityIssue.TOO_DARK, QualityIssue.TOO_BLURRY),
            metrics = mapOf("brightness" to 20.0, "blur_score" to 50.0),
            feedback = listOf("Image too dark", "Image too blurry")
        )
        `when`(mockQualityService.validateImageQuality(testBitmap)).thenReturn(issuesReport)
        
        val result = imageCaptureService.validateImageQuality(testBitmap)
        
        assertFalse("Should not be acceptable", result.isAcceptable)
        assertTrue("Should have quality issues", result.issues.isNotEmpty())
        assertTrue("Should have feedback", result.feedback.isNotEmpty())
        assertTrue("Quality score should be low", result.qualityScore < 50.0)
    }
    
    @Test
    fun `preprocessImage should handle processing metrics`() {
        val processedImage = ProcessedImage(
            bitmap = testBitmap,
            originalWidth = 800,
            originalHeight = 600,
            processedWidth = 224,
            processedHeight = 224,
            processingTimeMs = 120L
        )
        `when`(mockQualityService.preprocessForInference(any())).thenReturn(processedImage)
        
        val result = imageCaptureService.preprocessImage(testBitmap)
        
        assertEquals("Original width should be preserved", 800, result.originalWidth)
        assertEquals("Original height should be preserved", 600, result.originalHeight)
        assertEquals("Processed width should be 224", 224, result.processedWidth)
        assertEquals("Processed height should be 224", 224, result.processedHeight)
        assertTrue("Processing time should be recorded", result.processingTimeMs > 0)
    }
    
    @Test
    fun `service should handle null bitmap gracefully`() {
        // This test ensures the service doesn't crash with null inputs
        // In real Android, BitmapFactory.decodeFile can return null
        try {
            val result = imageCaptureService.loadImageFromPath("invalid_path")
            assertTrue("Should handle invalid path gracefully", result is ImageResult.Error)
        } catch (e: Exception) {
            fail("Service should not throw exceptions for invalid inputs")
        }
    }
    
    @Test
    fun `quality validation should be consistent`() {
        val report1 = ImageQualityReport(
            isAcceptable = true,
            qualityScore = 90.0,
            issues = emptyList(),
            metrics = mapOf("brightness" to 150.0),
            feedback = emptyList()
        )
        
        `when`(mockQualityService.validateImageQuality(testBitmap))
            .thenReturn(report1)
            .thenReturn(report1) // Same result for consistency
        
        val result1 = imageCaptureService.validateImageQuality(testBitmap)
        val result2 = imageCaptureService.validateImageQuality(testBitmap)
        
        assertEquals("Results should be consistent", result1, result2)
    }
}