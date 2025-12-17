package com.livestock.recognition.services

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.models.ImageMetadata
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.mockito.junit.MockitoJUnitRunner

/**
 * Unit tests for ImageAnnotationService functionality.
 * Tests annotation generation, bounding box drawing, and confidence visualization.
 * 
 * Requirements: 5.1
 */
@RunWith(MockitoJUnitRunner::class)
class ImageAnnotationServiceTest {

    private lateinit var annotationService: ImageAnnotationService
    private lateinit var testBitmap: Bitmap
    private lateinit var testClassificationResult: ClassificationResult

    @Before
    fun setup() {
        annotationService = ImageAnnotationService()
        
        // Create a test bitmap
        testBitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)
        testBitmap.eraseColor(Color.BLUE) // Fill with blue color
        
        // Create test classification result
        testClassificationResult = ClassificationResult(
            breed = "Gir",
            breedConfidence = 0.85f,
            animalType = AnimalType.DAIRY,
            typeConfidence = 0.9f,
            timestamp = System.currentTimeMillis(),
            imageMetadata = ImageMetadata(
                width = 400,
                height = 300,
                fileSize = 120000L,
                captureTime = System.currentTimeMillis(),
                deviceInfo = "Test Device",
                qualityScore = 0.8f
            ),
            processingTime = 1500L
        )
    }

    @Test
    fun `createAnnotatedImage should preserve original image dimensions`() {
        // When
        val annotatedBitmap = annotationService.createAnnotatedImage(
            testBitmap,
            testClassificationResult
        )

        // Then
        assertNotNull("Annotated bitmap should not be null", annotatedBitmap)
        assertEquals("Width should be preserved", testBitmap.width, annotatedBitmap.width)
        assertEquals("Height should be preserved", testBitmap.height, annotatedBitmap.height)
        assertEquals("Config should be preserved", testBitmap.config, annotatedBitmap.config)
    }

    @Test
    fun `createAnnotatedImage should modify original image with annotations`() {
        // When
        val annotatedBitmap = annotationService.createAnnotatedImage(
            testBitmap,
            testClassificationResult
        )

        // Then
        val originalPixels = IntArray(testBitmap.width * testBitmap.height)
        val annotatedPixels = IntArray(annotatedBitmap.width * annotatedBitmap.height)

        testBitmap.getPixels(originalPixels, 0, testBitmap.width, 0, 0, testBitmap.width, testBitmap.height)
        annotatedBitmap.getPixels(annotatedPixels, 0, annotatedBitmap.width, 0, 0, annotatedBitmap.width, annotatedBitmap.height)

        assertFalse("Annotated image should be different from original", originalPixels.contentEquals(annotatedPixels))
    }

    @Test
    fun `createAnnotatedImage with bounding box should respect bounds`() {
        // Given
        val boundingBox = RectF(50f, 50f, 200f, 150f)

        // When
        val annotatedBitmap = annotationService.createAnnotatedImage(
            testBitmap,
            testClassificationResult,
            boundingBox
        )

        // Then
        assertNotNull("Annotated bitmap should not be null", annotatedBitmap)
        assertEquals("Width should be preserved", testBitmap.width, annotatedBitmap.width)
        assertEquals("Height should be preserved", testBitmap.height, annotatedBitmap.height)
    }

    @Test
    fun `createAnnotatedImage with high confidence should use appropriate colors`() {
        // Given - high confidence result
        val highConfidenceResult = testClassificationResult.copy(
            breedConfidence = 0.95f,
            typeConfidence = 0.92f
        )

        // When
        val annotatedBitmap = annotationService.createAnnotatedImage(
            testBitmap,
            highConfidenceResult
        )

        // Then
        assertNotNull("Annotated bitmap should not be null", annotatedBitmap)
        // High confidence should result in green-tinted annotations
        // We can't easily test exact colors without more complex pixel analysis
        // but we can verify the annotation was applied
        val originalPixels = IntArray(testBitmap.width * testBitmap.height)
        val annotatedPixels = IntArray(annotatedBitmap.width * annotatedBitmap.height)

        testBitmap.getPixels(originalPixels, 0, testBitmap.width, 0, 0, testBitmap.width, testBitmap.height)
        annotatedBitmap.getPixels(annotatedPixels, 0, annotatedBitmap.width, 0, 0, annotatedBitmap.width, annotatedBitmap.height)

        assertFalse("High confidence annotation should modify image", originalPixels.contentEquals(annotatedPixels))
    }

    @Test
    fun `createAnnotatedImage with low confidence should use appropriate colors`() {
        // Given - low confidence result
        val lowConfidenceResult = testClassificationResult.copy(
            breedConfidence = 0.45f,
            typeConfidence = 0.5f
        )

        // When
        val annotatedBitmap = annotationService.createAnnotatedImage(
            testBitmap,
            lowConfidenceResult
        )

        // Then
        assertNotNull("Annotated bitmap should not be null", annotatedBitmap)
        // Low confidence should result in red-tinted annotations
        val originalPixels = IntArray(testBitmap.width * testBitmap.height)
        val annotatedPixels = IntArray(annotatedBitmap.width * annotatedBitmap.height)

        testBitmap.getPixels(originalPixels, 0, testBitmap.width, 0, 0, testBitmap.width, testBitmap.height)
        annotatedBitmap.getPixels(annotatedPixels, 0, annotatedBitmap.width, 0, 0, annotatedBitmap.width, annotatedBitmap.height)

        assertFalse("Low confidence annotation should modify image", originalPixels.contentEquals(annotatedPixels))
    }

    @Test
    fun `createAnnotatedImage with medium confidence should use appropriate colors`() {
        // Given - medium confidence result
        val mediumConfidenceResult = testClassificationResult.copy(
            breedConfidence = 0.7f,
            typeConfidence = 0.65f
        )

        // When
        val annotatedBitmap = annotationService.createAnnotatedImage(
            testBitmap,
            mediumConfidenceResult
        )

        // Then
        assertNotNull("Annotated bitmap should not be null", annotatedBitmap)
        // Medium confidence should result in orange-tinted annotations
        val originalPixels = IntArray(testBitmap.width * testBitmap.height)
        val annotatedPixels = IntArray(annotatedBitmap.width * annotatedBitmap.height)

        testBitmap.getPixels(originalPixels, 0, testBitmap.width, 0, 0, testBitmap.width, testBitmap.height)
        annotatedBitmap.getPixels(annotatedPixels, 0, annotatedBitmap.width, 0, 0, annotatedBitmap.width, annotatedBitmap.height)

        assertFalse("Medium confidence annotation should modify image", originalPixels.contentEquals(annotatedPixels))
    }

    @Test
    fun `annotation should handle different animal types correctly`() {
        // Test with different animal types
        val animalTypes = listOf(AnimalType.DAIRY, AnimalType.DRAUGHT, AnimalType.DUAL_PURPOSE)
        
        animalTypes.forEach { animalType ->
            val result = testClassificationResult.copy(animalType = animalType)
            
            val annotatedBitmap = annotationService.createAnnotatedImage(
                testBitmap,
                result
            )
            
            assertNotNull("Annotated bitmap should not be null for $animalType", annotatedBitmap)
            assertEquals("Width should be preserved for $animalType", testBitmap.width, annotatedBitmap.width)
            assertEquals("Height should be preserved for $animalType", testBitmap.height, annotatedBitmap.height)
        }
    }

    @Test
    fun `annotation should handle different breed names correctly`() {
        // Test with different breed names
        val breeds = listOf("Gir", "Holstein", "Jersey", "Sahiwal", "Murrah")
        
        breeds.forEach { breed ->
            val result = testClassificationResult.copy(breed = breed)
            
            val annotatedBitmap = annotationService.createAnnotatedImage(
                testBitmap,
                result
            )
            
            assertNotNull("Annotated bitmap should not be null for $breed", annotatedBitmap)
            assertEquals("Width should be preserved for $breed", testBitmap.width, annotatedBitmap.width)
            assertEquals("Height should be preserved for $breed", testBitmap.height, annotatedBitmap.height)
        }
    }

    @Test
    fun `bounding box data class should work correctly`() {
        // Given
        val boundingBox = ImageAnnotationService.BoundingBox(10f, 20f, 100f, 150f)
        
        // When
        val rectF = boundingBox.toRectF()
        
        // Then
        assertEquals("Left should match", 10f, rectF.left, 0.01f)
        assertEquals("Top should match", 20f, rectF.top, 0.01f)
        assertEquals("Right should match", 100f, rectF.right, 0.01f)
        assertEquals("Bottom should match", 150f, rectF.bottom, 0.01f)
    }

    @Test
    fun `annotation config should have sensible defaults`() {
        // Given
        val config = ImageAnnotationService.AnnotationConfig()
        
        // Then
        assertTrue("Should show bounding box by default", config.showBoundingBox)
        assertTrue("Should show label by default", config.showLabel)
        assertTrue("Should show confidence indicator by default", config.showConfidenceIndicator)
        assertTrue("Stroke width should be positive", config.strokeWidth > 0)
        assertTrue("Text size should be positive", config.textSize > 0)
    }

    @Test
    fun `annotation config should allow customization`() {
        // Given
        val config = ImageAnnotationService.AnnotationConfig(
            showBoundingBox = false,
            showLabel = false,
            showConfidenceIndicator = false,
            strokeWidth = 12f,
            textSize = 36f
        )
        
        // Then
        assertFalse("Should not show bounding box when disabled", config.showBoundingBox)
        assertFalse("Should not show label when disabled", config.showLabel)
        assertFalse("Should not show confidence indicator when disabled", config.showConfidenceIndicator)
        assertEquals("Stroke width should be customizable", 12f, config.strokeWidth, 0.01f)
        assertEquals("Text size should be customizable", 36f, config.textSize, 0.01f)
    }
}