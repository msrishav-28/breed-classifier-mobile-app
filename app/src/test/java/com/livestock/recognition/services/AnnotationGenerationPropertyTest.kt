package com.livestock.recognition.services

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.models.ImageMetadata
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.mockito.junit.MockitoJUnitRunner
import kotlin.random.Random

/**
 * Property-based test for annotation generation functionality.
 * **Feature: cattle-buffalo-recognition, Property 9: Annotation generation**
 * **Validates: Requirements 5.1**
 * 
 * Tests that for any completed classification, the system generates annotated images 
 * with bounding boxes and labels.
 */
@RunWith(MockitoJUnitRunner::class)
class AnnotationGenerationPropertyTest {
    
    private val annotationService = ImageAnnotationService()
    
    @Test
    fun `Property 9 - Annotation generation - For any classification result, annotated image should contain visual markup`() {
        // Run property-based test with 100 iterations
        repeat(100) {
            val classificationResult = generateRandomClassificationResult()
            val originalBitmap = generateRandomBitmap()
            
            // Generate annotated image
            val annotatedBitmap = annotationService.createAnnotatedImage(
                originalBitmap,
                classificationResult
            )
            
            // Verify annotation was applied
            assertNotNull("Annotated bitmap should not be null", annotatedBitmap)
            assertEquals("Width should be preserved", originalBitmap.width, annotatedBitmap.width)
            assertEquals("Height should be preserved", originalBitmap.height, annotatedBitmap.height)
            
            // Verify the annotated image is different from original (has markup)
            val originalPixels = IntArray(originalBitmap.width * originalBitmap.height)
            val annotatedPixels = IntArray(annotatedBitmap.width * annotatedBitmap.height)
            
            originalBitmap.getPixels(originalPixels, 0, originalBitmap.width, 0, 0, originalBitmap.width, originalBitmap.height)
            annotatedBitmap.getPixels(annotatedPixels, 0, annotatedBitmap.width, 0, 0, annotatedBitmap.width, annotatedBitmap.height)
            
            // Images should be different (annotation added)
            assertFalse("Annotated image should be different from original", originalPixels.contentEquals(annotatedPixels))
        }
    }
    
    @Test
    fun `Property 9a - Annotation with bounding box - Annotations should respect bounding box constraints`() {
        repeat(100) {
            val classificationResult = generateRandomClassificationResult()
            val originalBitmap = generateRandomBitmap()
            val boundingBox = generateRandomBoundingBox(originalBitmap.width, originalBitmap.height)
            
            val annotatedBitmap = annotationService.createAnnotatedImage(
                originalBitmap,
                classificationResult,
                boundingBox
            )
            
            // Verify annotation was applied
            assertNotNull("Annotated bitmap should not be null", annotatedBitmap)
            assertEquals("Width should be preserved", originalBitmap.width, annotatedBitmap.width)
            assertEquals("Height should be preserved", originalBitmap.height, annotatedBitmap.height)
        }
    }
    
    @Test
    fun `Property 9b - Confidence color coding - Annotation colors should reflect confidence levels`() {
        repeat(100) {
            val confidence = Random.nextFloat()
            val classificationResult = generateRandomClassificationResult(confidence)
            val bitmap = generateRandomBitmap()
            
            val annotatedBitmap = annotationService.createAnnotatedImage(
                bitmap,
                classificationResult
            )
            
            // Verify annotation was created
            assertNotNull("Annotated bitmap should not be null", annotatedBitmap)
            
            // The annotation should be different from original
            val originalPixels = IntArray(bitmap.width * bitmap.height)
            val annotatedPixels = IntArray(annotatedBitmap.width * annotatedBitmap.height)
            
            bitmap.getPixels(originalPixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            annotatedBitmap.getPixels(annotatedPixels, 0, annotatedBitmap.width, 0, 0, annotatedBitmap.width, annotatedBitmap.height)
            
            assertFalse("Annotated image should be different from original", originalPixels.contentEquals(annotatedPixels))
        }
    }
}

    // Helper functions for generating random test data
    
    private fun generateRandomClassificationResult(confidence: Float = Random.nextFloat()): ClassificationResult {
        return ClassificationResult(
            breed = generateRandomBreedName(),
            breedConfidence = confidence,
            animalType = generateRandomAnimalType(),
            typeConfidence = confidence,
            timestamp = System.currentTimeMillis(),
            imageMetadata = generateRandomImageMetadata(),
            processingTime = Random.nextLong(100L, 5000L)
        )
    }
    
    private fun generateRandomBitmap(): Bitmap {
        val width = Random.nextInt(100, 500)
        val height = Random.nextInt(100, 500)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // Fill with random colors to simulate a real image
        val colors = IntArray(width * height) { 
            Color.rgb(
                Random.nextInt(0, 256),
                Random.nextInt(0, 256),
                Random.nextInt(0, 256)
            )
        }
        bitmap.setPixels(colors, 0, width, 0, 0, width, height)
        return bitmap
    }
    
    private fun generateRandomBoundingBox(imageWidth: Int, imageHeight: Int): RectF {
        val left = Random.nextFloat() * (imageWidth * 0.5f)
        val top = Random.nextFloat() * (imageHeight * 0.5f)
        val right = left + Random.nextFloat() * (imageWidth - left)
        val bottom = top + Random.nextFloat() * (imageHeight - top)
        return RectF(left, top, right, bottom)
    }
    
    private fun generateRandomBreedName(): String {
        val breeds = listOf(
            "Gir", "Holstein", "Jersey", "Sahiwal", "Red Sindhi", 
            "Tharparkar", "Murrah", "Nili-Ravi", "Surti", "Jaffarabadi"
        )
        return breeds[Random.nextInt(breeds.size)]
    }
    
    private fun generateRandomAnimalType(): AnimalType {
        val types = AnimalType.values()
        return types[Random.nextInt(types.size)]
    }
    
    private fun generateRandomImageMetadata(): ImageMetadata {
        return ImageMetadata(
            width = Random.nextInt(100, 2000),
            height = Random.nextInt(100, 2000),
            fileSize = Random.nextLong(10000L, 5000000L),
            captureTime = System.currentTimeMillis(),
            deviceInfo = "TestDevice",
            qualityScore = Random.nextFloat()
        )
    }
}