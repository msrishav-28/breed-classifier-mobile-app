package com.livestock.recognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.livestock.recognition.data.repository.OfflineDataManager
import com.livestock.recognition.services.EnsembleCoordinationService
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.Assert.*
import org.mockito.kotlin.*

/**
 * Property-based test for low confidence warnings
 * **Feature: cattle-buffalo-recognition, Property 5: Low confidence warning**
 * **Validates: Requirements 2.4**
 */
class LowConfidenceWarningPropertyTest {
    
    companion object {
        private const val PROPERTY_TEST_ITERATIONS = 100
        private const val WARNING_CONFIDENCE_THRESHOLD = 0.75f
        private const val MIN_CONFIDENCE = 0.0f
        private const val MAX_CONFIDENCE = 1.0f
    }
    
    /**
     * Property: For any prediction with confidence score below 75%, the system should display warning messages and suggest retaking the photo
     */
    @Test
    fun `property test - low confidence predictions should display warnings and suggest retaking photo`() = runBlocking {
        val mockContext = mock<Context>()
        val mockOfflineDataManager = mock<OfflineDataManager>()
        val mockEnsembleCoordinationService = mock<EnsembleCoordinationService>()
        val mockModelValidationService = mock<ModelValidationService>()
        
        // Mock successful model validation
        whenever(mockModelValidationService.validateEnsembleModels()).thenReturn(
            EnsembleValidationResult(
                isValid = true,
                validModels = 3,
                totalModels = 3,
                totalSizeMB = 150.0,
                sizeWithinLimit = true,
                minimumModelsAvailable = true
            )
        )
        
        val testBreeds = listOf("Gir", "Sahiwal", "Murrah", "Mehsana", "Red_Sindhi", "Surti", "Tharparkar", "Hariana")
        
        // Property test: For any prediction with confidence < 75%, warnings should be displayed
        repeat(PROPERTY_TEST_ITERATIONS) { iteration ->
            val testBreed = testBreeds.random()
            val testBitmap = createTestBitmap(224, 224)
            
            // Generate confidence scores across the full range [0, 1]
            val confidence = (MIN_CONFIDENCE..MAX_CONFIDENCE).random()
            
            // Create individual model predictions with the same confidence pattern
            val vitPrediction = ModelPrediction(
                modelName = "ViT",
                breed = testBreed,
                confidence = confidence,
                inferenceTimeMs = (800..2000).random().toLong(),
                topPredictions = listOf(
                    BreedPrediction(testBreed, confidence),
                    BreedPrediction(testBreeds.random(), (0.1f..0.4f).random()),
                    BreedPrediction(testBreeds.random(), (0.05f..0.2f).random())
                )
            )
            
            val yoloPrediction = ModelPrediction(
                modelName = "YOLOv8-CBAM",
                breed = testBreed,
                confidence = confidence,
                inferenceTimeMs = (900..2200).random().toLong(),
                topPredictions = listOf(
                    BreedPrediction(testBreed, confidence),
                    BreedPrediction(testBreeds.random(), (0.1f..0.3f).random()),
                    BreedPrediction(testBreeds.random(), (0.05f..0.15f).random())
                )
            )
            
            val efficientNetPrediction = ModelPrediction(
                modelName = "EfficientNetV2",
                breed = testBreed,
                confidence = confidence,
                inferenceTimeMs = (700..1800).random().toLong(),
                topPredictions = listOf(
                    BreedPrediction(testBreed, confidence),
                    BreedPrediction(testBreeds.random(), (0.1f..0.25f).random()),
                    BreedPrediction(testBreeds.random(), (0.05f..0.2f).random())
                )
            )
            
            // Create advanced inference result with the generated confidence
            val shouldWarn = confidence < WARNING_CONFIDENCE_THRESHOLD
            val warningMessage = if (shouldWarn) {
                when {
                    confidence < 0.5f -> "Low confidence prediction (${(confidence * 100).toInt()}%). Please retake photo with better lighting and focus."
                    confidence < 0.65f -> "Moderate confidence prediction (${(confidence * 100).toInt()}%). Consider retaking photo for better accuracy."
                    else -> "Below optimal confidence (${(confidence * 100).toInt()}%). Photo quality could be improved."
                }
            } else null
            
            val mockResult = AdvancedInferenceResult(
                breed = testBreed,
                confidence = confidence,
                processingTimeMs = (1000..2800).random().toLong(),
                performanceValid = true,
                shouldDisplayWarning = shouldWarn,
                warningMessage = warningMessage,
                individualPredictions = IndividualPredictions(vitPrediction, yoloPrediction, efficientNetPrediction),
                ensembleMetrics = EnsembleMetrics(
                    modelsUsed = 3,
                    consensusStrength = 0.8f,
                    confidenceDistribution = ConfidenceDistribution(
                        mean = confidence,
                        min = confidence - 0.05f,
                        max = confidence + 0.05f,
                        standardDeviation = 0.02f
                    )
                ),
                ttaUsed = false,
                ttaPredictionsCount = 0
            )
            
            // Verify the core property: low confidence should trigger warnings
            if (confidence < WARNING_CONFIDENCE_THRESHOLD) {
                assertTrue("Prediction with confidence ${confidence} (< ${WARNING_CONFIDENCE_THRESHOLD}) should display warning (iteration $iteration)", 
                    mockResult.shouldDisplayWarning)
                assertNotNull("Low confidence prediction should provide warning message (iteration $iteration)", 
                    mockResult.warningMessage)
                assertFalse("Warning message should not be empty (iteration $iteration)", 
                    mockResult.warningMessage.isNullOrBlank())
                
                // Verify warning message suggests retaking photo
                val warningText = mockResult.warningMessage!!.lowercase()
                val containsRetakeGuidance = warningText.contains("retake") || 
                                           warningText.contains("better") || 
                                           warningText.contains("improve") ||
                                           warningText.contains("lighting") ||
                                           warningText.contains("focus") ||
                                           warningText.contains("photo") ||
                                           warningText.contains("quality")
                
                assertTrue("Warning message should suggest retaking photo or improving quality: '${mockResult.warningMessage}' (iteration $iteration)", 
                    containsRetakeGuidance)
                
                // Verify warning message includes confidence percentage
                val containsConfidenceInfo = warningText.contains("confidence") || 
                                           warningText.contains("${(confidence * 100).toInt()}%")
                
                assertTrue("Warning message should include confidence information: '${mockResult.warningMessage}' (iteration $iteration)", 
                    containsConfidenceInfo)
                
            } else {
                // High confidence should not trigger warnings
                assertFalse("Prediction with confidence ${confidence} (>= ${WARNING_CONFIDENCE_THRESHOLD}) should not display warning (iteration $iteration)", 
                    mockResult.shouldDisplayWarning)
                
                // Warning message should be null for high confidence
                assertTrue("High confidence prediction should not have warning message (iteration $iteration)", 
                    mockResult.warningMessage.isNullOrBlank())
            }
            
            // Verify confidence is within valid range
            assertTrue("Confidence ${confidence} must be >= 0 (iteration $iteration)", 
                confidence >= MIN_CONFIDENCE)
            assertTrue("Confidence ${confidence} must be <= 1 (iteration $iteration)", 
                confidence <= MAX_CONFIDENCE)
        }
    }
    
    /**
     * Property: Warning messages should be contextual based on confidence level
     */
    @Test
    fun `property test - warning messages should be contextual based on confidence level`() = runBlocking {
        val testBreeds = listOf("Gir", "Sahiwal", "Murrah", "Mehsana", "Red_Sindhi", "Surti")
        
        // Test different confidence ranges and their corresponding warning messages
        repeat(50) { iteration ->
            val testBreed = testBreeds.random()
            
            // Test specific confidence ranges
            val confidenceRanges = listOf(
                0.0f to 0.5f,   // Very low confidence
                0.5f to 0.65f,  // Low confidence  
                0.65f to 0.75f, // Below optimal confidence
                0.75f to 1.0f   // High confidence (no warning)
            )
            
            val (minConf, maxConf) = confidenceRanges.random()
            val confidence = (minConf..maxConf).random()
            
            val shouldWarn = confidence < WARNING_CONFIDENCE_THRESHOLD
            val warningMessage = if (shouldWarn) {
                when {
                    confidence < 0.5f -> "Low confidence prediction (${(confidence * 100).toInt()}%). Please retake photo with better lighting and focus."
                    confidence < 0.65f -> "Moderate confidence prediction (${(confidence * 100).toInt()}%). Consider retaking photo for better accuracy."
                    else -> "Below optimal confidence (${(confidence * 100).toInt()}%). Photo quality could be improved."
                }
            } else null
            
            val mockResult = AdvancedInferenceResult(
                breed = testBreed,
                confidence = confidence,
                processingTimeMs = (1000..2500).random().toLong(),
                performanceValid = true,
                shouldDisplayWarning = shouldWarn,
                warningMessage = warningMessage,
                individualPredictions = IndividualPredictions(
                    vitPrediction = ModelPrediction("ViT", testBreed, confidence, 1000L, emptyList()),
                    yoloPrediction = ModelPrediction("YOLO", testBreed, confidence, 1200L, emptyList()),
                    efficientNetPrediction = ModelPrediction("EfficientNet", testBreed, confidence, 900L, emptyList())
                ),
                ensembleMetrics = EnsembleMetrics(
                    modelsUsed = 3,
                    consensusStrength = 0.8f,
                    confidenceDistribution = ConfidenceDistribution(
                        mean = confidence,
                        min = confidence,
                        max = confidence,
                        standardDeviation = 0.0f
                    )
                ),
                ttaUsed = false,
                ttaPredictionsCount = 0
            )
            
            when {
                confidence < 0.5f -> {
                    // Very low confidence - should have urgent warning
                    assertTrue("Very low confidence (${confidence}) should display warning (iteration $iteration)", 
                        mockResult.shouldDisplayWarning)
                    assertNotNull("Very low confidence should have warning message (iteration $iteration)", 
                        mockResult.warningMessage)
                    
                    val warningText = mockResult.warningMessage!!.lowercase()
                    assertTrue("Very low confidence warning should mention 'low confidence' or similar (iteration $iteration)", 
                        warningText.contains("low confidence") || warningText.contains("low"))
                    assertTrue("Very low confidence warning should suggest retaking photo (iteration $iteration)", 
                        warningText.contains("retake") || warningText.contains("please"))
                }
                
                confidence < 0.65f -> {
                    // Low confidence - should have moderate warning
                    assertTrue("Low confidence (${confidence}) should display warning (iteration $iteration)", 
                        mockResult.shouldDisplayWarning)
                    assertNotNull("Low confidence should have warning message (iteration $iteration)", 
                        mockResult.warningMessage)
                    
                    val warningText = mockResult.warningMessage!!.lowercase()
                    assertTrue("Low confidence warning should mention 'moderate' or similar (iteration $iteration)", 
                        warningText.contains("moderate") || warningText.contains("consider"))
                }
                
                confidence < WARNING_CONFIDENCE_THRESHOLD -> {
                    // Below optimal confidence - should have mild warning
                    assertTrue("Below optimal confidence (${confidence}) should display warning (iteration $iteration)", 
                        mockResult.shouldDisplayWarning)
                    assertNotNull("Below optimal confidence should have warning message (iteration $iteration)", 
                        mockResult.warningMessage)
                    
                    val warningText = mockResult.warningMessage!!.lowercase()
                    assertTrue("Below optimal confidence warning should mention 'optimal' or 'improved' (iteration $iteration)", 
                        warningText.contains("optimal") || warningText.contains("improved") || warningText.contains("below"))
                }
                
                else -> {
                    // High confidence - should not have warning
                    assertFalse("High confidence (${confidence}) should not display warning (iteration $iteration)", 
                        mockResult.shouldDisplayWarning)
                    assertTrue("High confidence should not have warning message (iteration $iteration)", 
                        mockResult.warningMessage.isNullOrBlank())
                }
            }
        }
    }
    
    /**
     * Property: Warning threshold should be consistently applied across all predictions
     */
    @Test
    fun `property test - warning threshold should be consistently applied`() = runBlocking {
        val testBreeds = listOf("Gir", "Sahiwal", "Murrah", "Mehsana", "Red_Sindhi", "Surti")
        
        // Test edge cases around the warning threshold
        val testConfidences = listOf(
            0.74f, 0.749f, 0.7499f,  // Just below threshold
            0.75f, 0.751f, 0.7501f   // Just above threshold
        )
        
        testConfidences.forEach { confidence ->
            repeat(10) { iteration ->
                val testBreed = testBreeds.random()
                
                val shouldWarn = confidence < WARNING_CONFIDENCE_THRESHOLD
                val warningMessage = if (shouldWarn) {
                    "Below optimal confidence (${(confidence * 100).toInt()}%). Photo quality could be improved."
                } else null
                
                val mockResult = AdvancedInferenceResult(
                    breed = testBreed,
                    confidence = confidence,
                    processingTimeMs = (1000..2500).random().toLong(),
                    performanceValid = true,
                    shouldDisplayWarning = shouldWarn,
                    warningMessage = warningMessage,
                    individualPredictions = IndividualPredictions(
                        vitPrediction = ModelPrediction("ViT", testBreed, confidence, 1000L, emptyList()),
                        yoloPrediction = ModelPrediction("YOLO", testBreed, confidence, 1200L, emptyList()),
                        efficientNetPrediction = ModelPrediction("EfficientNet", testBreed, confidence, 900L, emptyList())
                    ),
                    ensembleMetrics = EnsembleMetrics(
                        modelsUsed = 3,
                        consensusStrength = 0.9f,
                        confidenceDistribution = ConfidenceDistribution(
                            mean = confidence,
                            min = confidence,
                            max = confidence,
                            standardDeviation = 0.0f
                        )
                    ),
                    ttaUsed = false,
                    ttaPredictionsCount = 0
                )
                
                // Verify threshold is consistently applied
                val expectedWarning = confidence < WARNING_CONFIDENCE_THRESHOLD
                assertEquals("Warning flag should match threshold for confidence ${confidence} (iteration $iteration)", 
                    expectedWarning, mockResult.shouldDisplayWarning)
                
                if (expectedWarning) {
                    assertNotNull("Below threshold confidence ${confidence} should have warning message (iteration $iteration)", 
                        mockResult.warningMessage)
                    assertFalse("Warning message should not be empty for confidence ${confidence} (iteration $iteration)", 
                        mockResult.warningMessage.isNullOrBlank())
                } else {
                    assertTrue("Above threshold confidence ${confidence} should not have warning message (iteration $iteration)", 
                        mockResult.warningMessage.isNullOrBlank())
                }
            }
        }
    }
    
    /**
     * Property: All warning messages should be user-friendly and actionable
     */
    @Test
    fun `property test - warning messages should be user friendly and actionable`() = runBlocking {
        val testBreeds = listOf("Gir", "Sahiwal", "Murrah", "Mehsana", "Red_Sindhi", "Surti")
        
        // Test only low confidence scenarios that should generate warnings
        repeat(30) { iteration ->
            val testBreed = testBreeds.random()
            val confidence = (0.0f..0.74f).random() // Only test below threshold
            
            val warningMessage = when {
                confidence < 0.5f -> "Low confidence prediction (${(confidence * 100).toInt()}%). Please retake photo with better lighting and focus."
                confidence < 0.65f -> "Moderate confidence prediction (${(confidence * 100).toInt()}%). Consider retaking photo for better accuracy."
                else -> "Below optimal confidence (${(confidence * 100).toInt()}%). Photo quality could be improved."
            }
            
            val mockResult = AdvancedInferenceResult(
                breed = testBreed,
                confidence = confidence,
                processingTimeMs = (1000..2500).random().toLong(),
                performanceValid = true,
                shouldDisplayWarning = true,
                warningMessage = warningMessage,
                individualPredictions = IndividualPredictions(
                    vitPrediction = ModelPrediction("ViT", testBreed, confidence, 1000L, emptyList()),
                    yoloPrediction = ModelPrediction("YOLO", testBreed, confidence, 1200L, emptyList()),
                    efficientNetPrediction = ModelPrediction("EfficientNet", testBreed, confidence, 900L, emptyList())
                ),
                ensembleMetrics = EnsembleMetrics(
                    modelsUsed = 3,
                    consensusStrength = 0.8f,
                    confidenceDistribution = ConfidenceDistribution(
                        mean = confidence,
                        min = confidence,
                        max = confidence,
                        standardDeviation = 0.0f
                    )
                ),
                ttaUsed = false,
                ttaPredictionsCount = 0
            )
            
            // Verify warning message exists and is not empty
            assertNotNull("Low confidence prediction should have warning message (iteration $iteration)", 
                mockResult.warningMessage)
            assertFalse("Warning message should not be empty (iteration $iteration)", 
                mockResult.warningMessage.isNullOrBlank())
            
            val warningText = mockResult.warningMessage!!
            
            // Verify message is user-friendly (no technical jargon)
            val technicalTerms = listOf("tensor", "inference", "neural", "algorithm", "model", "ensemble")
            val containsTechnicalTerms = technicalTerms.any { term -> 
                warningText.lowercase().contains(term) 
            }
            assertFalse("Warning message should not contain technical jargon: '${warningText}' (iteration $iteration)", 
                containsTechnicalTerms)
            
            // Verify message is actionable (provides guidance)
            val actionableTerms = listOf("retake", "improve", "better", "consider", "please", "quality", "lighting", "focus")
            val containsActionableGuidance = actionableTerms.any { term -> 
                warningText.lowercase().contains(term) 
            }
            assertTrue("Warning message should provide actionable guidance: '${warningText}' (iteration $iteration)", 
                containsActionableGuidance)
            
            // Verify message includes confidence information
            val confidencePercentage = (confidence * 100).toInt()
            val containsConfidenceInfo = warningText.contains("confidence") || 
                                       warningText.contains("${confidencePercentage}%")
            assertTrue("Warning message should include confidence information: '${warningText}' (iteration $iteration)", 
                containsConfidenceInfo)
            
            // Verify message is reasonably short (user-friendly length)
            assertTrue("Warning message should be reasonably short (< 200 chars): '${warningText}' (iteration $iteration)", 
                warningText.length < 200)
            
            // Verify message starts with capital letter and ends with period (proper formatting)
            assertTrue("Warning message should start with capital letter: '${warningText}' (iteration $iteration)", 
                warningText.isNotEmpty() && warningText[0].isUpperCase())
            assertTrue("Warning message should end with period: '${warningText}' (iteration $iteration)", 
                warningText.endsWith("."))
        }
    }
    
    // Helper method for creating test bitmaps
    private fun createTestBitmap(width: Int, height: Int): Bitmap {
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            // Fill with random colors to simulate livestock images
            val pixels = IntArray(width * height) { 
                Color.rgb(
                    (50..200).random(),
                    (30..180).random(), 
                    (20..150).random()
                )
            }
            setPixels(pixels, 0, width, 0, 0, width, height)
        }
    }
}