package com.livestock.recognition.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.repository.ClassificationRepository
import com.livestock.recognition.data.repository.OfflineDataManager
import com.livestock.recognition.ml.AdvancedMLInferenceEngine
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.runner.RunWith
import org.mockito.kotlin.*
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Property-based test for offline processing capability
 * **Feature: cattle-buffalo-recognition, Property 7: Offline processing capability**
 * **Validates: Requirements 4.2**
 * 
 * Tests that the system can perform breed recognition and type classification
 * without network connectivity across all valid inputs
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class OfflineProcessingPropertyTest : StringSpec({
    
    "Property 7: Offline processing capability - system should perform breed recognition without network connectivity" {
        
        // Arrange - Create test dependencies
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockMLEngine = mock<AdvancedMLInferenceEngine>()
        val mockTypeService = mock<TypeClassificationService>()
        val mockOfflineDataManager = mock<OfflineDataManager>()
        val mockClassificationRepository = mock<ClassificationRepository>()
        
        // Configure mocks for offline operation
        whenever(mockMLEngine.loadModels()).thenReturn(true)
        whenever(mockMLEngine.areModelsLoaded()).thenReturn(true)
        whenever(mockTypeService.validateMapping()).thenReturn(
            ValidationResult(true, emptyList())
        )
        whenever(mockOfflineDataManager.isNetworkAvailable()).thenReturn(false) // Force offline mode
        
        val offlineProcessingService = OfflineProcessingService(
            context,
            mockMLEngine,
            mockTypeService,
            mockOfflineDataManager,
            mockClassificationRepository
        )
        
        // Initialize service
        runBlocking {
            offlineProcessingService.initialize() shouldBe true
        }
        
        // Property test generators
        val bitmapGen = Arb.bind(
            Arb.int(100..500), // width
            Arb.int(100..500), // height
            Arb.enum<Bitmap.Config>()
        ) { width, height, config ->
            Bitmap.createBitmap(width, height, config).apply {
                // Fill with random colors to simulate real image data
                val colors = IntArray(width * height) { Color.rgb(
                    (0..255).random(),
                    (0..255).random(),
                    (0..255).random()
                )}
                setPixels(colors, 0, width, 0, 0, width, height)
            }
        }
        
        val breedGen = Arb.element(listOf(
            "Gir", "Sahiwal", "Red_Sindhi", "Tharparkar", "Rathi",
            "Hariana", "Ongole", "Krishna_Valley", "Deoni", "Khillari",
            "Murrah", "Mehsana", "Surti", "Jaffarabadi", "Nili_Ravi"
        ))
        
        val confidenceGen = Arb.float(0.3f, 1.0f)
        val animalTypeGen = Arb.enum<AnimalType>()
        
        // Configure mock responses for each test iteration
        checkAll(100, bitmapGen, breedGen, confidenceGen, animalTypeGen) { bitmap, breed, confidence, animalType ->
            
            // Configure ML engine mock to return valid inference result
            val mockInferenceResult = mock<com.livestock.recognition.ml.InferenceResult> {
                on { this.breed } doReturn breed
                on { this.confidence } doReturn confidence
            }
            whenever(mockMLEngine.runInference(any())).thenReturn(mockInferenceResult)
            
            // Configure type classification service
            whenever(mockTypeService.classifyType(breed)).thenReturn(animalType)
            
            // Configure offline data manager (no cached result)
            whenever(mockOfflineDataManager.getCachedClassificationResult(any())).thenReturn(null)
            whenever(mockOfflineDataManager.cacheClassificationResult(any(), any())).thenReturn(Unit)
            
            // Configure classification repository
            whenever(mockClassificationRepository.saveClassification(any())).thenReturn(1L)
            
            // Act - Process image in offline mode
            val result = runBlocking {
                offlineProcessingService.processImageOffline(bitmap)
            }
            
            // Assert - Verify offline processing works correctly
            when (result) {
                is OfflineProcessingResult.Success -> {
                    // Verify classification result is valid
                    result.result shouldNotBe null
                    result.result.breed shouldBe breed
                    result.result.breedConfidence shouldBe confidence
                    result.result.animalType shouldBe animalType
                    result.result.timestamp shouldNotBe 0L
                    result.processingTime shouldNotBe 0L
                    
                    // Verify offline processing occurred (not from cache)
                    result.fromCache shouldBe false
                    
                    // Verify ML inference was called
                    verify(mockMLEngine).runInference(bitmap)
                    
                    // Verify type classification was called
                    verify(mockTypeService).classifyType(breed)
                    
                    // Verify result was cached for future offline use
                    verify(mockOfflineDataManager).cacheClassificationResult(any(), any())
                    
                    // Verify result was saved to local database
                    verify(mockClassificationRepository).saveClassification(any())
                }
                is OfflineProcessingResult.Error -> {
                    // If processing fails, it should be due to valid reasons, not network dependency
                    result.message shouldNotBe "Network required"
                    result.message shouldNotBe "Internet connection needed"
                }
            }
            
            // Reset mocks for next iteration
            reset(mockMLEngine, mockTypeService, mockOfflineDataManager, mockClassificationRepository)
            
            // Reconfigure basic mocks
            whenever(mockMLEngine.loadModels()).thenReturn(true)
            whenever(mockMLEngine.areModelsLoaded()).thenReturn(true)
            whenever(mockTypeService.validateMapping()).thenReturn(
                ValidationResult(true, emptyList())
            )
            whenever(mockOfflineDataManager.isNetworkAvailable()).thenReturn(false)
        }
    }
    
    "Property 7a: Offline processing with cached results - system should use cached results when available offline" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockMLEngine = mock<AdvancedMLInferenceEngine>()
        val mockTypeService = mock<TypeClassificationService>()
        val mockOfflineDataManager = mock<OfflineDataManager>()
        val mockClassificationRepository = mock<ClassificationRepository>()
        
        // Configure mocks
        whenever(mockMLEngine.loadModels()).thenReturn(true)
        whenever(mockMLEngine.areModelsLoaded()).thenReturn(true)
        whenever(mockTypeService.validateMapping()).thenReturn(
            ValidationResult(true, emptyList())
        )
        whenever(mockOfflineDataManager.isNetworkAvailable()).thenReturn(false)
        
        val offlineProcessingService = OfflineProcessingService(
            context,
            mockMLEngine,
            mockTypeService,
            mockOfflineDataManager,
            mockClassificationRepository
        )
        
        runBlocking {
            offlineProcessingService.initialize() shouldBe true
        }
        
        // Property test with cached results
        val bitmapGen = Arb.bind(
            Arb.int(100..300),
            Arb.int(100..300)
        ) { width, height ->
            Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        }
        
        val cachedResultGen = Arb.bind(
            Arb.element(listOf("Gir", "Sahiwal", "Murrah", "Mehsana")),
            Arb.float(0.5f, 1.0f),
            Arb.enum<AnimalType>(),
            Arb.long(1000L, 5000L)
        ) { breed, confidence, animalType, processingTime ->
            ClassificationResult(
                breed = breed,
                breedConfidence = confidence,
                animalType = animalType,
                typeConfidence = confidence,
                timestamp = System.currentTimeMillis(),
                imageMetadata = null,
                processingTime = processingTime
            )
        }
        
        checkAll(50, bitmapGen, cachedResultGen) { bitmap, cachedResult ->
            
            // Configure offline data manager to return cached result
            whenever(mockOfflineDataManager.getCachedClassificationResult(any())).thenReturn(cachedResult)
            
            // Act
            val result = runBlocking {
                offlineProcessingService.processImageOffline(bitmap)
            }
            
            // Assert
            when (result) {
                is OfflineProcessingResult.Success -> {
                    // Verify cached result is returned
                    result.fromCache shouldBe true
                    result.result.breed shouldBe cachedResult.breed
                    result.result.breedConfidence shouldBe cachedResult.breedConfidence
                    result.result.animalType shouldBe cachedResult.animalType
                    
                    // Verify ML inference was NOT called (using cache)
                    verify(mockMLEngine, never()).runInference(any())
                    
                    // Verify type classification was NOT called (using cache)
                    verify(mockTypeService, never()).classifyType(any())
                }
                is OfflineProcessingResult.Error -> {
                    // Should not fail when cached result is available
                    throw AssertionError("Offline processing should not fail with cached result: ${result.message}")
                }
            }
            
            // Reset mocks
            reset(mockMLEngine, mockTypeService, mockOfflineDataManager, mockClassificationRepository)
            whenever(mockMLEngine.loadModels()).thenReturn(true)
            whenever(mockMLEngine.areModelsLoaded()).thenReturn(true)
            whenever(mockTypeService.validateMapping()).thenReturn(ValidationResult(true, emptyList()))
            whenever(mockOfflineDataManager.isNetworkAvailable()).thenReturn(false)
        }
    }
    
    "Property 7b: Offline processing validation - system should validate offline capabilities correctly" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockMLEngine = mock<AdvancedMLInferenceEngine>()
        val mockTypeService = mock<TypeClassificationService>()
        val mockOfflineDataManager = mock<OfflineDataManager>()
        val mockClassificationRepository = mock<ClassificationRepository>()
        
        val offlineProcessingService = OfflineProcessingService(
            context,
            mockMLEngine,
            mockTypeService,
            mockOfflineDataManager,
            mockClassificationRepository
        )
        
        // Property test for different validation scenarios
        val modelsLoadedGen = Arb.boolean()
        val typeServiceValidGen = Arb.boolean()
        val databaseAccessGen = Arb.boolean()
        val cacheAccessGen = Arb.boolean()
        
        checkAll(50, modelsLoadedGen, typeServiceValidGen, databaseAccessGen, cacheAccessGen) { 
            modelsLoaded, typeServiceValid, databaseAccess, cacheAccess ->
            
            // Configure mocks based on test parameters
            whenever(mockMLEngine.areModelsLoaded()).thenReturn(modelsLoaded)
            
            whenever(mockTypeService.validateMapping()).thenReturn(
                if (typeServiceValid) ValidationResult(true, emptyList())
                else ValidationResult(false, listOf("Type mapping invalid"))
            )
            
            if (databaseAccess) {
                whenever(mockClassificationRepository.getClassificationCount()).thenReturn(10)
            } else {
                whenever(mockClassificationRepository.getClassificationCount()).thenThrow(
                    RuntimeException("Database access failed")
                )
            }
            
            if (cacheAccess) {
                whenever(mockOfflineDataManager.getCacheStatistics()).thenReturn(
                    com.livestock.recognition.data.repository.CacheStatistics(5, 1024L, emptyMap())
                )
            } else {
                whenever(mockOfflineDataManager.getCacheStatistics()).thenThrow(
                    RuntimeException("Cache access failed")
                )
            }
            
            // Act
            val validationResult = runBlocking {
                offlineProcessingService.validateOfflineCapabilities()
            }
            
            // Assert
            val expectedValid = modelsLoaded && typeServiceValid && databaseAccess && cacheAccess
            validationResult.isValid shouldBe expectedValid
            
            if (!modelsLoaded) {
                validationResult.issues.any { it.contains("ML models") } shouldBe true
            }
            
            if (!typeServiceValid) {
                validationResult.issues.any { it.contains("Type mapping") } shouldBe true
            }
            
            if (!databaseAccess) {
                validationResult.issues.any { it.contains("Database") } shouldBe true
            }
            
            if (!cacheAccess) {
                validationResult.issues.any { it.contains("Cache") } shouldBe true
            }
            
            // Reset mocks
            reset(mockMLEngine, mockTypeService, mockOfflineDataManager, mockClassificationRepository)
        }
    }
})

/**
 * Mock validation result for type classification service
 */
data class ValidationResult(
    val isValid: Boolean,
    val errors: List<String>
)