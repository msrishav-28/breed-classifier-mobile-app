package com.livestock.recognition.services

import android.content.Context
import android.content.res.AssetManager
import com.livestock.recognition.data.models.AnimalType
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations
import java.io.ByteArrayInputStream

/**
 * Property-based tests for type classification completeness.
 * **Feature: cattle-buffalo-recognition, Property 6: Type classification completeness**
 */
class TypeClassificationServiceTest {

    @Mock
    private lateinit var mockContext: Context
    
    @Mock
    private lateinit var mockAssetManager: AssetManager
    
    private lateinit var typeClassificationService: TypeClassificationService
    
    private val csvData = """
        breed_name,scientific_name,origin,animal_type,milk_yield_min,milk_yield_max,characteristics
        Gir,Bos indicus,Gujarat,dairy,8,12,"Heat tolerant|Disease resistant"
        Sahiwal,Bos indicus,Punjab,dairy,10,16,"High milk yield|Heat tolerant"
        Tharparkar,Bos indicus,Rajasthan,dual_purpose,6,10,"Drought tolerant|Hardy"
        Kangayam,Bos indicus,Tamil Nadu,draught,2,4,"Excellent draft animal|Heat tolerant"
        Murrah,Bubalus bubalis,Haryana,dairy,12,18,"High milk yield|Heat tolerant"
        Jaffarabadi,Bubalus bubalis,Gujarat,dual_purpose,8,14,"Large size|Good milk yield"
    """.trimIndent()

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        
        // Mock the asset manager to return our test CSV data
        `when`(mockContext.assets).thenReturn(mockAssetManager)
        `when`(mockAssetManager.open("data/breed_mapping.csv"))
            .thenReturn(ByteArrayInputStream(csvData.toByteArray()))
        
        typeClassificationService = TypeClassificationService(mockContext)
    }

    /**
     * Property test for type classification completeness.
     * **Validates: Requirements 3.1**
     */
    @Test
    fun `property test - type classification completeness for all breeds`() {
        // Initialize the service
        val initResult = typeClassificationService.initialize()
        assertTrue("Service should initialize successfully", initResult.isValid)
        
        // Get all breed names from the service
        val allBreeds = typeClassificationService.getAllBreedNames()
        
        // Property: For any identified breed, the system should classify it as exactly one of Dairy, Draught, or Dual-purpose type
        allBreeds.forEach { breed ->
            val classifiedType = typeClassificationService.classifyType(breed)
            
            // Verify that every breed has exactly one classification
            assertNotNull("Breed '$breed' should have a type classification", classifiedType)
            
            // Verify that the classification is one of the valid types
            assertTrue(
                "Breed '$breed' should be classified as one of the valid animal types",
                classifiedType in AnimalType.values()
            )
            
            // Verify consistency - calling multiple times should return the same result
            val secondClassification = typeClassificationService.classifyType(breed)
            assertEquals(
                "Type classification should be consistent for breed '$breed'",
                classifiedType,
                secondClassification
            )
        }
    }

    /**
     * Property test for breed existence completeness.
     * **Validates: Requirements 3.1**
     */
    @Test
    fun `property test - breed existence check completeness`() {
        // Initialize the service
        val initResult = typeClassificationService.initialize()
        assertTrue("Service should initialize successfully", initResult.isValid)
        
        // Get all breed names
        val allBreeds = typeClassificationService.getAllBreedNames()
        
        // Property: For any breed in the database, hasBreed should return true
        allBreeds.forEach { breed ->
            assertTrue(
                "hasBreed should return true for existing breed '$breed'",
                typeClassificationService.hasBreed(breed)
            )
            
            // Test case insensitivity
            assertTrue(
                "hasBreed should be case insensitive for breed '$breed'",
                typeClassificationService.hasBreed(breed.uppercase())
            )
            
            assertTrue(
                "hasBreed should be case insensitive for breed '$breed'",
                typeClassificationService.hasBreed(breed.lowercase())
            )
        }
    }

    /**
     * Property test for breed characteristics completeness.
     * **Validates: Requirements 3.1**
     */
    @Test
    fun `property test - breed characteristics completeness`() {
        // Initialize the service
        val initResult = typeClassificationService.initialize()
        assertTrue("Service should initialize successfully", initResult.isValid)
        
        // Get all breed names
        val allBreeds = typeClassificationService.getAllBreedNames()
        
        // Property: For any breed in the database, getBreedCharacteristics should return complete information
        allBreeds.forEach { breed ->
            val breedInfo = typeClassificationService.getBreedCharacteristics(breed)
            
            assertNotNull("Breed characteristics should exist for breed '$breed'", breedInfo)
            breedInfo?.let { info ->
                assertFalse("Breed name should not be empty for '$breed'", info.name.isBlank())
                assertFalse("Scientific name should not be empty for '$breed'", info.scientificName.isBlank())
                assertFalse("Origin should not be empty for '$breed'", info.origin.isBlank())
                assertTrue("Characteristics should not be empty for '$breed'", info.characteristics.isNotEmpty())
                assertTrue("Min milk yield should be non-negative for '$breed'", info.averageMilkYieldMin >= 0)
                assertTrue("Max milk yield should be non-negative for '$breed'", info.averageMilkYieldMax >= 0)
                assertTrue("Min yield should not exceed max yield for '$breed'", info.averageMilkYieldMin <= info.averageMilkYieldMax)
                
                // Verify that the breed info validates successfully
                val validation = info.validate()
                assertTrue("Breed info should be valid for '$breed': ${validation.errors}", validation.isValid)
            }
        }
    }

    /**
     * Property test for type-based breed filtering completeness.
     * **Validates: Requirements 3.1**
     */
    @Test
    fun `property test - type-based breed filtering completeness`() {
        // Initialize the service
        val initResult = typeClassificationService.initialize()
        assertTrue("Service should initialize successfully", initResult.isValid)
        
        // Property: For any animal type, getBreedsByType should return only breeds of that type
        AnimalType.values().forEach { animalType ->
            val breedsOfType = typeClassificationService.getBreedsByType(animalType)
            
            // Verify that all returned breeds are actually of the requested type
            breedsOfType.forEach { breedInfo ->
                assertEquals(
                    "Breed '${breedInfo.name}' should be of type $animalType",
                    animalType,
                    breedInfo.primaryUse
                )
                
                // Verify consistency with classifyType method
                val classifiedType = typeClassificationService.classifyType(breedInfo.name)
                assertEquals(
                    "classifyType should return same type as getBreedsByType for '${breedInfo.name}'",
                    animalType,
                    classifiedType
                )
            }
        }
    }

    /**
     * Property test for mapping validation completeness.
     * **Validates: Requirements 3.1**
     */
    @Test
    fun `property test - mapping validation completeness`() {
        // Initialize the service
        val initResult = typeClassificationService.initialize()
        assertTrue("Service should initialize successfully", initResult.isValid)
        
        // Property: validateMapping should verify all breed data integrity
        val validationResult = typeClassificationService.validateMapping()
        
        if (!validationResult.isValid) {
            // If validation fails, each error should be specific and actionable
            validationResult.errors.forEach { error ->
                assertFalse("Validation error should not be empty", error.isBlank())
                assertTrue("Validation error should contain breed name", error.contains(":"))
            }
        }
        
        // All breeds should have valid data if validation passes
        if (validationResult.isValid) {
            val allBreeds = typeClassificationService.getAllBreedNames()
            allBreeds.forEach { breed ->
                val breedInfo = typeClassificationService.getBreedCharacteristics(breed)
                assertNotNull("Valid breed should have characteristics: $breed", breedInfo)
                
                breedInfo?.let { info ->
                    val individualValidation = info.validate()
                    assertTrue(
                        "Individual breed validation should pass for '$breed': ${individualValidation.errors}",
                        individualValidation.isValid
                    )
                }
            }
        }
    }

    /**
     * Test for non-existent breeds to ensure proper handling.
     */
    @Test
    fun `test non-existent breed handling`() {
        // Initialize the service
        val initResult = typeClassificationService.initialize()
        assertTrue("Service should initialize successfully", initResult.isValid)
        
        val nonExistentBreeds = listOf("NonExistentBreed", "FakeBreed", "UnknownBreed")
        
        nonExistentBreeds.forEach { breed ->
            assertNull("Non-existent breed should return null type", typeClassificationService.classifyType(breed))
            assertNull("Non-existent breed should return null characteristics", typeClassificationService.getBreedCharacteristics(breed))
            assertFalse("Non-existent breed should return false for hasBreed", typeClassificationService.hasBreed(breed))
            assertNull("Non-existent breed should return null embedding", typeClassificationService.getMetricLearningEmbedding(breed))
        }
    }
}