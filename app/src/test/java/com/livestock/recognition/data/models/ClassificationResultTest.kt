package com.livestock.recognition.data.models

import com.google.gson.Gson
import org.junit.Test
import org.junit.Assert.*
import kotlin.random.Random

/**
 * Property-based tests for data model serialization and validation.
 * **Feature: cattle-buffalo-recognition, Property 12: JSON serialization round trip**
 */
class ClassificationResultTest {

    private val gson = Gson()

    /**
     * Property test for JSON serialization round trip.
     * **Validates: Requirements 8.3**
     */
    @Test
    fun `property test - JSON serialization round trip for ClassificationResult`() {
        // Run 100 iterations to simulate property-based testing
        repeat(100) {
            val original = generateRandomClassificationResult()
            
            // Serialize to JSON
            val json = gson.toJson(original)
            
            // Deserialize back from JSON
            val deserialized = gson.fromJson(json, ClassificationResult::class.java)
            
            // Verify round trip consistency
            assertEquals("Breed should match after round trip", original.breed, deserialized.breed)
            assertEquals("Breed confidence should match after round trip", original.breedConfidence, deserialized.breedConfidence, 0.001f)
            assertEquals("Animal type should match after round trip", original.animalType, deserialized.animalType)
            assertEquals("Type confidence should match after round trip", original.typeConfidence, deserialized.typeConfidence, 0.001f)
            assertEquals("Timestamp should match after round trip", original.timestamp, deserialized.timestamp)
            assertEquals("Processing time should match after round trip", original.processingTime, deserialized.processingTime)
            
            // Verify image metadata round trip
            assertEquals("Image width should match after round trip", original.imageMetadata.width, deserialized.imageMetadata.width)
            assertEquals("Image height should match after round trip", original.imageMetadata.height, deserialized.imageMetadata.height)
            assertEquals("File size should match after round trip", original.imageMetadata.fileSize, deserialized.imageMetadata.fileSize)
            assertEquals("Capture time should match after round trip", original.imageMetadata.captureTime, deserialized.imageMetadata.captureTime)
            assertEquals("Device info should match after round trip", original.imageMetadata.deviceInfo, deserialized.imageMetadata.deviceInfo)
            assertEquals("Quality score should match after round trip", original.imageMetadata.qualityScore, deserialized.imageMetadata.qualityScore, 0.001f)
        }
    }

    /**
     * Property test for JSON serialization round trip for BreedInfo.
     * **Validates: Requirements 8.3**
     */
    @Test
    fun `property test - JSON serialization round trip for BreedInfo`() {
        // Run 100 iterations to simulate property-based testing
        repeat(100) {
            val original = generateRandomBreedInfo()
            
            // Serialize to JSON
            val json = gson.toJson(original)
            
            // Deserialize back from JSON
            val deserialized = gson.fromJson(json, BreedInfo::class.java)
            
            // Verify round trip consistency
            assertEquals("Name should match after round trip", original.name, deserialized.name)
            assertEquals("Scientific name should match after round trip", original.scientificName, deserialized.scientificName)
            assertEquals("Origin should match after round trip", original.origin, deserialized.origin)
            assertEquals("Primary use should match after round trip", original.primaryUse, deserialized.primaryUse)
            assertEquals("Min milk yield should match after round trip", original.averageMilkYieldMin, deserialized.averageMilkYieldMin)
            assertEquals("Max milk yield should match after round trip", original.averageMilkYieldMax, deserialized.averageMilkYieldMax)
            assertEquals("Characteristics should match after round trip", original.characteristics, deserialized.characteristics)
            assertEquals("Image URL should match after round trip", original.imageUrl, deserialized.imageUrl)
        }
    }

    /**
     * Property test for AnimalType enum serialization.
     * **Validates: Requirements 8.3**
     */
    @Test
    fun `property test - JSON serialization round trip for AnimalType`() {
        // Test all enum values
        AnimalType.values().forEach { original ->
            // Serialize to JSON
            val json = gson.toJson(original)
            
            // Deserialize back from JSON
            val deserialized = gson.fromJson(json, AnimalType::class.java)
            
            // Verify round trip consistency
            assertEquals("AnimalType should match after round trip", original, deserialized)
            assertEquals("Display name should match after round trip", original.displayName, deserialized.displayName)
            assertEquals("Description should match after round trip", original.description, deserialized.description)
        }
    }

    // Generators for property-based testing

    private fun generateRandomClassificationResult(): ClassificationResult {
        return ClassificationResult(
            breed = generateRandomString(1, 50),
            breedConfidence = Random.nextFloat(),
            animalType = AnimalType.values().random(),
            typeConfidence = Random.nextFloat(),
            timestamp = Random.nextLong(1L, Long.MAX_VALUE),
            imageMetadata = generateRandomImageMetadata(),
            processingTime = Random.nextLong(0L, 10000L)
        )
    }

    private fun generateRandomImageMetadata(): ImageMetadata {
        return ImageMetadata(
            width = Random.nextInt(1, 4096),
            height = Random.nextInt(1, 4096),
            fileSize = Random.nextLong(1L, 100_000_000L),
            captureTime = Random.nextLong(1L, Long.MAX_VALUE),
            deviceInfo = generateRandomString(1, 100),
            qualityScore = Random.nextFloat()
        )
    }

    private fun generateRandomBreedInfo(): BreedInfo {
        val minYield = Random.nextInt(0, 50)
        val maxYield = Random.nextInt(minYield, minYield + 50)
        
        return BreedInfo(
            name = generateRandomString(1, 50),
            scientificName = generateRandomString(1, 100),
            origin = generateRandomString(1, 50),
            primaryUse = AnimalType.values().random(),
            averageMilkYieldMin = minYield,
            averageMilkYieldMax = maxYield,
            characteristics = generateRandomStringList(1, 10),
            imageUrl = if (Random.nextBoolean()) generateRandomString(1, 200) else null
        )
    }

    private fun generateRandomString(minLength: Int, maxLength: Int): String {
        val length = Random.nextInt(minLength, maxLength + 1)
        val chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
        return (1..length)
            .map { chars.random() }
            .joinToString("")
            .trim()
            .takeIf { it.isNotEmpty() } ?: "A" // Ensure non-empty string
    }

    private fun generateRandomStringList(minSize: Int, maxSize: Int): List<String> {
        val size = Random.nextInt(minSize, maxSize + 1)
        return (1..size).map { generateRandomString(1, 100) }
    }
}