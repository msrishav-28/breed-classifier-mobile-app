package com.livestock.recognition.services

import android.content.Context
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.models.BreedInfo
import com.livestock.recognition.data.models.ValidationResult
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Service for classifying animal breeds into types and managing breed information.
 * Provides breed-to-type mapping using CSV data and lookup functionality.
 */
class TypeClassificationService(private val context: Context) {

    private var breedDatabase: Map<String, BreedInfo> = emptyMap()
    private var isInitialized = false

    /**
     * Initializes the service by loading breed data from CSV file.
     * @return ValidationResult indicating success or failure with error messages
     */
    fun initialize(): ValidationResult {
        return try {
            loadBreedDatabase()
            isInitialized = true
            ValidationResult(true, emptyList())
        } catch (e: Exception) {
            ValidationResult(false, listOf("Failed to initialize breed database: ${e.message}"))
        }
    }

    /**
     * Classifies the animal type based on breed name.
     * @param breed The breed name to classify
     * @return AnimalType classification or null if breed not found
     */
    fun classifyType(breed: String): AnimalType? {
        if (!isInitialized) {
            val initResult = initialize()
            if (!initResult.isValid) {
                return null
            }
        }

        return breedDatabase[breed.lowercase()]?.primaryUse
    }

    /**
     * Retrieves detailed breed characteristics and information.
     * @param breed The breed name to look up
     * @return BreedInfo object with detailed information or null if not found
     */
    fun getBreedCharacteristics(breed: String): BreedInfo? {
        if (!isInitialized) {
            val initResult = initialize()
            if (!initResult.isValid) {
                return null
            }
        }

        return breedDatabase[breed.lowercase()]
    }

    /**
     * Validates the breed mapping data integrity.
     * @return ValidationResult indicating if all mappings are valid
     */
    fun validateMapping(): ValidationResult {
        if (!isInitialized) {
            val initResult = initialize()
            if (!initResult.isValid) {
                return initResult
            }
        }

        val errors = mutableListOf<String>()
        
        breedDatabase.values.forEach { breedInfo ->
            val validation = breedInfo.validate()
            if (!validation.isValid) {
                errors.addAll(validation.errors.map { "${breedInfo.name}: $it" })
            }
        }

        return ValidationResult(errors.isEmpty(), errors)
    }

    /**
     * Gets all available breed names in the database.
     * @return List of breed names
     */
    fun getAllBreedNames(): List<String> {
        if (!isInitialized) {
            initialize()
        }
        return breedDatabase.keys.toList()
    }

    /**
     * Gets all breeds of a specific animal type.
     * @param animalType The type to filter by
     * @return List of BreedInfo objects matching the type
     */
    fun getBreedsByType(animalType: AnimalType): List<BreedInfo> {
        if (!isInitialized) {
            initialize()
        }
        return breedDatabase.values.filter { it.primaryUse == animalType }
    }

    /**
     * Checks if a breed exists in the database.
     * @param breed The breed name to check
     * @return true if breed exists, false otherwise
     */
    fun hasBreed(breed: String): Boolean {
        if (!isInitialized) {
            initialize()
        }
        return breedDatabase.containsKey(breed.lowercase())
    }

    /**
     * Gets metric learning embedding vector for breed similarity analysis.
     * This is a placeholder for future metric learning implementation.
     * @param breed The breed name
     * @return FloatArray representing the embedding vector or null if not found
     */
    fun getMetricLearningEmbedding(breed: String): FloatArray? {
        // Placeholder implementation - would be replaced with actual metric learning embeddings
        val breedInfo = getBreedCharacteristics(breed) ?: return null
        
        // Simple feature vector based on breed characteristics
        return floatArrayOf(
            breedInfo.averageMilkYieldMin.toFloat(),
            breedInfo.averageMilkYieldMax.toFloat(),
            breedInfo.primaryUse.ordinal.toFloat(),
            breedInfo.characteristics.size.toFloat()
        )
    }

    /**
     * Loads breed database from CSV file in assets.
     */
    private fun loadBreedDatabase() {
        val inputStream = context.assets.open("data/breed_mapping.csv")
        val reader = BufferedReader(InputStreamReader(inputStream))
        
        val database = mutableMapOf<String, BreedInfo>()
        
        // Skip header line
        reader.readLine()
        
        reader.useLines { lines ->
            lines.forEach { line ->
                if (line.isNotBlank()) {
                    val parts = line.split(",")
                    if (parts.size >= 7) {
                        val breedName = parts[0].trim()
                        val scientificName = parts[1].trim()
                        val origin = parts[2].trim()
                        val animalType = when (parts[3].trim().lowercase()) {
                            "dairy" -> AnimalType.DAIRY
                            "draught" -> AnimalType.DRAUGHT
                            "dual_purpose" -> AnimalType.DUAL_PURPOSE
                            else -> AnimalType.DUAL_PURPOSE // Default fallback
                        }
                        val milkYieldMin = parts[4].trim().toIntOrNull() ?: 0
                        val milkYieldMax = parts[5].trim().toIntOrNull() ?: 0
                        val characteristics = parts[6].trim()
                            .removeSurrounding("\"")
                            .split("|")
                            .map { it.trim() }
                            .filter { it.isNotEmpty() }

                        val breedInfo = BreedInfo(
                            name = breedName,
                            scientificName = scientificName,
                            origin = origin,
                            primaryUse = animalType,
                            averageMilkYieldMin = milkYieldMin,
                            averageMilkYieldMax = milkYieldMax,
                            characteristics = characteristics,
                            imageUrl = null // Could be added later
                        )

                        database[breedName.lowercase()] = breedInfo
                    }
                }
            }
        }
        
        breedDatabase = database
    }
}