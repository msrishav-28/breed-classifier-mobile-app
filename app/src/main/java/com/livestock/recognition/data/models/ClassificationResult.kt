package com.livestock.recognition.data.models

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import com.google.gson.annotations.SerializedName

/**
 * Data class representing the result of breed classification and type identification.
 * Contains all information needed for display and report generation.
 */
@Parcelize
data class ClassificationResult(
    @SerializedName("breed")
    val breed: String,
    @SerializedName("breed_confidence")
    val breedConfidence: Float,
    @SerializedName("animal_type")
    val animalType: AnimalType,
    @SerializedName("type_confidence")
    val typeConfidence: Float,
    @SerializedName("timestamp")
    val timestamp: Long,
    @SerializedName("image_metadata")
    val imageMetadata: ImageMetadata,
    @SerializedName("processing_time")
    val processingTime: Long
) : Parcelable {
    
    /**
     * Validates the classification result data integrity.
     * @return ValidationResult indicating if the data is valid and any error messages
     */
    fun validate(): ValidationResult {
        val errors = mutableListOf<String>()
        
        if (breed.isBlank()) {
            errors.add("Breed name cannot be empty")
        }
        
        if (breedConfidence < 0f || breedConfidence > 1f) {
            errors.add("Breed confidence must be between 0.0 and 1.0")
        }
        
        if (typeConfidence < 0f || typeConfidence > 1f) {
            errors.add("Type confidence must be between 0.0 and 1.0")
        }
        
        if (timestamp <= 0) {
            errors.add("Timestamp must be positive")
        }
        
        if (processingTime < 0) {
            errors.add("Processing time cannot be negative")
        }
        
        val imageMetadataValidation = imageMetadata.validate()
        if (!imageMetadataValidation.isValid) {
            errors.addAll(imageMetadataValidation.errors.map { "Image metadata: $it" })
        }
        
        return ValidationResult(errors.isEmpty(), errors)
    }
}

/**
 * Metadata about the captured and processed image.
 */
@Parcelize
data class ImageMetadata(
    @SerializedName("width")
    val width: Int,
    @SerializedName("height")
    val height: Int,
    @SerializedName("file_size")
    val fileSize: Long,
    @SerializedName("capture_time")
    val captureTime: Long,
    @SerializedName("device_info")
    val deviceInfo: String,
    @SerializedName("quality_score")
    val qualityScore: Float
) : Parcelable {
    
    /**
     * Validates the image metadata integrity.
     * @return ValidationResult indicating if the data is valid and any error messages
     */
    fun validate(): ValidationResult {
        val errors = mutableListOf<String>()
        
        if (width <= 0) {
            errors.add("Image width must be positive")
        }
        
        if (height <= 0) {
            errors.add("Image height must be positive")
        }
        
        if (fileSize <= 0) {
            errors.add("File size must be positive")
        }
        
        if (captureTime <= 0) {
            errors.add("Capture time must be positive")
        }
        
        if (deviceInfo.isBlank()) {
            errors.add("Device info cannot be empty")
        }
        
        if (qualityScore < 0f || qualityScore > 1f) {
            errors.add("Quality score must be between 0.0 and 1.0")
        }
        
        return ValidationResult(errors.isEmpty(), errors)
    }
}

/**
 * Enumeration of animal types based on primary use.
 */
enum class AnimalType(val displayName: String, val description: String) {
    @SerializedName("dairy")
    DAIRY("Dairy", "High milk production breeds"),
    @SerializedName("draught")
    DRAUGHT("Draught", "Farm work and transportation"),
    @SerializedName("dual_purpose")
    DUAL_PURPOSE("Dual-purpose", "Both milk production and farm work")
}

/**
 * Information about a specific breed including characteristics and usage.
 */
@Parcelize
data class BreedInfo(
    @SerializedName("name")
    val name: String,
    @SerializedName("scientific_name")
    val scientificName: String,
    @SerializedName("origin")
    val origin: String,
    @SerializedName("primary_use")
    val primaryUse: AnimalType,
    @SerializedName("average_milk_yield_min")
    val averageMilkYieldMin: Int,
    @SerializedName("average_milk_yield_max")
    val averageMilkYieldMax: Int,
    @SerializedName("characteristics")
    val characteristics: List<String>,
    @SerializedName("image_url")
    val imageUrl: String?
) : Parcelable {
    
    /**
     * Validates the breed information data integrity.
     * @return ValidationResult indicating if the data is valid and any error messages
     */
    fun validate(): ValidationResult {
        val errors = mutableListOf<String>()
        
        if (name.isBlank()) {
            errors.add("Breed name cannot be empty")
        }
        
        if (scientificName.isBlank()) {
            errors.add("Scientific name cannot be empty")
        }
        
        if (origin.isBlank()) {
            errors.add("Origin cannot be empty")
        }
        
        if (averageMilkYieldMin < 0) {
            errors.add("Minimum milk yield cannot be negative")
        }
        
        if (averageMilkYieldMax < 0) {
            errors.add("Maximum milk yield cannot be negative")
        }
        
        if (averageMilkYieldMin > averageMilkYieldMax) {
            errors.add("Minimum milk yield cannot be greater than maximum")
        }
        
        if (characteristics.isEmpty()) {
            errors.add("Breed must have at least one characteristic")
        }
        
        if (characteristics.any { it.isBlank() }) {
            errors.add("Characteristics cannot contain empty strings")
        }
        
        return ValidationResult(errors.isEmpty(), errors)
    }
}

/**
 * Result of data validation containing validity status and error messages.
 */
data class ValidationResult(
    val isValid: Boolean,
    val errors: List<String>
)