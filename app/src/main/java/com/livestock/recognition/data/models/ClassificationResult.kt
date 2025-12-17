package com.livestock.recognition.data.models

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/**
 * Data class representing the result of breed classification and type identification.
 * Contains all information needed for display and report generation.
 */
@Parcelize
data class ClassificationResult(
    val breed: String,
    val breedConfidence: Float,
    val animalType: AnimalType,
    val typeConfidence: Float,
    val timestamp: Long,
    val imageMetadata: ImageMetadata,
    val processingTime: Long
) : Parcelable

/**
 * Metadata about the captured and processed image.
 */
@Parcelize
data class ImageMetadata(
    val width: Int,
    val height: Int,
    val fileSize: Long,
    val captureTime: Long,
    val deviceInfo: String,
    val qualityScore: Float
) : Parcelable

/**
 * Enumeration of animal types based on primary use.
 */
enum class AnimalType(val displayName: String, val description: String) {
    DAIRY("Dairy", "High milk production breeds"),
    DRAUGHT("Draught", "Farm work and transportation"),
    DUAL_PURPOSE("Dual-purpose", "Both milk production and farm work")
}

/**
 * Information about a specific breed including characteristics and usage.
 */
@Parcelize
data class BreedInfo(
    val name: String,
    val scientificName: String,
    val origin: String,
    val primaryUse: AnimalType,
    val averageMilkYieldMin: Int,
    val averageMilkYieldMax: Int,
    val characteristics: List<String>,
    val imageUrl: String?
) : Parcelable