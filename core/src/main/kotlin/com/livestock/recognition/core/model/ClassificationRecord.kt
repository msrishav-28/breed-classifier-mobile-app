package com.livestock.recognition.core.model

/**
 * The durable outcome of one classification run, independent of any
 * storage or UI representation.
 */
data class ClassificationRecord(
    val breedLabel: String,
    val confidence: Float,
    val animalType: AnimalType?,
    val alternatives: List<Prediction>,
    val capturedAtEpochMillis: Long,
    val processingTimeMillis: Long,
    val modelVersion: String?,
) {
    init {
        require(breedLabel.isNotBlank()) { "Breed label must not be blank" }
        require(confidence in 0f..1f) { "Confidence must be within [0, 1], was $confidence" }
        require(processingTimeMillis >= 0) { "Processing time must not be negative" }
    }
}
