package com.livestock.recognition.core.model

/**
 * A single class prediction produced by the classifier.
 *
 * @property label Raw model label, e.g. "Red_Sindhi".
 * @property confidence Probability in [0, 1].
 */
data class Prediction(
    val label: String,
    val confidence: Float,
) {
    init {
        require(label.isNotBlank()) { "Prediction label must not be blank" }
        require(confidence in 0f..1f) { "Confidence must be within [0, 1], was $confidence" }
    }
}
