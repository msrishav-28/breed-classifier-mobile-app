package com.livestock.recognition.core.classify

/**
 * Product policy for how much to trust a prediction. Kept in one place so
 * the UI, reports and tests cannot drift apart.
 */
object ConfidencePolicy {

    /** At or above this the prediction is presented without reservations. */
    const val HIGH_THRESHOLD = 0.75f

    /** Between this and [HIGH_THRESHOLD] the prediction is plausible but flagged. */
    const val MEDIUM_THRESHOLD = 0.50f

    enum class Level { HIGH, MEDIUM, LOW }

    fun levelFor(confidence: Float): Level = when {
        confidence >= HIGH_THRESHOLD -> Level.HIGH
        confidence >= MEDIUM_THRESHOLD -> Level.MEDIUM
        else -> Level.LOW
    }

    /** Whether the UI should show a "consider retaking the photo" warning. */
    fun requiresWarning(confidence: Float): Boolean = confidence < HIGH_THRESHOLD
}
