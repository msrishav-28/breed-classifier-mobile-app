package com.livestock.recognition.core.classify

import com.livestock.recognition.core.model.Prediction
import kotlin.math.exp

/**
 * Turns raw model output into ranked predictions.
 *
 * The model contract (see docs/MODEL.md) is a probability distribution over
 * the label set, but this processor is defensive: scores are clamped and
 * renormalised so that malformed output can never produce confidences
 * outside [0, 1].
 */
object PredictionPostProcessor {

    /**
     * Ranks the [scores] against [labels] and returns the [k] most confident
     * predictions in descending confidence order.
     *
     * @param scores raw model output, one score per label
     * @param labels class labels in model output order
     * @param k number of predictions to return; capped at the label count
     * @param scoresAreLogits set when the model emits unnormalised logits,
     *   in which case a softmax is applied first
     */
    fun topPredictions(
        scores: FloatArray,
        labels: List<String>,
        k: Int,
        scoresAreLogits: Boolean = false,
    ): List<Prediction> {
        require(scores.size == labels.size) {
            "Model emitted ${scores.size} scores but ${labels.size} labels are configured"
        }
        require(k > 0) { "k must be positive, was $k" }
        if (scores.isEmpty()) return emptyList()

        val probabilities = if (scoresAreLogits) softmax(scores) else normalize(scores)

        return labels.indices
            .sortedByDescending { probabilities[it] }
            .take(k.coerceAtMost(labels.size))
            .map { Prediction(labels[it], probabilities[it]) }
    }

    /**
     * Clamps scores to be non-negative and rescales them to sum to 1.
     * A degenerate all-zero input yields a uniform distribution rather than
     * NaNs.
     */
    fun normalize(scores: FloatArray): FloatArray {
        val clamped = FloatArray(scores.size) { i ->
            val s = scores[i]
            if (s.isNaN() || s < 0f) 0f else s
        }
        val sum = clamped.sum()
        return if (sum <= 0f || sum.isNaN() || sum.isInfinite()) {
            FloatArray(scores.size) { 1f / scores.size }
        } else {
            FloatArray(scores.size) { i -> (clamped[i] / sum).coerceIn(0f, 1f) }
        }
    }

    /** Numerically stable softmax. */
    fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return FloatArray(0)
        val max = logits.max()
        val exps = DoubleArray(logits.size) { i -> exp((logits[i] - max).toDouble()) }
        val sum = exps.sum()
        return FloatArray(logits.size) { i -> (exps[i] / sum).toFloat().coerceIn(0f, 1f) }
    }
}
