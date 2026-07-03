package com.livestock.recognition.ml

import android.graphics.Bitmap
import com.livestock.recognition.core.model.Prediction
import java.io.Closeable

/**
 * Output of one classification run.
 *
 * @property predictions ranked predictions, most confident first; never empty
 * @property processingTimeMillis wall-clock inference time including preprocessing
 * @property modelVersion identifier of the model that produced the result
 */
data class ClassificationOutput(
    val predictions: List<Prediction>,
    val processingTimeMillis: Long,
    val modelVersion: String,
)

/**
 * A breed classifier. Implementations must be safe to call from any
 * dispatcher and must serialise access to any underlying native resources.
 */
interface BreedClassifier : Closeable {

    val modelVersion: String

    /**
     * Classifies [bitmap] and returns ranked predictions.
     *
     * @throws ClassificationException when inference fails
     */
    suspend fun classify(bitmap: Bitmap): ClassificationOutput
}

class ClassificationException(message: String, cause: Throwable? = null) :
    Exception(message, cause)
