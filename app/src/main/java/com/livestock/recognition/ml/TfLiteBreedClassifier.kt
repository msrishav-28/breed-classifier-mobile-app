package com.livestock.recognition.ml

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import com.livestock.recognition.core.classify.PredictionPostProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TensorFlow Lite implementation of [BreedClassifier].
 *
 * Model contract (see docs/MODEL.md):
 *  - asset `models/breed_classifier.tflite`
 *  - input: float32 tensor [1, 224, 224, 3], RGB pixel values in 0..255
 *    (any normalisation is baked into the model)
 *  - output: float32 tensor [1, N] of class probabilities
 *  - asset `models/labels.txt`: N labels, one per line, in output order
 */
class TfLiteBreedClassifier private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    override val modelVersion: String,
) : BreedClassifier {

    private val mutex = Mutex()

    override suspend fun classify(bitmap: Bitmap): ClassificationOutput =
        withContext(Dispatchers.Default) {
            val start = SystemClock.elapsedRealtime()
            val input = preprocess(bitmap)
            val output = Array(1) { FloatArray(labels.size) }

            mutex.withLock {
                try {
                    interpreter.run(input, output)
                } catch (e: RuntimeException) {
                    throw ClassificationException("Model inference failed", e)
                }
            }

            val predictions = PredictionPostProcessor.topPredictions(
                scores = output[0],
                labels = labels,
                k = TOP_K,
            )
            ClassificationOutput(
                predictions = predictions,
                processingTimeMillis = SystemClock.elapsedRealtime() - start,
                modelVersion = modelVersion,
            )
        }

    /** Center-crops to a square, scales to the model input size and packs floats. */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val cropSize = minOf(bitmap.width, bitmap.height)
        val cropX = (bitmap.width - cropSize) / 2
        val cropY = (bitmap.height - cropSize) / 2
        val square = Bitmap.createBitmap(bitmap, cropX, cropY, cropSize, cropSize)
        val scaled = Bitmap.createScaledBitmap(square, INPUT_SIZE, INPUT_SIZE, true)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val buffer = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * CHANNELS * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF).toFloat())
            buffer.putFloat(((pixel shr 8) and 0xFF).toFloat())
            buffer.putFloat((pixel and 0xFF).toFloat())
        }
        buffer.rewind()

        if (scaled !== square) scaled.recycle()
        if (square !== bitmap) square.recycle()
        return buffer
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        const val MODEL_ASSET = "models/breed_classifier.tflite"
        const val LABELS_ASSET = "models/labels.txt"
        const val INPUT_SIZE = 224
        private const val CHANNELS = 3
        private const val TOP_K = 3
        private const val NUM_THREADS = 4

        /** Cheap check for whether the model files are bundled in this build. */
        fun isModelBundled(assets: AssetManager): Boolean = try {
            assets.openFd(MODEL_ASSET).close()
            assets.open(LABELS_ASSET).close()
            true
        } catch (_: IOException) {
            false
        }

        /**
         * Loads the bundled model and validates it against the contract.
         *
         * @throws ClassificationException when the model or labels are
         *   missing or inconsistent with each other
         */
        fun create(context: Context): TfLiteBreedClassifier {
            val labels = try {
                context.assets.open(LABELS_ASSET).bufferedReader()
                    .readLines()
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
            } catch (e: IOException) {
                throw ClassificationException("Label file '$LABELS_ASSET' is missing", e)
            }
            if (labels.isEmpty()) {
                throw ClassificationException("Label file '$LABELS_ASSET' is empty")
            }

            val interpreter = try {
                val options = Interpreter.Options()
                options.setNumThreads(NUM_THREADS)
                Interpreter(mapAsset(context, MODEL_ASSET), options)
            } catch (e: IOException) {
                throw ClassificationException("Model '$MODEL_ASSET' is missing", e)
            } catch (e: RuntimeException) {
                throw ClassificationException("Model '$MODEL_ASSET' could not be loaded", e)
            }

            try {
                validateTensors(interpreter, labels.size)
            } catch (e: ClassificationException) {
                interpreter.close()
                throw e
            }

            return TfLiteBreedClassifier(
                interpreter = interpreter,
                labels = labels,
                modelVersion = "$MODEL_ASSET (${labels.size} classes)",
            )
        }

        private fun validateTensors(interpreter: Interpreter, labelCount: Int) {
            val input = interpreter.getInputTensor(0)
            val expectedInput = intArrayOf(1, INPUT_SIZE, INPUT_SIZE, CHANNELS)
            if (!input.shape().contentEquals(expectedInput) || input.dataType() != DataType.FLOAT32) {
                throw ClassificationException(
                    "Model input tensor is ${input.dataType()} ${input.shape().contentToString()}, " +
                        "expected FLOAT32 ${expectedInput.contentToString()}"
                )
            }
            val output = interpreter.getOutputTensor(0)
            val outputClasses = output.shape().lastOrNull() ?: 0
            if (outputClasses != labelCount) {
                throw ClassificationException(
                    "Model emits $outputClasses classes but '$LABELS_ASSET' lists $labelCount"
                )
            }
        }

        private fun mapAsset(context: Context, assetPath: String): MappedByteBuffer {
            context.assets.openFd(assetPath).use { fd ->
                FileInputStream(fd.fileDescriptor).use { stream ->
                    return stream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength,
                    )
                }
            }
        }
    }
}
