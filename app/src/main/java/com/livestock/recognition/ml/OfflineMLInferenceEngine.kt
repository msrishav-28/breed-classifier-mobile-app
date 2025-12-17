package com.livestock.recognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.repository.OfflineDataManager
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Offline-capable ML inference engine
 * Runs ensemble models locally without network dependency
 * Implements caching and fallback mechanisms
 */
@Singleton
class OfflineMLInferenceEngine @Inject constructor(
    private val context: Context,
    private val offlineDataManager: OfflineDataManager
) {
    
    companion object {
        private const val TAG = "OfflineMLInferenceEngine"
        
        // Model file names
        private const val VIT_MODEL_FILE = "vit_model_quantized.tflite"
        private const val YOLO_MODEL_FILE = "yolov8_cbam_model.tflite"
        private const val EFFICIENTNET_MODEL_FILE = "efficientnetv2_model.tflite"
        
        // Model input sizes
        private const val VIT_INPUT_SIZE = 224
        private const val YOLO_INPUT_SIZE = 640
        private const val EFFICIENTNET_INPUT_SIZE = 384
        
        // Ensemble weights
        private const val VIT_WEIGHT = 0.5f
        private const val YOLO_WEIGHT = 0.3f
        private const val EFFICIENTNET_WEIGHT = 0.2f
        
        // Confidence thresholds
        private const val MIN_CONFIDENCE_THRESHOLD = 0.3f
        private const val HIGH_CONFIDENCE_THRESHOLD = 0.8f
    }
    
    private var vitInterpreter: Interpreter? = null
    private var yoloInterpreter: Interpreter? = null
    private var efficientNetInterpreter: Interpreter? = null
    
    private var vitImageProcessor: ImageProcessor? = null
    private var yoloImageProcessor: ImageProcessor? = null
    private var efficientNetImageProcessor: ImageProcessor? = null
    
    private var classLabels: List<String> = emptyList()
    private var isInitialized = false
    
    /**
     * Initialize the offline ML inference engine
     */
    suspend fun initialize(): Boolean {
        return try {
            Log.d(TAG, "Initializing offline ML inference engine...")
            
            // Load class labels
            classLabels = loadClassLabels()
            if (classLabels.isEmpty()) {
                Log.e(TAG, "Failed to load class labels")
                return false
            }
            
            // Load models
            val vitLoaded = loadViTModel()
            val yoloLoaded = loadYOLOModel()
            val efficientNetLoaded = loadEfficientNetModel()
            
            // At least one model must be loaded for basic functionality
            if (!vitLoaded && !yoloLoaded && !efficientNetLoaded) {
                Log.e(TAG, "Failed to load any models")
                return false
            }
            
            // Create image processors
            createImageProcessors()
            
            isInitialized = true
            Log.d(TAG, "Offline ML inference engine initialized successfully")
            Log.d(TAG, "Models loaded - ViT: $vitLoaded, YOLO: $yoloLoaded, EfficientNet: $efficientNetLoaded")
            
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize offline ML inference engine", e)
            false
        }
    }
    
    /**
     * Run inference on image using available models
     */
    suspend fun runInference(bitmap: Bitmap): InferenceResult? {
        if (!isInitialized) {
            Log.e(TAG, "Inference engine not initialized")
            return null
        }
        
        return try {
            Log.d(TAG, "Running offline inference...")
            
            val startTime = System.currentTimeMillis()
            
            // Run individual model inferences
            val vitResult = runViTInference(bitmap)
            val yoloResult = runYOLOInference(bitmap)
            val efficientNetResult = runEfficientNetInference(bitmap)
            
            // Combine results using ensemble voting
            val ensembleResult = combineEnsembleResults(vitResult, yoloResult, efficientNetResult)
            
            val processingTime = System.currentTimeMillis() - startTime
            
            if (ensembleResult != null) {
                Log.d(TAG, "Inference completed in ${processingTime}ms")
                Log.d(TAG, "Result: ${ensembleResult.breed} (${ensembleResult.confidence})")
                
                InferenceResult(
                    breed = ensembleResult.breed,
                    confidence = ensembleResult.confidence,
                    processingTime = processingTime,
                    individualResults = mapOf(
                        "vit" to vitResult,
                        "yolo" to yoloResult,
                        "efficientnet" to efficientNetResult
                    ).filterValues { it != null }
                )
            } else {
                Log.e(TAG, "Ensemble inference failed")
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            null
        }
    }
    
    /**
     * Load Vision Transformer model
     */
    private fun loadViTModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(VIT_MODEL_FILE)
            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(true)
                }
                vitInterpreter = Interpreter(modelBuffer, options)
                Log.d(TAG, "ViT model loaded successfully")
                true
            } else {
                Log.w(TAG, "ViT model file not found")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ViT model", e)
            false
        }
    }
    
    /**
     * Load YOLOv8 with CBAM model
     */
    private fun loadYOLOModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(YOLO_MODEL_FILE)
            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(true)
                }
                yoloInterpreter = Interpreter(modelBuffer, options)
                Log.d(TAG, "YOLO model loaded successfully")
                true
            } else {
                Log.w(TAG, "YOLO model file not found")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load YOLO model", e)
            false
        }
    }
    
    /**
     * Load EfficientNetV2 model
     */
    private fun loadEfficientNetModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(EFFICIENTNET_MODEL_FILE)
            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(true)
                }
                efficientNetInterpreter = Interpreter(modelBuffer, options)
                Log.d(TAG, "EfficientNet model loaded successfully")
                true
            } else {
                Log.w(TAG, "EfficientNet model file not found")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load EfficientNet model", e)
            false
        }
    }
    
    /**
     * Load model file from assets
     */
    private fun loadModelFile(fileName: String): MappedByteBuffer? {
        return try {
            val fileDescriptor = context.assets.openFd("models/$fileName")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load model file: $fileName", e)
            null
        }
    }
    
    /**
     * Load class labels from assets
     */
    private fun loadClassLabels(): List<String> {
        return try {
            val labels = context.assets.open("data/class_labels.txt").bufferedReader().readLines()
            Log.d(TAG, "Loaded ${labels.size} class labels")
            labels
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load class labels", e)
            // Fallback to hardcoded labels
            listOf(
                "Gir", "Sahiwal", "Red_Sindhi", "Tharparkar", "Rathi",
                "Hariana", "Ongole", "Krishna_Valley", "Deoni", "Khillari",
                "Malvi", "Nimari", "Dangi", "Gaolao", "Bachaur",
                "Murrah", "Mehsana", "Surti", "Jaffarabadi", "Nili_Ravi",
                "Kundi", "Bhadawari", "Toda"
            )
        }
    }
    
    /**
     * Create image processors for each model
     */
    private fun createImageProcessors() {
        vitImageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(VIT_INPUT_SIZE, VIT_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        
        yoloImageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        
        efficientNetImageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()
    }
    
    /**
     * Run ViT inference
     */
    private fun runViTInference(bitmap: Bitmap): ModelResult? {
        return try {
            val interpreter = vitInterpreter ?: return null
            val processor = vitImageProcessor ?: return null
            
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = processor.process(tensorImage)
            
            // Prepare output buffer
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, classLabels.size), org.tensorflow.lite.DataType.FLOAT32)
            
            // Run inference
            interpreter.run(processedImage.buffer, outputBuffer.buffer)
            
            // Get predictions
            val predictions = outputBuffer.floatArray
            val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: 0
            val confidence = predictions[maxIndex]
            
            if (confidence > MIN_CONFIDENCE_THRESHOLD && maxIndex < classLabels.size) {
                ModelResult(classLabels[maxIndex], confidence)
            } else {
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "ViT inference failed", e)
            null
        }
    }
    
    /**
     * Run YOLO inference
     */
    private fun runYOLOInference(bitmap: Bitmap): ModelResult? {
        return try {
            val interpreter = yoloInterpreter ?: return null
            val processor = yoloImageProcessor ?: return null
            
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = processor.process(tensorImage)
            
            // Prepare output buffer (YOLO has different output format)
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, classLabels.size), org.tensorflow.lite.DataType.FLOAT32)
            
            // Run inference
            interpreter.run(processedImage.buffer, outputBuffer.buffer)
            
            // Get predictions (simplified - real YOLO output processing is more complex)
            val predictions = outputBuffer.floatArray
            val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: 0
            val confidence = predictions[maxIndex]
            
            if (confidence > MIN_CONFIDENCE_THRESHOLD && maxIndex < classLabels.size) {
                ModelResult(classLabels[maxIndex], confidence)
            } else {
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "YOLO inference failed", e)
            null
        }
    }
    
    /**
     * Run EfficientNet inference
     */
    private fun runEfficientNetInference(bitmap: Bitmap): ModelResult? {
        return try {
            val interpreter = efficientNetInterpreter ?: return null
            val processor = efficientNetImageProcessor ?: return null
            
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = processor.process(tensorImage)
            
            // Prepare output buffer
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, classLabels.size), org.tensorflow.lite.DataType.FLOAT32)
            
            // Run inference
            interpreter.run(processedImage.buffer, outputBuffer.buffer)
            
            // Get predictions
            val predictions = outputBuffer.floatArray
            val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: 0
            val confidence = predictions[maxIndex]
            
            if (confidence > MIN_CONFIDENCE_THRESHOLD && maxIndex < classLabels.size) {
                ModelResult(classLabels[maxIndex], confidence)
            } else {
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "EfficientNet inference failed", e)
            null
        }
    }
    
    /**
     * Combine ensemble results using weighted voting
     */
    private fun combineEnsembleResults(
        vitResult: ModelResult?,
        yoloResult: ModelResult?,
        efficientNetResult: ModelResult?
    ): ModelResult? {
        
        val results = listOfNotNull(vitResult, yoloResult, efficientNetResult)
        if (results.isEmpty()) {
            return null
        }
        
        // If only one model produced a result, use it
        if (results.size == 1) {
            return results.first()
        }
        
        // Weighted voting
        val breedScores = mutableMapOf<String, Float>()
        
        vitResult?.let { result ->
            breedScores[result.breed] = breedScores.getOrDefault(result.breed, 0f) + 
                    result.confidence * VIT_WEIGHT
        }
        
        yoloResult?.let { result ->
            breedScores[result.breed] = breedScores.getOrDefault(result.breed, 0f) + 
                    result.confidence * YOLO_WEIGHT
        }
        
        efficientNetResult?.let { result ->
            breedScores[result.breed] = breedScores.getOrDefault(result.breed, 0f) + 
                    result.confidence * EFFICIENTNET_WEIGHT
        }
        
        // Find best result
        val bestBreed = breedScores.maxByOrNull { it.value }
        
        return if (bestBreed != null && bestBreed.value > MIN_CONFIDENCE_THRESHOLD) {
            ModelResult(bestBreed.key, bestBreed.value)
        } else {
            null
        }
    }
    
    /**
     * Check if models are loaded and ready
     */
    fun areModelsLoaded(): Boolean {
        return isInitialized && (vitInterpreter != null || yoloInterpreter != null || efficientNetInterpreter != null)
    }
    
    /**
     * Get model status information
     */
    fun getModelStatus(): ModelStatus {
        return ModelStatus(
            vitLoaded = vitInterpreter != null,
            yoloLoaded = yoloInterpreter != null,
            efficientNetLoaded = efficientNetInterpreter != null,
            classLabelsCount = classLabels.size,
            isInitialized = isInitialized
        )
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            vitInterpreter?.close()
            yoloInterpreter?.close()
            efficientNetInterpreter?.close()
            
            vitInterpreter = null
            yoloInterpreter = null
            efficientNetInterpreter = null
            
            isInitialized = false
            
            Log.d(TAG, "Offline ML inference engine cleaned up")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup inference engine", e)
        }
    }
}

/**
 * Result from individual model inference
 */
data class ModelResult(
    val breed: String,
    val confidence: Float
)

/**
 * Combined inference result
 */
data class InferenceResult(
    val breed: String,
    val confidence: Float,
    val processingTime: Long,
    val individualResults: Map<String, ModelResult>
)

/**
 * Model loading status
 */
data class ModelStatus(
    val vitLoaded: Boolean,
    val yoloLoaded: Boolean,
    val efficientNetLoaded: Boolean,
    val classLabelsCount: Int,
    val isInitialized: Boolean
)