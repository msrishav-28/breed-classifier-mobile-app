package com.livestock.recognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.repository.OfflineDataManager
import com.livestock.recognition.services.EnsembleCoordinationService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.withContext
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
import kotlin.math.max

/**
 * Advanced ML inference engine for ensemble models
 * Implements individual model predictions and ensemble coordination
 * Requirements: 2.1, 2.2, 2.3, 2.4
 */
@Singleton
class AdvancedMLInferenceEngine @Inject constructor(
    private val context: Context,
    private val offlineDataManager: OfflineDataManager,
    private val ensembleCoordinationService: EnsembleCoordinationService,
    private val modelValidationService: ModelValidationService
) {
    
    companion object {
        private const val TAG = "AdvancedMLInferenceEngine"
        
        // Model file names
        private const val VIT_MODEL_FILE = "vit_model_quantized.tflite"
        private const val YOLO_MODEL_FILE = "yolov8_cbam_model_quantized.tflite"
        private const val EFFICIENTNET_MODEL_FILE = "efficientnetv2_model_quantized.tflite"
        
        // Model input sizes
        private const val VIT_INPUT_SIZE = 224
        private const val YOLO_INPUT_SIZE = 640
        private const val EFFICIENTNET_INPUT_SIZE = 384
        
        // Performance thresholds
        private const val MAX_INFERENCE_TIME_MS = 3000L
        private const val MIN_CONFIDENCE_THRESHOLD = 0.3f
        private const val HIGH_CONFIDENCE_THRESHOLD = 0.75f
        private const val WARNING_CONFIDENCE_THRESHOLD = 0.75f
        
        // Test-time augmentation settings
        private const val TTA_ENABLED = true
        private const val TTA_ROTATIONS = 4
    }
    
    private var vitInterpreter: Interpreter? = null
    private var yoloInterpreter: Interpreter? = null
    private var efficientNetInterpreter: Interpreter? = null
    
    private var vitImageProcessor: ImageProcessor? = null
    private var yoloImageProcessor: ImageProcessor? = null
    private var efficientNetImageProcessor: ImageProcessor? = null
    
    private var classLabels: List<String> = emptyList()
    private var isInitialized = false
    private var performanceMetrics = PerformanceMetrics()
    
    /**
     * Initialize the advanced ML inference engine
     */
    suspend fun initialize(): AdvancedInitializationResult = withContext(Dispatchers.IO) {
        Log.d(TAG, "Initializing advanced ML inference engine...")
        
        try {
            val startTime = System.currentTimeMillis()
            
            // Validate models first
            val validationResult = modelValidationService.validateEnsembleModels()
            if (!validationResult.isValid) {
                return@withContext AdvancedInitializationResult(
                    success = false,
                    error = "Model validation failed: ${validationResult.error}"
                )
            }
            
            // Load class labels
            classLabels = loadClassLabels()
            if (classLabels.isEmpty()) {
                return@withContext AdvancedInitializationResult(
                    success = false,
                    error = "Failed to load class labels"
                )
            }
            
            // Load models in parallel
            val modelLoadingTasks = listOf(
                async { loadViTModel() },
                async { loadYOLOModel() },
                async { loadEfficientNetModel() }
            )
            
            val modelResults = modelLoadingTasks.awaitAll()
            val vitLoaded = modelResults[0]
            val yoloLoaded = modelResults[1]
            val efficientNetLoaded = modelResults[2]
            
            // At least one model must be loaded
            if (!vitLoaded && !yoloLoaded && !efficientNetLoaded) {
                return@withContext AdvancedInitializationResult(
                    success = false,
                    error = "Failed to load any models"
                )
            }
            
            // Create image processors
            createImageProcessors()
            
            // Initialize ensemble coordination service
            ensembleCoordinationService.initialize(
                vitAvailable = vitLoaded,
                yoloAvailable = yoloLoaded,
                efficientNetAvailable = efficientNetLoaded
            )
            
            val initTime = System.currentTimeMillis() - startTime
            isInitialized = true
            
            Log.d(TAG, "Advanced ML inference engine initialized in ${initTime}ms")
            Log.d(TAG, "Models loaded - ViT: $vitLoaded, YOLO: $yoloLoaded, EfficientNet: $efficientNetLoaded")
            
            AdvancedInitializationResult(
                success = true,
                initializationTimeMs = initTime,
                modelsLoaded = ModelLoadStatus(
                    vitLoaded = vitLoaded,
                    yoloLoaded = yoloLoaded,
                    efficientNetLoaded = efficientNetLoaded
                ),
                classLabelsCount = classLabels.size,
                validationResult = validationResult
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize advanced ML inference engine", e)
            AdvancedInitializationResult(
                success = false,
                error = "Initialization failed: ${e.message}"
            )
        }
    }
    
    /**
     * Run advanced inference with ensemble coordination
     */
    suspend fun runAdvancedInference(bitmap: Bitmap): AdvancedInferenceResult? = withContext(Dispatchers.Default) {
        if (!isInitialized) {
            Log.e(TAG, "Advanced inference engine not initialized")
            return@withContext null
        }
        
        return@withContext try {
            val startTime = System.currentTimeMillis()
            Log.d(TAG, "Running advanced ensemble inference...")
            
            // Run individual model predictions in parallel
            val predictionTasks = listOf(
                async { runViTPrediction(bitmap) },
                async { runYOLOPrediction(bitmap) },
                async { runEfficientNetPrediction(bitmap) }
            )
            
            val predictions = predictionTasks.awaitAll()
            val vitPrediction = predictions[0]
            val yoloPrediction = predictions[1]
            val efficientNetPrediction = predictions[2]
            
            // Apply test-time augmentation if enabled
            val ttaPredictions = if (TTA_ENABLED) {
                runTestTimeAugmentation(bitmap)
            } else {
                emptyList()
            }
            
            // Combine predictions using ensemble coordination
            val ensembleResult = ensembleCoordinationService.combineAdvancedPredictions(
                vitPrediction = vitPrediction,
                yoloPrediction = yoloPrediction,
                efficientNetPrediction = efficientNetPrediction,
                ttaPredictions = ttaPredictions
            )
            
            val processingTime = System.currentTimeMillis() - startTime
            
            // Update performance metrics
            performanceMetrics.recordInference(processingTime, ensembleResult?.confidence ?: 0f)
            
            // Check performance requirements
            val performanceValid = processingTime <= MAX_INFERENCE_TIME_MS
            
            if (ensembleResult != null) {
                Log.d(TAG, "Advanced inference completed in ${processingTime}ms")
                Log.d(TAG, "Result: ${ensembleResult.breed} (confidence: ${ensembleResult.confidence})")
                
                // Check if warning should be displayed
                val shouldWarn = ensembleResult.confidence < WARNING_CONFIDENCE_THRESHOLD
                
                AdvancedInferenceResult(
                    breed = ensembleResult.breed,
                    confidence = ensembleResult.confidence,
                    processingTimeMs = processingTime,
                    performanceValid = performanceValid,
                    shouldDisplayWarning = shouldWarn,
                    warningMessage = if (shouldWarn) generateWarningMessage(ensembleResult.confidence) else null,
                    individualPredictions = IndividualPredictions(
                        vitPrediction = vitPrediction,
                        yoloPrediction = yoloPrediction,
                        efficientNetPrediction = efficientNetPrediction
                    ),
                    ensembleMetrics = EnsembleMetrics(
                        modelsUsed = listOfNotNull(vitPrediction, yoloPrediction, efficientNetPrediction).size,
                        consensusStrength = calculateConsensusStrength(vitPrediction, yoloPrediction, efficientNetPrediction),
                        confidenceDistribution = calculateConfidenceDistribution(vitPrediction, yoloPrediction, efficientNetPrediction)
                    ),
                    ttaUsed = TTA_ENABLED,
                    ttaPredictionsCount = ttaPredictions.size
                )
            } else {
                Log.e(TAG, "Advanced ensemble inference failed")
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Advanced inference failed", e)
            null
        }
    }
    
    /**
     * Run Vision Transformer prediction
     */
    private suspend fun runViTPrediction(bitmap: Bitmap): ModelPrediction? = withContext(Dispatchers.Default) {
        return@withContext try {
            val interpreter = vitInterpreter ?: return@withContext null
            val processor = vitImageProcessor ?: return@withContext null
            
            val startTime = System.currentTimeMillis()
            
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = processor.process(tensorImage)
            
            // Prepare output buffer
            val outputBuffer = TensorBuffer.createFixedSize(
                intArrayOf(1, classLabels.size), 
                org.tensorflow.lite.DataType.FLOAT32
            )
            
            // Run inference
            interpreter.run(processedImage.buffer, outputBuffer.buffer)
            
            // Get predictions
            val predictions = outputBuffer.floatArray
            val topPredictions = getTopPredictions(predictions, 3)
            
            val inferenceTime = System.currentTimeMillis() - startTime
            
            if (topPredictions.isNotEmpty() && topPredictions[0].confidence > MIN_CONFIDENCE_THRESHOLD) {
                ModelPrediction(
                    modelName = "ViT",
                    breed = topPredictions[0].breed,
                    confidence = topPredictions[0].confidence,
                    inferenceTimeMs = inferenceTime,
                    topPredictions = topPredictions
                )
            } else {
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "ViT prediction failed", e)
            null
        }
    }
    
    /**
     * Run YOLOv8-CBAM prediction
     */
    private suspend fun runYOLOPrediction(bitmap: Bitmap): ModelPrediction? = withContext(Dispatchers.Default) {
        return@withContext try {
            val interpreter = yoloInterpreter ?: return@withContext null
            val processor = yoloImageProcessor ?: return@withContext null
            
            val startTime = System.currentTimeMillis()
            
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = processor.process(tensorImage)
            
            // YOLO has different output format - simplified for this implementation
            val outputBuffer = TensorBuffer.createFixedSize(
                intArrayOf(1, classLabels.size), 
                org.tensorflow.lite.DataType.FLOAT32
            )
            
            // Run inference
            interpreter.run(processedImage.buffer, outputBuffer.buffer)
            
            // Process YOLO output (simplified - real YOLO processing is more complex)
            val predictions = outputBuffer.floatArray
            val topPredictions = getTopPredictions(predictions, 3)
            
            val inferenceTime = System.currentTimeMillis() - startTime
            
            if (topPredictions.isNotEmpty() && topPredictions[0].confidence > MIN_CONFIDENCE_THRESHOLD) {
                ModelPrediction(
                    modelName = "YOLOv8-CBAM",
                    breed = topPredictions[0].breed,
                    confidence = topPredictions[0].confidence,
                    inferenceTimeMs = inferenceTime,
                    topPredictions = topPredictions
                )
            } else {
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "YOLO prediction failed", e)
            null
        }
    }
    
    /**
     * Run EfficientNetV2 prediction
     */
    private suspend fun runEfficientNetPrediction(bitmap: Bitmap): ModelPrediction? = withContext(Dispatchers.Default) {
        return@withContext try {
            val interpreter = efficientNetInterpreter ?: return@withContext null
            val processor = efficientNetImageProcessor ?: return@withContext null
            
            val startTime = System.currentTimeMillis()
            
            // Preprocess image
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = processor.process(tensorImage)
            
            // Prepare output buffer
            val outputBuffer = TensorBuffer.createFixedSize(
                intArrayOf(1, classLabels.size), 
                org.tensorflow.lite.DataType.FLOAT32
            )
            
            // Run inference
            interpreter.run(processedImage.buffer, outputBuffer.buffer)
            
            // Get predictions
            val predictions = outputBuffer.floatArray
            val topPredictions = getTopPredictions(predictions, 3)
            
            val inferenceTime = System.currentTimeMillis() - startTime
            
            if (topPredictions.isNotEmpty() && topPredictions[0].confidence > MIN_CONFIDENCE_THRESHOLD) {
                ModelPrediction(
                    modelName = "EfficientNetV2",
                    breed = topPredictions[0].breed,
                    confidence = topPredictions[0].confidence,
                    inferenceTimeMs = inferenceTime,
                    topPredictions = topPredictions
                )
            } else {
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "EfficientNet prediction failed", e)
            null
        }
    }
    
    /**
     * Run test-time augmentation
     */
    private suspend fun runTestTimeAugmentation(bitmap: Bitmap): List<ModelPrediction> = withContext(Dispatchers.Default) {
        val ttaPredictions = mutableListOf<ModelPrediction>()
        
        try {
            // Apply different augmentations
            for (rotation in 0 until TTA_ROTATIONS) {
                val rotatedBitmap = rotateBitmap(bitmap, rotation * 90f)
                
                // Run inference on augmented image (using best available model)
                val prediction = when {
                    vitInterpreter != null -> runViTPrediction(rotatedBitmap)
                    yoloInterpreter != null -> runYOLOPrediction(rotatedBitmap)
                    efficientNetInterpreter != null -> runEfficientNetPrediction(rotatedBitmap)
                    else -> null
                }
                
                prediction?.let { ttaPredictions.add(it) }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Test-time augmentation failed", e)
        }
        
        return@withContext ttaPredictions
    }
    
    /**
     * Get top predictions from model output
     */
    private fun getTopPredictions(predictions: FloatArray, topK: Int): List<BreedPrediction> {
        return predictions.mapIndexed { index, confidence ->
            BreedPrediction(
                breed = if (index < classLabels.size) classLabels[index] else "Unknown",
                confidence = confidence
            )
        }
        .sortedByDescending { it.confidence }
        .take(topK)
    }
    
    /**
     * Calculate consensus strength between models
     */
    private fun calculateConsensusStrength(
        vitPrediction: ModelPrediction?,
        yoloPrediction: ModelPrediction?,
        efficientNetPrediction: ModelPrediction?
    ): Float {
        val predictions = listOfNotNull(vitPrediction, yoloPrediction, efficientNetPrediction)
        if (predictions.size < 2) return 1.0f
        
        val topBreeds = predictions.map { it.breed }
        val mostCommonBreed = topBreeds.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
        
        return if (mostCommonBreed != null) {
            topBreeds.count { it == mostCommonBreed }.toFloat() / predictions.size
        } else {
            0.0f
        }
    }
    
    /**
     * Calculate confidence distribution across models
     */
    private fun calculateConfidenceDistribution(
        vitPrediction: ModelPrediction?,
        yoloPrediction: ModelPrediction?,
        efficientNetPrediction: ModelPrediction?
    ): ConfidenceDistribution {
        val confidences = listOfNotNull(vitPrediction, yoloPrediction, efficientNetPrediction)
            .map { it.confidence }
        
        return if (confidences.isNotEmpty()) {
            ConfidenceDistribution(
                mean = confidences.average().toFloat(),
                min = confidences.minOrNull() ?: 0f,
                max = confidences.maxOrNull() ?: 0f,
                standardDeviation = calculateStandardDeviation(confidences)
            )
        } else {
            ConfidenceDistribution(0f, 0f, 0f, 0f)
        }
    }
    
    /**
     * Calculate standard deviation of confidence scores
     */
    private fun calculateStandardDeviation(values: List<Float>): Float {
        if (values.size < 2) return 0f
        
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return kotlin.math.sqrt(variance).toFloat()
    }
    
    /**
     * Generate warning message for low confidence predictions
     */
    private fun generateWarningMessage(confidence: Float): String {
        return when {
            confidence < 0.5f -> "Low confidence prediction (${(confidence * 100).toInt()}%). Please retake photo with better lighting and focus."
            confidence < 0.65f -> "Moderate confidence prediction (${(confidence * 100).toInt()}%). Consider retaking photo for better accuracy."
            else -> "Below optimal confidence (${(confidence * 100).toInt()}%). Photo quality could be improved."
        }
    }
    
    /**
     * Rotate bitmap for test-time augmentation
     */
    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    
    // Model loading methods (similar to OfflineMLInferenceEngine but with enhancements)
    private fun loadViTModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(VIT_MODEL_FILE)
            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(true)
                    setUseXNNPACK(true)
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
    
    private fun loadYOLOModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(YOLO_MODEL_FILE)
            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(true)
                    setUseXNNPACK(true)
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
    
    private fun loadEfficientNetModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(EFFICIENTNET_MODEL_FILE)
            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(true)
                    setUseXNNPACK(true)
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
     * Get performance metrics
     */
    fun getPerformanceMetrics(): PerformanceMetrics = performanceMetrics
    
    /**
     * Get engine status
     */
    fun getEngineStatus(): AdvancedEngineStatus {
        return AdvancedEngineStatus(
            isInitialized = isInitialized,
            modelsLoaded = ModelLoadStatus(
                vitLoaded = vitInterpreter != null,
                yoloLoaded = yoloInterpreter != null,
                efficientNetLoaded = efficientNetInterpreter != null
            ),
            classLabelsCount = classLabels.size,
            performanceMetrics = performanceMetrics
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
            
            Log.d(TAG, "Advanced ML inference engine cleaned up")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup advanced inference engine", e)
        }
    }
}

// Data classes for advanced inference results
data class AdvancedInitializationResult(
    val success: Boolean,
    val initializationTimeMs: Long = 0,
    val modelsLoaded: ModelLoadStatus = ModelLoadStatus(),
    val classLabelsCount: Int = 0,
    val validationResult: EnsembleValidationResult? = null,
    val error: String? = null
)

data class AdvancedInferenceResult(
    val breed: String,
    val confidence: Float,
    val processingTimeMs: Long,
    val performanceValid: Boolean,
    val shouldDisplayWarning: Boolean,
    val warningMessage: String?,
    val individualPredictions: IndividualPredictions,
    val ensembleMetrics: EnsembleMetrics,
    val ttaUsed: Boolean,
    val ttaPredictionsCount: Int
)

data class ModelLoadStatus(
    val vitLoaded: Boolean = false,
    val yoloLoaded: Boolean = false,
    val efficientNetLoaded: Boolean = false
)

data class ModelPrediction(
    val modelName: String,
    val breed: String,
    val confidence: Float,
    val inferenceTimeMs: Long,
    val topPredictions: List<BreedPrediction>
)

data class BreedPrediction(
    val breed: String,
    val confidence: Float
)

data class IndividualPredictions(
    val vitPrediction: ModelPrediction?,
    val yoloPrediction: ModelPrediction?,
    val efficientNetPrediction: ModelPrediction?
)

data class EnsembleMetrics(
    val modelsUsed: Int,
    val consensusStrength: Float,
    val confidenceDistribution: ConfidenceDistribution
)

data class ConfidenceDistribution(
    val mean: Float,
    val min: Float,
    val max: Float,
    val standardDeviation: Float
)

data class AdvancedEngineStatus(
    val isInitialized: Boolean,
    val modelsLoaded: ModelLoadStatus,
    val classLabelsCount: Int,
    val performanceMetrics: PerformanceMetrics
)

data class PerformanceMetrics(
    var totalInferences: Int = 0,
    var averageInferenceTimeMs: Float = 0f,
    var averageConfidence: Float = 0f,
    var fastInferences: Int = 0, // Under 2 seconds
    var slowInferences: Int = 0, // Over 3 seconds
    var highConfidenceInferences: Int = 0 // Above 0.8
) {
    fun recordInference(timeMs: Long, confidence: Float) {
        totalInferences++
        averageInferenceTimeMs = (averageInferenceTimeMs * (totalInferences - 1) + timeMs) / totalInferences
        averageConfidence = (averageConfidence * (totalInferences - 1) + confidence) / totalInferences
        
        when {
            timeMs < 2000 -> fastInferences++
            timeMs > 3000 -> slowInferences++
        }
        
        if (confidence > 0.8f) {
            highConfidenceInferences++
        }
    }
}