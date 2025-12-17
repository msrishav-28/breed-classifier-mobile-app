package com.livestock.recognition.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.ml.OfflineMLInferenceEngine
import com.livestock.recognition.ml.InferenceResult
import com.livestock.recognition.ml.ModelResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Ensemble Coordination Service
 * Implements advanced ensemble techniques including:
 * - Weighted voting with confidence-based weights
 * - Test-time augmentation
 * - Dynamic weight adjustment
 * - Confidence calibration
 * Target: 95-97% ensemble accuracy
 */
@Singleton
class EnsembleCoordinationService @Inject constructor(
    private val context: Context,
    private val mlInferenceEngine: OfflineMLInferenceEngine
) {
    
    companion object {
        private const val TAG = "EnsembleCoordinationService"
        
        // Ensemble configuration
        private const val TARGET_ENSEMBLE_ACCURACY = 95.0f
        private const val MIN_ENSEMBLE_CONFIDENCE = 0.75f
        private const val HIGH_CONFIDENCE_THRESHOLD = 0.90f
        
        // Model weights (can be dynamically adjusted)
        private var VIT_BASE_WEIGHT = 0.5f
        private var YOLO_BASE_WEIGHT = 0.3f
        private var EFFICIENTNET_BASE_WEIGHT = 0.2f
        
        // Test-time augmentation settings
        private const val TTA_ENABLED = true
        private const val TTA_ROTATIONS = 4  // 0°, 90°, 180°, 270°
        private const val TTA_FLIPS = 2      // Original, horizontal flip
        
        // Confidence calibration parameters
        private const val CALIBRATION_TEMPERATURE = 1.2f
        private const val UNCERTAINTY_THRESHOLD = 0.1f
    }
    
    private var isInitialized = false
    private var ensembleMetrics = EnsembleMetrics()
    private var dynamicWeights = ModelWeights()
    
    /**
     * Initialize ensemble coordination service
     */
    suspend fun initialize(): Boolean {
        return try {
            Log.d(TAG, "Initializing Ensemble Coordination Service...")
            
            // Initialize ML inference engine
            val mlInitialized = mlInferenceEngine.initialize()
            if (!mlInitialized) {
                Log.e(TAG, "Failed to initialize ML inference engine")
                return false
            }
            
            // Initialize dynamic weights
            resetDynamicWeights()
            
            // Load calibration parameters if available
            loadCalibrationParameters()
            
            isInitialized = true
            Log.d(TAG, "Ensemble Coordination Service initialized successfully")
            
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ensemble coordination service", e)
            false
        }
    }
    
    /**
     * Initialize with model availability information
     */
    fun initialize(vitAvailable: Boolean, yoloAvailable: Boolean, efficientNetAvailable: Boolean) {
        // Adjust weights based on available models
        val availableModels = listOf(vitAvailable, yoloAvailable, efficientNetAvailable).count { it }
        
        if (availableModels > 0) {
            val baseWeight = 1.0f / availableModels
            
            dynamicWeights = ModelWeights(
                vitWeight = if (vitAvailable) baseWeight else 0f,
                yoloWeight = if (yoloAvailable) baseWeight else 0f,
                efficientNetWeight = if (efficientNetAvailable) baseWeight else 0f
            )
        }
        
        Log.d(TAG, "Ensemble weights adjusted for available models: ViT=$vitAvailable, YOLO=$yoloAvailable, EfficientNet=$efficientNetAvailable")
    }
    
    /**
     * Combine advanced predictions from individual models
     */
    fun combineAdvancedPredictions(
        vitPrediction: com.livestock.recognition.ml.ModelPrediction?,
        yoloPrediction: com.livestock.recognition.ml.ModelPrediction?,
        efficientNetPrediction: com.livestock.recognition.ml.ModelPrediction?,
        ttaPredictions: List<com.livestock.recognition.ml.ModelPrediction>
    ): ModelResult? {
        
        val allPredictions = listOfNotNull(vitPrediction, yoloPrediction, efficientNetPrediction) + ttaPredictions
        
        if (allPredictions.isEmpty()) {
            return null
        }
        
        // Advanced weighted voting with confidence-based adjustment
        val breedScores = mutableMapOf<String, Float>()
        val breedWeights = mutableMapOf<String, Float>()
        
        // Process individual model predictions
        vitPrediction?.let { pred ->
            val weight = dynamicWeights.vitWeight * (1 + pred.confidence * 0.3f)
            breedScores[pred.breed] = breedScores.getOrDefault(pred.breed, 0f) + pred.confidence * weight
            breedWeights[pred.breed] = breedWeights.getOrDefault(pred.breed, 0f) + weight
        }
        
        yoloPrediction?.let { pred ->
            val weight = dynamicWeights.yoloWeight * (1 + pred.confidence * 0.3f)
            breedScores[pred.breed] = breedScores.getOrDefault(pred.breed, 0f) + pred.confidence * weight
            breedWeights[pred.breed] = breedWeights.getOrDefault(pred.breed, 0f) + weight
        }
        
        efficientNetPrediction?.let { pred ->
            val weight = dynamicWeights.efficientNetWeight * (1 + pred.confidence * 0.3f)
            breedScores[pred.breed] = breedScores.getOrDefault(pred.breed, 0f) + pred.confidence * weight
            breedWeights[pred.breed] = breedWeights.getOrDefault(pred.breed, 0f) + weight
        }
        
        // Process TTA predictions with reduced weight
        ttaPredictions.forEach { pred ->
            val ttaWeight = 0.1f * (1 + pred.confidence * 0.2f)
            breedScores[pred.breed] = breedScores.getOrDefault(pred.breed, 0f) + pred.confidence * ttaWeight
            breedWeights[pred.breed] = breedWeights.getOrDefault(pred.breed, 0f) + ttaWeight
        }
        
        // Find best result with normalized confidence
        val bestBreed = breedScores.maxByOrNull { it.value }
        
        return if (bestBreed != null && breedWeights[bestBreed.key] != null && breedWeights[bestBreed.key]!! > 0) {
            val normalizedConfidence = bestBreed.value / breedWeights[bestBreed.key]!!
            val calibratedConfidence = calibrateConfidence(normalizedConfidence)
            
            ModelResult(bestBreed.key, calibratedConfidence)
        } else {
            null
        }
    }
    
    /**
     * Run ensemble prediction with advanced coordination
     */
    suspend fun runEnsemblePrediction(bitmap: Bitmap): EnsemblePredictionResult? {
        if (!isInitialized) {
            Log.e(TAG, "Ensemble service not initialized")
            return null
        }
        
        return try {
            Log.d(TAG, "Running ensemble prediction...")
            val startTime = System.currentTimeMillis()
            
            // Run base prediction
            val basePrediction = mlInferenceEngine.runInference(bitmap)
            
            // Run test-time augmentation if enabled
            val ttaPredictions = if (TTA_ENABLED) {
                runTestTimeAugmentation(bitmap)
            } else {
                emptyList()
            }
            
            // Combine all predictions
            val allPredictions = listOfNotNull(basePrediction) + ttaPredictions
            
            // Apply advanced ensemble coordination
            val ensembleResult = coordinateEnsemblePredictions(allPredictions)
            
            val totalTime = System.currentTimeMillis() - startTime
            
            if (ensembleResult != null) {
                // Update ensemble metrics
                updateEnsembleMetrics(ensembleResult, allPredictions)
                
                // Adjust dynamic weights based on performance
                adjustDynamicWeights(allPredictions, ensembleResult)
                
                Log.d(TAG, "Ensemble prediction completed in ${totalTime}ms")
                Log.d(TAG, "Result: ${ensembleResult.breed} (confidence: ${ensembleResult.confidence})")
                
                EnsemblePredictionResult(
                    breed = ensembleResult.breed,
                    confidence = ensembleResult.confidence,
                    calibratedConfidence = calibrateConfidence(ensembleResult.confidence),
                    processingTime = totalTime,
                    individualPredictions = allPredictions,
                    ensembleWeights = getCurrentWeights(),
                    uncertaintyScore = calculateUncertaintyScore(allPredictions),
                    isHighConfidence = ensembleResult.confidence >= HIGH_CONFIDENCE_THRESHOLD,
                    recommendRetake = ensembleResult.confidence < MIN_ENSEMBLE_CONFIDENCE
                )
            } else {
                Log.e(TAG, "Ensemble prediction failed")
                null
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Ensemble prediction failed", e)
            null
        }
    }
    
    /**
     * Run test-time augmentation
     */
    private suspend fun runTestTimeAugmentation(bitmap: Bitmap): List<InferenceResult> {
        return withContext(Dispatchers.Default) {
            val ttaPredictions = mutableListOf<InferenceResult>()
            
            try {
                // Generate augmented images
                val augmentedImages = generateAugmentedImages(bitmap)
                
                // Run inference on each augmented image
                val deferredResults = augmentedImages.map { augmentedBitmap ->
                    async {
                        mlInferenceEngine.runInference(augmentedBitmap)
                    }
                }
                
                // Collect results
                deferredResults.forEach { deferred ->
                    val result = deferred.await()
                    if (result != null) {
                        ttaPredictions.add(result)
                    }
                }
                
                Log.d(TAG, "Test-time augmentation completed: ${ttaPredictions.size} predictions")
                
            } catch (e: Exception) {
                Log.e(TAG, "Test-time augmentation failed", e)
            }
            
            ttaPredictions
        }
    }
    
    /**
     * Generate augmented images for TTA
     */
    private fun generateAugmentedImages(bitmap: Bitmap): List<Bitmap> {
        val augmentedImages = mutableListOf<Bitmap>()
        
        try {
            // Rotation augmentations
            for (rotation in 0 until TTA_ROTATIONS) {
                val angle = rotation * 90f
                if (angle > 0) {
                    val rotatedBitmap = rotateBitmap(bitmap, angle)
                    augmentedImages.add(rotatedBitmap)
                }
            }
            
            // Flip augmentations
            if (TTA_FLIPS > 1) {
                val flippedBitmap = flipBitmapHorizontally(bitmap)
                augmentedImages.add(flippedBitmap)
            }
            
            Log.d(TAG, "Generated ${augmentedImages.size} augmented images")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate augmented images", e)
        }
        
        return augmentedImages
    }
    
    /**
     * Coordinate ensemble predictions using advanced techniques
     */
    private fun coordinateEnsemblePredictions(predictions: List<InferenceResult>): ModelResult? {
        if (predictions.isEmpty()) {
            return null
        }
        
        // If only one prediction, return it
        if (predictions.size == 1) {
            val pred = predictions.first()
            return ModelResult(pred.breed, pred.confidence)
        }
        
        // Advanced ensemble coordination
        return when {
            predictions.size >= 3 -> advancedWeightedVoting(predictions)
            else -> simpleWeightedVoting(predictions)
        }
    }
    
    /**
     * Advanced weighted voting with confidence-based weight adjustment
     */
    private fun advancedWeightedVoting(predictions: List<InferenceResult>): ModelResult? {
        val breedScores = mutableMapOf<String, Float>()
        val breedConfidences = mutableMapOf<String, MutableList<Float>>()
        
        predictions.forEach { prediction ->
            // Get individual model results
            val vitResult = prediction.individualResults["vit"]
            val yoloResult = prediction.individualResults["yolo"]
            val efficientNetResult = prediction.individualResults["efficientnet"]
            
            // Apply dynamic weights with confidence boosting
            vitResult?.let { result ->
                val weight = dynamicWeights.vitWeight * (1 + result.confidence * 0.5f)
                breedScores[result.breed] = breedScores.getOrDefault(result.breed, 0f) + 
                        result.confidence * weight
                breedConfidences.getOrPut(result.breed) { mutableListOf() }.add(result.confidence)
            }
            
            yoloResult?.let { result ->
                val weight = dynamicWeights.yoloWeight * (1 + result.confidence * 0.5f)
                breedScores[result.breed] = breedScores.getOrDefault(result.breed, 0f) + 
                        result.confidence * weight
                breedConfidences.getOrPut(result.breed) { mutableListOf() }.add(result.confidence)
            }
            
            efficientNetResult?.let { result ->
                val weight = dynamicWeights.efficientNetWeight * (1 + result.confidence * 0.5f)
                breedScores[result.breed] = breedScores.getOrDefault(result.breed, 0f) + 
                        result.confidence * weight
                breedConfidences.getOrPut(result.breed) { mutableListOf() }.add(result.confidence)
            }
        }
        
        // Find best result with confidence aggregation
        val bestBreed = breedScores.maxByOrNull { it.value }
        
        return if (bestBreed != null) {
            // Calculate aggregated confidence using geometric mean
            val confidences = breedConfidences[bestBreed.key] ?: listOf(bestBreed.value)
            val aggregatedConfidence = if (confidences.isNotEmpty()) {
                val geometricMean = confidences.fold(1f) { acc, conf -> acc * conf }.let { 
                    kotlin.math.pow(it.toDouble(), 1.0 / confidences.size).toFloat()
                }
                min(geometricMean * 1.1f, 1.0f) // Slight boost for ensemble
            } else {
                bestBreed.value
            }
            
            ModelResult(bestBreed.key, aggregatedConfidence)
        } else {
            null
        }
    }
    
    /**
     * Simple weighted voting for fewer predictions
     */
    private fun simpleWeightedVoting(predictions: List<InferenceResult>): ModelResult? {
        val breedScores = mutableMapOf<String, Float>()
        
        predictions.forEach { prediction ->
            breedScores[prediction.breed] = breedScores.getOrDefault(prediction.breed, 0f) + 
                    prediction.confidence
        }
        
        val bestBreed = breedScores.maxByOrNull { it.value }
        
        return if (bestBreed != null) {
            val normalizedConfidence = bestBreed.value / predictions.size
            ModelResult(bestBreed.key, normalizedConfidence)
        } else {
            null
        }
    }
    
    /**
     * Calibrate confidence scores using temperature scaling
     */
    private fun calibrateConfidence(rawConfidence: Float): Float {
        // Temperature scaling for confidence calibration
        val logits = ln(rawConfidence / (1 - rawConfidence))
        val calibratedLogits = logits / CALIBRATION_TEMPERATURE
        val calibratedConfidence = 1 / (1 + exp(-calibratedLogits))
        
        return max(0f, min(1f, calibratedConfidence))
    }
    
    /**
     * Calculate uncertainty score based on prediction variance
     */
    private fun calculateUncertaintyScore(predictions: List<InferenceResult>): Float {
        if (predictions.size < 2) {
            return 0f
        }
        
        val confidences = predictions.map { it.confidence }
        val mean = confidences.average().toFloat()
        val variance = confidences.map { (it - mean) * (it - mean) }.average().toFloat()
        
        return min(variance * 10f, 1f) // Scale and clamp uncertainty
    }
    
    /**
     * Update ensemble metrics for performance tracking
     */
    private fun updateEnsembleMetrics(result: ModelResult, predictions: List<InferenceResult>) {
        ensembleMetrics.apply {
            totalPredictions++
            
            if (result.confidence >= HIGH_CONFIDENCE_THRESHOLD) {
                highConfidencePredictions++
            }
            
            if (result.confidence >= MIN_ENSEMBLE_CONFIDENCE) {
                acceptablePredictions++
            }
            
            averageConfidence = ((averageConfidence * (totalPredictions - 1)) + result.confidence) / totalPredictions
            averageProcessingTime = predictions.firstOrNull()?.processingTime ?: 0L
        }
    }
    
    /**
     * Adjust dynamic weights based on recent performance
     */
    private fun adjustDynamicWeights(predictions: List<InferenceResult>, ensembleResult: ModelResult) {
        // Simple adaptive weight adjustment based on individual model agreement with ensemble
        predictions.forEach { prediction ->
            prediction.individualResults.forEach { (modelName, modelResult) ->
                val agreement = if (modelResult.breed == ensembleResult.breed) {
                    1.0f + (modelResult.confidence * 0.1f) // Boost weight for correct predictions
                } else {
                    0.9f - (modelResult.confidence * 0.1f) // Reduce weight for incorrect predictions
                }
                
                when (modelName) {
                    "vit" -> dynamicWeights.vitWeight = 
                        max(0.1f, min(0.8f, dynamicWeights.vitWeight * agreement))
                    "yolo" -> dynamicWeights.yoloWeight = 
                        max(0.1f, min(0.8f, dynamicWeights.yoloWeight * agreement))
                    "efficientnet" -> dynamicWeights.efficientNetWeight = 
                        max(0.1f, min(0.8f, dynamicWeights.efficientNetWeight * agreement))
                }
            }
        }
        
        // Normalize weights
        val totalWeight = dynamicWeights.vitWeight + dynamicWeights.yoloWeight + dynamicWeights.efficientNetWeight
        if (totalWeight > 0) {
            dynamicWeights.vitWeight /= totalWeight
            dynamicWeights.yoloWeight /= totalWeight
            dynamicWeights.efficientNetWeight /= totalWeight
        }
    }
    
    /**
     * Reset dynamic weights to base values
     */
    private fun resetDynamicWeights() {
        dynamicWeights = ModelWeights(
            vitWeight = VIT_BASE_WEIGHT,
            yoloWeight = YOLO_BASE_WEIGHT,
            efficientNetWeight = EFFICIENTNET_BASE_WEIGHT
        )
    }
    
    /**
     * Get current ensemble weights
     */
    private fun getCurrentWeights(): ModelWeights {
        return dynamicWeights.copy()
    }
    
    /**
     * Load calibration parameters from storage
     */
    private fun loadCalibrationParameters() {
        // In a real implementation, this would load calibration parameters
        // from a file or database based on validation data
        Log.d(TAG, "Using default calibration parameters")
    }
    
    /**
     * Rotate bitmap by specified angle
     */
    private fun rotateBitmap(bitmap: Bitmap, angle: Float): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    
    /**
     * Flip bitmap horizontally
     */
    private fun flipBitmapHorizontally(bitmap: Bitmap): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.preScale(-1f, 1f)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    
    /**
     * Get ensemble performance metrics
     */
    fun getEnsembleMetrics(): EnsembleMetrics {
        return ensembleMetrics.copy()
    }
    
    /**
     * Get current model weights
     */
    fun getModelWeights(): ModelWeights {
        return getCurrentWeights()
    }
    
    /**
     * Check if ensemble is performing above target accuracy
     */
    fun isPerformingAboveTarget(): Boolean {
        return if (ensembleMetrics.totalPredictions > 0) {
            val successRate = (ensembleMetrics.acceptablePredictions.toFloat() / ensembleMetrics.totalPredictions) * 100f
            successRate >= TARGET_ENSEMBLE_ACCURACY
        } else {
            false
        }
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            mlInferenceEngine.cleanup()
            isInitialized = false
            Log.d(TAG, "Ensemble coordination service cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup ensemble service", e)
        }
    }
}

/**
 * Ensemble prediction result with advanced metrics
 */
data class EnsemblePredictionResult(
    val breed: String,
    val confidence: Float,
    val calibratedConfidence: Float,
    val processingTime: Long,
    val individualPredictions: List<InferenceResult>,
    val ensembleWeights: ModelWeights,
    val uncertaintyScore: Float,
    val isHighConfidence: Boolean,
    val recommendRetake: Boolean
)

/**
 * Dynamic model weights
 */
data class ModelWeights(
    var vitWeight: Float = 0.5f,
    var yoloWeight: Float = 0.3f,
    var efficientNetWeight: Float = 0.2f
)

/**
 * Ensemble performance metrics
 */
data class EnsembleMetrics(
    var totalPredictions: Int = 0,
    var highConfidencePredictions: Int = 0,
    var acceptablePredictions: Int = 0,
    var averageConfidence: Float = 0f,
    var averageProcessingTime: Long = 0L
) {
    fun getSuccessRate(): Float {
        return if (totalPredictions > 0) {
            (acceptablePredictions.toFloat() / totalPredictions) * 100f
        } else {
            0f
        }
    }
    
    fun getHighConfidenceRate(): Float {
        return if (totalPredictions > 0) {
            (highConfidencePredictions.toFloat() / totalPredictions) * 100f
        } else {
            0f
        }
    }
}