package com.livestock.recognition.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.ml.ModelResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.*

/**
 * Metric Learning Service
 * Provides similarity-based breed classification using triplet loss embeddings
 * Used as fallback when ensemble accuracy falls below 95%
 */
@Singleton
class MetricLearningService @Inject constructor(
    private val context: Context
) {
    
    companion object {
        private const val TAG = "MetricLearningService"
        
        // Model files
        private const val METRIC_MODEL_FILE = "metric_learning_model.tflite"
        private const val EMBEDDINGS_FILE = "embedding_database.bin"
        private const val CENTROIDS_FILE = "class_centroids.bin"
        
        // Configuration
        private const val EMBEDDING_DIM = 512
        private const val K_NEIGHBORS = 5
        private const val SIMILARITY_THRESHOLD = 0.8f
        private const val ENSEMBLE_ACCURACY_THRESHOLD = 95.0f
        private const val LOW_CONFIDENCE_THRESHOLD = 0.75f
        
        // Input size
        private const val INPUT_SIZE = 224
    }
    
    private var interpreter: Interpreter? = null
    private var imageProcessor: ImageProcessor? = null
    private var embeddingDatabase: Array<FloatArray>? = null
    private var embeddingLabels: IntArray? = null
    private var classCentroids: Map<Int, FloatArray>? = null
    private var classLabels: List<String> = emptyList()
    private var isInitialized = false
    
    /**
     * Initialize metric learning service
     */
    suspend fun initialize(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Initializing Metric Learning Service...")
                
                // Load class labels
                classLabels = loadClassLabels()
                if (classLabels.isEmpty()) {
                    Log.e(TAG, "Failed to load class labels")
                    return@withContext false
                }
                
                // Load metric learning model
                val modelLoaded = loadMetricModel()
                if (!modelLoaded) {
                    Log.e(TAG, "Failed to load metric learning model")
                    return@withContext false
                }
                
                // Load embedding database
                val embeddingsLoaded = loadEmbeddingDatabase()
                if (!embeddingsLoaded) {
                    Log.w(TAG, "Failed to load embedding database - similarity features disabled")
                }
                
                // Create image processor
                createImageProcessor()
                
                isInitialized = true
                Log.d(TAG, "Metric Learning Service initialized successfully")
                
                true
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize metric learning service", e)
                false
            }
        }
    }
    
    /**
     * Check if metric learning should be used based on ensemble performance
     */
    fun shouldUseMetricLearning(ensembleAccuracy: Float, confidence: Float): Boolean {
        return ensembleAccuracy < ENSEMBLE_ACCURACY_THRESHOLD || 
               confidence < LOW_CONFIDENCE_THRESHOLD
    }
    
    /**
     * Get embedding for input image
     */
    suspend fun getImageEmbedding(bitmap: Bitmap): FloatArray? {
        if (!isInitialized) {
            Log.e(TAG, "Service not initialized")
            return null
        }
        
        return withContext(Dispatchers.Default) {
            try {
                val interpreter = this@MetricLearningService.interpreter ?: return@withContext null
                val processor = imageProcessor ?: return@withContext null
                
                // Preprocess image
                val tensorImage = TensorImage.fromBitmap(bitmap)
                val processedImage = processor.process(tensorImage)
                
                // Prepare output buffer for embedding
                val embeddingBuffer = TensorBuffer.createFixedSize(
                    intArrayOf(1, EMBEDDING_DIM), 
                    org.tensorflow.lite.DataType.FLOAT32
                )
                
                // Run inference to get embedding
                interpreter.run(processedImage.buffer, embeddingBuffer.buffer)
                
                // Get embedding and normalize
                val embedding = embeddingBuffer.floatArray
                normalizeEmbedding(embedding)
                
                embedding
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get image embedding", e)
                null
            }
        }
    }
    
    /**
     * Perform similarity-based classification
     */
    suspend fun similarityBasedClassification(bitmap: Bitmap): SimilarityResult? {
        if (!isInitialized || embeddingDatabase == null) {
            Log.e(TAG, "Service not initialized or embedding database not loaded")
            return null
        }
        
        return withContext(Dispatchers.Default) {
            try {
                // Get query embedding
                val queryEmbedding = getImageEmbedding(bitmap) ?: return@withContext null
                
                // Method 1: K-nearest neighbors
                val knnResult = performKNNClassification(queryEmbedding)
                
                // Method 2: Centroid-based classification
                val centroidResult = performCentroidClassification(queryEmbedding)
                
                // Combine results
                val finalResult = combineSimilarityResults(knnResult, centroidResult)
                
                Log.d(TAG, "Similarity classification: ${finalResult?.breed} (${finalResult?.confidence})")
                
                finalResult
                
            } catch (e: Exception) {
                Log.e(TAG, "Similarity-based classification failed", e)
                null
            }
        }
    }
    
    /**
     * Perform K-nearest neighbors classification
     */
    private fun performKNNClassification(queryEmbedding: FloatArray): SimilarityResult? {
        val database = embeddingDatabase ?: return null
        val labels = embeddingLabels ?: return null
        
        // Calculate similarities with all embeddings
        val similarities = database.mapIndexed { index, embedding ->
            index to cosineSimilarity(queryEmbedding, embedding)
        }
        
        // Get top-k most similar
        val topK = similarities.sortedByDescending { it.second }.take(K_NEIGHBORS)
        
        // Vote among top-k neighbors
        val labelCounts = mutableMapOf<Int, Int>()
        val labelSimilarities = mutableMapOf<Int, Float>()
        
        topK.forEach { (index, similarity) ->
            val label = labels[index]
            labelCounts[label] = labelCounts.getOrDefault(label, 0) + 1
            labelSimilarities[label] = maxOf(labelSimilarities.getOrDefault(label, 0f), similarity)
        }
        
        // Find most voted label
        val bestLabel = labelCounts.maxByOrNull { it.value }?.key ?: return null
        val confidence = labelCounts[bestLabel]!!.toFloat() / K_NEIGHBORS
        val similarity = labelSimilarities[bestLabel] ?: 0f
        
        return if (bestLabel < classLabels.size) {
            SimilarityResult(
                breed = classLabels[bestLabel],
                confidence = confidence,
                similarity = similarity,
                method = "KNN"
            )
        } else {
            null
        }
    }
    
    /**
     * Perform centroid-based classification
     */
    private fun performCentroidClassification(queryEmbedding: FloatArray): SimilarityResult? {
        val centroids = classCentroids ?: return null
        
        // Calculate similarities with all centroids
        val centroidSimilarities = centroids.mapValues { (_, centroid) ->
            cosineSimilarity(queryEmbedding, centroid)
        }
        
        // Find best centroid
        val bestCentroid = centroidSimilarities.maxByOrNull { it.value } ?: return null
        val classIndex = bestCentroid.key
        val similarity = bestCentroid.value
        
        return if (classIndex < classLabels.size) {
            SimilarityResult(
                breed = classLabels[classIndex],
                confidence = similarity,
                similarity = similarity,
                method = "Centroid"
            )
        } else {
            null
        }
    }
    
    /**
     * Combine KNN and centroid results
     */
    private fun combineSimilarityResults(
        knnResult: SimilarityResult?,
        centroidResult: SimilarityResult?
    ): SimilarityResult? {
        
        return when {
            knnResult == null && centroidResult == null -> null
            knnResult == null -> centroidResult
            centroidResult == null -> knnResult
            
            // Use centroid if high confidence
            centroidResult.similarity > SIMILARITY_THRESHOLD -> centroidResult
            
            // Otherwise use KNN
            else -> knnResult
        }
    }
    
    /**
     * Calculate cosine similarity between two embeddings
     */
    private fun cosineSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) {
            return 0f
        }
        
        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f
        
        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }
        
        val magnitude = sqrt(norm1) * sqrt(norm2)
        return if (magnitude > 0f) dotProduct / magnitude else 0f
    }
    
    /**
     * Normalize embedding to unit length
     */
    private fun normalizeEmbedding(embedding: FloatArray) {
        val norm = sqrt(embedding.sumOf { (it * it).toDouble() }).toFloat()
        if (norm > 0f) {
            for (i in embedding.indices) {
                embedding[i] /= norm
            }
        }
    }
    
    /**
     * Load metric learning model
     */
    private fun loadMetricModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(METRIC_MODEL_FILE)
            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(2) // Use fewer threads for metric learning
                    setUseNNAPI(true)
                }
                interpreter = Interpreter(modelBuffer, options)
                Log.d(TAG, "Metric learning model loaded successfully")
                true
            } else {
                Log.w(TAG, "Metric learning model file not found")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load metric learning model", e)
            false
        }
    }
    
    /**
     * Load embedding database from assets
     */
    private fun loadEmbeddingDatabase(): Boolean {
        return try {
            // In a real implementation, this would load the embedding database
            // and class centroids from binary files created during training
            
            // For now, create dummy data structure
            Log.w(TAG, "Using dummy embedding database - implement actual loading")
            
            // This would be replaced with actual loading code:
            // val embeddingData = loadBinaryFile(EMBEDDINGS_FILE)
            // val centroidData = loadBinaryFile(CENTROIDS_FILE)
            // Parse and populate embeddingDatabase, embeddingLabels, classCentroids
            
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load embedding database", e)
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
     * Create image processor
     */
    private fun createImageProcessor() {
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()
    }
    
    /**
     * Get service status
     */
    fun getServiceStatus(): MetricLearningStatus {
        return MetricLearningStatus(
            isInitialized = isInitialized,
            modelLoaded = interpreter != null,
            embeddingDatabaseLoaded = embeddingDatabase != null,
            centroidsLoaded = classCentroids != null,
            classLabelsCount = classLabels.size
        )
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            interpreter?.close()
            interpreter = null
            embeddingDatabase = null
            embeddingLabels = null
            classCentroids = null
            isInitialized = false
            
            Log.d(TAG, "Metric learning service cleaned up")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup metric learning service", e)
        }
    }
}

/**
 * Similarity-based classification result
 */
data class SimilarityResult(
    val breed: String,
    val confidence: Float,
    val similarity: Float,
    val method: String
)

/**
 * Metric learning service status
 */
data class MetricLearningStatus(
    val isInitialized: Boolean,
    val modelLoaded: Boolean,
    val embeddingDatabaseLoaded: Boolean,
    val centroidsLoaded: Boolean,
    val classLabelsCount: Int
)