package com.livestock.recognition.services

import android.content.Context
import android.graphics.Bitmap
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.repository.OfflineDataManager
import com.livestock.recognition.data.repository.ClassificationRepository
import com.livestock.recognition.ml.AdvancedMLInferenceEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.security.MessageDigest
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Service for handling offline processing capabilities
 * Ensures all ML models work without network connectivity
 * Implements result caching and network state handling
 */
@Singleton
class OfflineProcessingService @Inject constructor(
    private val context: Context,
    private val mlInferenceEngine: AdvancedMLInferenceEngine,
    private val typeClassificationService: TypeClassificationService,
    private val offlineDataManager: OfflineDataManager,
    private val classificationRepository: ClassificationRepository
) {
    
    companion object {
        private const val TAG = "OfflineProcessingService"
        private const val CACHE_EXPIRY_HOURS = 24
        private const val MAX_OFFLINE_RESULTS = 1000
    }
    
    private var isOfflineMode = false
    
    /**
     * Initialize offline processing service
     * Loads models and prepares for offline operation
     */
    suspend fun initialize(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Initializing offline processing service...")
                
                // Check network state
                updateNetworkState()
                
                // Ensure ML models are loaded locally
                val modelsLoaded = mlInferenceEngine.loadModels()
                if (!modelsLoaded) {
                    Log.e(TAG, "Failed to load ML models for offline processing")
                    return@withContext false
                }
                
                // Verify type classification service is ready
                val typeServiceReady = typeClassificationService.validateMapping()
                if (!typeServiceReady.isValid) {
                    Log.e(TAG, "Type classification service not ready: ${typeServiceReady.errors}")
                    return@withContext false
                }
                
                // Clean up expired cache
                offlineDataManager.cleanupExpiredCache()
                
                Log.d(TAG, "Offline processing service initialized successfully")
                Log.d(TAG, "Network available: ${!isOfflineMode}")
                
                true
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize offline processing service", e)
                false
            }
        }
    }
    
    /**
     * Process image for breed recognition in offline mode
     * Uses cached results when available, performs inference when needed
     */
    suspend fun processImageOffline(bitmap: Bitmap, imagePath: String? = null): OfflineProcessingResult {
        return withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Processing image in offline mode...")
                
                // Generate image hash for caching
                val imageHash = generateImageHash(bitmap)
                
                // Check for cached result first
                val cachedResult = offlineDataManager.getCachedClassificationResult(imageHash)
                if (cachedResult != null) {
                    Log.d(TAG, "Using cached classification result")
                    return@withContext OfflineProcessingResult.Success(
                        result = cachedResult,
                        fromCache = true,
                        processingTime = 0L
                    )
                }
                
                // Perform ML inference
                val startTime = System.currentTimeMillis()
                
                // Run ensemble inference
                val inferenceResult = mlInferenceEngine.runInference(bitmap)
                if (inferenceResult == null) {
                    Log.e(TAG, "ML inference failed")
                    return@withContext OfflineProcessingResult.Error("ML inference failed")
                }
                
                // Classify animal type
                val animalType = typeClassificationService.classifyType(inferenceResult.breed)
                
                // Create classification result
                val classificationResult = ClassificationResult(
                    breed = inferenceResult.breed,
                    breedConfidence = inferenceResult.confidence,
                    animalType = animalType,
                    typeConfidence = inferenceResult.confidence, // Use same confidence for now
                    timestamp = System.currentTimeMillis(),
                    imageMetadata = null, // Would be populated with actual metadata
                    processingTime = System.currentTimeMillis() - startTime
                )
                
                // Cache the result for future offline use
                offlineDataManager.cacheClassificationResult(imageHash, classificationResult)
                
                // Save to local database
                val savedId = classificationRepository.saveClassification(classificationResult)
                
                Log.d(TAG, "Image processed successfully in offline mode")
                Log.d(TAG, "Breed: ${classificationResult.breed}, Confidence: ${classificationResult.breedConfidence}")
                
                OfflineProcessingResult.Success(
                    result = classificationResult,
                    fromCache = false,
                    processingTime = classificationResult.processingTime
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to process image offline", e)
                OfflineProcessingResult.Error("Processing failed: ${e.message}")
            }
        }
    }
    
    /**
     * Check if device is currently in offline mode
     */
    fun isOfflineMode(): Boolean {
        updateNetworkState()
        return isOfflineMode
    }
    
    /**
     * Update network state and offline mode status
     */
    private fun updateNetworkState() {
        isOfflineMode = !offlineDataManager.isNetworkAvailable()
    }
    
    /**
     * Get offline processing statistics
     */
    suspend fun getOfflineStatistics(): OfflineStatistics {
        return withContext(Dispatchers.IO) {
            try {
                val totalClassifications = classificationRepository.getClassificationCount()
                val unsyncedCount = classificationRepository.getUnsyncedCount()
                val cacheStats = offlineDataManager.getCacheStatistics()
                val averageProcessingTime = classificationRepository.getAverageProcessingTime()
                
                OfflineStatistics(
                    totalOfflineClassifications = totalClassifications,
                    unsyncedClassifications = unsyncedCount,
                    cacheEntries = cacheStats.totalEntries,
                    cacheSizeBytes = cacheStats.totalSizeBytes,
                    averageProcessingTimeMs = averageProcessingTime,
                    isOfflineMode = isOfflineMode
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get offline statistics", e)
                OfflineStatistics(0, 0, 0, 0L, 0f, isOfflineMode)
            }
        }
    }
    
    /**
     * Synchronize offline data when network becomes available
     */
    suspend fun synchronizeWhenOnline(): SyncResult {
        return withContext(Dispatchers.IO) {
            updateNetworkState()
            
            if (isOfflineMode) {
                Log.d(TAG, "Still offline, cannot synchronize")
                return@withContext SyncResult(false, 0, "Device is offline")
            }
            
            Log.d(TAG, "Network available, starting synchronization...")
            
            try {
                // Synchronize offline data
                val syncResult = offlineDataManager.synchronizeOfflineData()
                
                if (syncResult.success) {
                    Log.d(TAG, "Synchronization completed: ${syncResult.syncedCount} items synced")
                } else {
                    Log.w(TAG, "Synchronization failed: ${syncResult.message}")
                }
                
                syncResult
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to synchronize offline data", e)
                SyncResult(false, 0, "Sync failed: ${e.message}")
            }
        }
    }
    
    /**
     * Validate offline processing capabilities
     */
    suspend fun validateOfflineCapabilities(): OfflineValidationResult {
        return withContext(Dispatchers.IO) {
            val issues = mutableListOf<String>()
            val warnings = mutableListOf<String>()
            
            try {
                // Check ML models
                if (!mlInferenceEngine.areModelsLoaded()) {
                    issues.add("ML models not loaded")
                }
                
                // Check type classification service
                val typeValidation = typeClassificationService.validateMapping()
                if (!typeValidation.isValid) {
                    issues.addAll(typeValidation.errors)
                }
                
                // Check database access
                try {
                    classificationRepository.getClassificationCount()
                } catch (e: Exception) {
                    issues.add("Database access failed: ${e.message}")
                }
                
                // Check cache system
                try {
                    offlineDataManager.getCacheStatistics()
                } catch (e: Exception) {
                    issues.add("Cache system failed: ${e.message}")
                }
                
                // Check storage space
                val cacheStats = offlineDataManager.getCacheStatistics()
                if (cacheStats.totalSizeBytes > 50 * 1024 * 1024) { // 50MB warning
                    warnings.add("Cache size is large: ${cacheStats.totalSizeBytes / (1024 * 1024)}MB")
                }
                
                // Check unsynced data count
                val unsyncedCount = classificationRepository.getUnsyncedCount()
                if (unsyncedCount > 100) {
                    warnings.add("Many unsynced classifications: $unsyncedCount")
                }
                
                OfflineValidationResult(
                    isValid = issues.isEmpty(),
                    issues = issues,
                    warnings = warnings
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to validate offline capabilities", e)
                OfflineValidationResult(
                    isValid = false,
                    issues = listOf("Validation failed: ${e.message}"),
                    warnings = emptyList()
                )
            }
        }
    }
    
    /**
     * Clean up offline data based on retention policy
     */
    suspend fun cleanupOfflineData(retentionDays: Int = 30): CleanupResult {
        return withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Cleaning up offline data older than $retentionDays days...")
                
                // Clean up old classifications
                val deletedClassifications = classificationRepository.cleanupOldData(retentionDays)
                
                // Clean up expired cache
                val deletedCacheEntries = offlineDataManager.cleanupExpiredCache()
                
                Log.d(TAG, "Cleanup completed: $deletedClassifications classifications, $deletedCacheEntries cache entries")
                
                CleanupResult(
                    success = true,
                    deletedClassifications = deletedClassifications,
                    deletedCacheEntries = deletedCacheEntries,
                    message = "Cleanup completed successfully"
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to cleanup offline data", e)
                CleanupResult(
                    success = false,
                    deletedClassifications = 0,
                    deletedCacheEntries = 0,
                    message = "Cleanup failed: ${e.message}"
                )
            }
        }
    }
    
    /**
     * Generate hash for image to use as cache key
     */
    private fun generateImageHash(bitmap: Bitmap): String {
        try {
            // Create a smaller version for hashing
            val smallBitmap = Bitmap.createScaledBitmap(bitmap, 64, 64, false)
            
            // Convert to byte array
            val pixels = IntArray(64 * 64)
            smallBitmap.getPixels(pixels, 0, 64, 0, 0, 64, 64)
            
            // Create hash
            val digest = MessageDigest.getInstance("MD5")
            val bytes = pixels.map { it.toByte() }.toByteArray()
            val hashBytes = digest.digest(bytes)
            
            // Convert to hex string
            return hashBytes.joinToString("") { "%02x".format(it) }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate image hash", e)
            return System.currentTimeMillis().toString()
        }
    }
    
    /**
     * Get cached results count
     */
    suspend fun getCachedResultsCount(): Int {
        return try {
            offlineDataManager.getCacheStatistics().totalEntries
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get cached results count", e)
            0
        }
    }
    
    /**
     * Clear all offline data
     */
    suspend fun clearAllOfflineData() {
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Clearing all offline data...")
                
                // Clear cache
                offlineDataManager.clearAllCache()
                
                // Clear classifications (keep synced ones)
                // classificationRepository.deleteAllClassifications() // Uncomment if needed
                
                Log.d(TAG, "All offline data cleared")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to clear offline data", e)
            }
        }
    }
}

/**
 * Result of offline processing operation
 */
sealed class OfflineProcessingResult {
    data class Success(
        val result: ClassificationResult,
        val fromCache: Boolean,
        val processingTime: Long
    ) : OfflineProcessingResult()
    
    data class Error(val message: String) : OfflineProcessingResult()
}

/**
 * Statistics about offline processing
 */
data class OfflineStatistics(
    val totalOfflineClassifications: Int,
    val unsyncedClassifications: Int,
    val cacheEntries: Int,
    val cacheSizeBytes: Long,
    val averageProcessingTimeMs: Float,
    val isOfflineMode: Boolean
)

/**
 * Result of offline capabilities validation
 */
data class OfflineValidationResult(
    val isValid: Boolean,
    val issues: List<String>,
    val warnings: List<String>
)

/**
 * Result of cleanup operation
 */
data class CleanupResult(
    val success: Boolean,
    val deletedClassifications: Int,
    val deletedCacheEntries: Int,
    val message: String
)

/**
 * Result of synchronization operation
 */
data class SyncResult(
    val success: Boolean,
    val syncedCount: Int,
    val message: String
)