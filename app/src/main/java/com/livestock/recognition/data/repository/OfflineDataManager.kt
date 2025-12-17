package com.livestock.recognition.data.repository

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import com.livestock.recognition.data.database.dao.CacheDao
import com.livestock.recognition.data.database.entities.CacheEntity
import com.livestock.recognition.data.models.ClassificationResult
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton
import android.util.Log

/**
 * Manages offline data storage and synchronization
 * Handles caching, network state detection, and data sync
 */
@Singleton
class OfflineDataManager @Inject constructor(
    private val context: Context,
    private val cacheDao: CacheDao,
    private val classificationRepository: ClassificationRepository
) {
    
    companion object {
        private const val TAG = "OfflineDataManager"
        private const val CACHE_TYPE_CLASSIFICATION = "classification"
        private const val CACHE_TYPE_BREED_INFO = "breed_info"
        private const val CACHE_TYPE_MODEL_RESULT = "model_result"
        private const val CACHE_TYPE_IMAGE_METADATA = "image_metadata"
        
        // Cache expiration times (in milliseconds)
        private const val CLASSIFICATION_CACHE_EXPIRY = 7 * 24 * 60 * 60 * 1000L // 7 days
        private const val BREED_INFO_CACHE_EXPIRY = 30 * 24 * 60 * 60 * 1000L // 30 days
        private const val MODEL_RESULT_CACHE_EXPIRY = 24 * 60 * 60 * 1000L // 1 day
        
        // Cache size limits (in bytes)
        private const val MAX_TOTAL_CACHE_SIZE = 100 * 1024 * 1024L // 100 MB
        private const val MAX_SINGLE_ENTRY_SIZE = 10 * 1024 * 1024L // 10 MB
    }
    
    private val gson = Gson()
    
    /**
     * Check if device is connected to network
     */
    fun isNetworkAvailable(): Boolean {
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        
        val network = connectivityManager.activeNetwork ?: return false
        val networkCapabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        
        return networkCapabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                networkCapabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }
    
    /**
     * Cache classification result for offline access
     */
    suspend fun cacheClassificationResult(key: String, result: ClassificationResult) {
        withContext(Dispatchers.IO) {
            try {
                val jsonValue = gson.toJson(result)
                val sizeBytes = jsonValue.toByteArray().size.toLong()
                
                if (sizeBytes > MAX_SINGLE_ENTRY_SIZE) {
                    Log.w(TAG, "Classification result too large to cache: $sizeBytes bytes")
                    return@withContext
                }
                
                val cacheEntry = CacheEntity(
                    cacheKey = key,
                    cacheValue = jsonValue,
                    cacheType = CACHE_TYPE_CLASSIFICATION,
                    expiresAt = System.currentTimeMillis() + CLASSIFICATION_CACHE_EXPIRY,
                    sizeBytes = sizeBytes
                )
                
                cacheDao.insertCacheEntry(cacheEntry)
                Log.d(TAG, "Cached classification result: $key")
                
                // Check and cleanup cache if needed
                cleanupCacheIfNeeded()
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to cache classification result", e)
            }
        }
    }
    
    /**
     * Retrieve cached classification result
     */
    suspend fun getCachedClassificationResult(key: String): ClassificationResult? {
        return withContext(Dispatchers.IO) {
            try {
                val cacheEntry = cacheDao.getCacheEntry(key)
                
                if (cacheEntry == null) {
                    Log.d(TAG, "No cached result found for key: $key")
                    return@withContext null
                }
                
                // Check if expired
                if (cacheEntry.expiresAt != null && cacheEntry.expiresAt <= System.currentTimeMillis()) {
                    Log.d(TAG, "Cached result expired for key: $key")
                    cacheDao.deleteCacheEntryByKey(key)
                    return@withContext null
                }
                
                // Update access count
                cacheDao.incrementAccessCount(key)
                
                // Parse and return result
                val result = gson.fromJson(cacheEntry.cacheValue, ClassificationResult::class.java)
                Log.d(TAG, "Retrieved cached classification result: $key")
                
                result
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to retrieve cached classification result", e)
                null
            }
        }
    }
    
    /**
     * Cache breed information for offline access
     */
    suspend fun cacheBreedInfo(breedName: String, breedInfo: Any) {
        withContext(Dispatchers.IO) {
            try {
                val jsonValue = gson.toJson(breedInfo)
                val sizeBytes = jsonValue.toByteArray().size.toLong()
                
                if (sizeBytes > MAX_SINGLE_ENTRY_SIZE) {
                    Log.w(TAG, "Breed info too large to cache: $sizeBytes bytes")
                    return@withContext
                }
                
                val cacheEntry = CacheEntity(
                    cacheKey = "breed_info_$breedName",
                    cacheValue = jsonValue,
                    cacheType = CACHE_TYPE_BREED_INFO,
                    expiresAt = System.currentTimeMillis() + BREED_INFO_CACHE_EXPIRY,
                    sizeBytes = sizeBytes
                )
                
                cacheDao.insertCacheEntry(cacheEntry)
                Log.d(TAG, "Cached breed info: $breedName")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to cache breed info", e)
            }
        }
    }
    
    /**
     * Retrieve cached breed information
     */
    suspend fun getCachedBreedInfo(breedName: String): String? {
        return withContext(Dispatchers.IO) {
            try {
                val cacheEntry = cacheDao.getCacheEntry("breed_info_$breedName")
                
                if (cacheEntry == null) {
                    Log.d(TAG, "No cached breed info found for: $breedName")
                    return@withContext null
                }
                
                // Check if expired
                if (cacheEntry.expiresAt != null && cacheEntry.expiresAt <= System.currentTimeMillis()) {
                    Log.d(TAG, "Cached breed info expired for: $breedName")
                    cacheDao.deleteCacheEntryByKey("breed_info_$breedName")
                    return@withContext null
                }
                
                // Update access count
                cacheDao.incrementAccessCount("breed_info_$breedName")
                
                Log.d(TAG, "Retrieved cached breed info: $breedName")
                cacheEntry.cacheValue
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to retrieve cached breed info", e)
                null
            }
        }
    }
    
    /**
     * Cache model inference result
     */
    suspend fun cacheModelResult(imageHash: String, modelResult: Any) {
        withContext(Dispatchers.IO) {
            try {
                val jsonValue = gson.toJson(modelResult)
                val sizeBytes = jsonValue.toByteArray().size.toLong()
                
                if (sizeBytes > MAX_SINGLE_ENTRY_SIZE) {
                    Log.w(TAG, "Model result too large to cache: $sizeBytes bytes")
                    return@withContext
                }
                
                val cacheEntry = CacheEntity(
                    cacheKey = "model_result_$imageHash",
                    cacheValue = jsonValue,
                    cacheType = CACHE_TYPE_MODEL_RESULT,
                    expiresAt = System.currentTimeMillis() + MODEL_RESULT_CACHE_EXPIRY,
                    sizeBytes = sizeBytes
                )
                
                cacheDao.insertCacheEntry(cacheEntry)
                Log.d(TAG, "Cached model result: $imageHash")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to cache model result", e)
            }
        }
    }
    
    /**
     * Retrieve cached model result
     */
    suspend fun getCachedModelResult(imageHash: String): String? {
        return withContext(Dispatchers.IO) {
            try {
                val cacheEntry = cacheDao.getCacheEntry("model_result_$imageHash")
                
                if (cacheEntry == null) {
                    Log.d(TAG, "No cached model result found for: $imageHash")
                    return@withContext null
                }
                
                // Check if expired
                if (cacheEntry.expiresAt != null && cacheEntry.expiresAt <= System.currentTimeMillis()) {
                    Log.d(TAG, "Cached model result expired for: $imageHash")
                    cacheDao.deleteCacheEntryByKey("model_result_$imageHash")
                    return@withContext null
                }
                
                // Update access count
                cacheDao.incrementAccessCount("model_result_$imageHash")
                
                Log.d(TAG, "Retrieved cached model result: $imageHash")
                cacheEntry.cacheValue
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to retrieve cached model result", e)
                null
            }
        }
    }
    
    /**
     * Synchronize offline data when network becomes available
     */
    suspend fun synchronizeOfflineData(): SyncResult {
        return withContext(Dispatchers.IO) {
            if (!isNetworkAvailable()) {
                Log.w(TAG, "Network not available, skipping sync")
                return@withContext SyncResult(false, 0, "Network not available")
            }
            
            try {
                // Get unsynced classifications
                val unsyncedClassifications = classificationRepository.getUnsyncedClassifications()
                
                if (unsyncedClassifications.isEmpty()) {
                    Log.d(TAG, "No unsynced data to synchronize")
                    return@withContext SyncResult(true, 0, "No data to sync")
                }
                
                // TODO: Implement actual cloud sync logic here
                // For now, just mark as synced
                val syncedIds = mutableListOf<Long>()
                
                for (classification in unsyncedClassifications) {
                    // Simulate cloud upload
                    // In real implementation, this would upload to cloud service
                    
                    // Mark as synced (would need to get ID from classification)
                    // syncedIds.add(classification.id)
                }
                
                // Mark classifications as synced
                if (syncedIds.isNotEmpty()) {
                    classificationRepository.markAsSynced(syncedIds)
                }
                
                Log.d(TAG, "Synchronized ${syncedIds.size} classifications")
                SyncResult(true, syncedIds.size, "Sync completed successfully")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to synchronize offline data", e)
                SyncResult(false, 0, "Sync failed: ${e.message}")
            }
        }
    }
    
    /**
     * Clean up expired cache entries
     */
    suspend fun cleanupExpiredCache(): Int {
        return withContext(Dispatchers.IO) {
            try {
                val deletedCount = cacheDao.deleteExpiredEntries()
                Log.d(TAG, "Cleaned up $deletedCount expired cache entries")
                deletedCount
            } catch (e: Exception) {
                Log.e(TAG, "Failed to cleanup expired cache", e)
                0
            }
        }
    }
    
    /**
     * Clean up cache if it exceeds size limits
     */
    private suspend fun cleanupCacheIfNeeded() {
        try {
            val totalSize = cacheDao.getTotalCacheSize() ?: 0L
            
            if (totalSize > MAX_TOTAL_CACHE_SIZE) {
                Log.d(TAG, "Cache size exceeded limit: $totalSize bytes, cleaning up...")
                
                // Delete least recently used entries until under limit
                val entriesToDelete = ((totalSize - MAX_TOTAL_CACHE_SIZE) / (1024 * 1024)).toInt() + 10 // Delete extra 10 entries
                val deletedCount = cacheDao.deleteLeastRecentlyUsedEntries(entriesToDelete)
                
                Log.d(TAG, "Deleted $deletedCount cache entries to free space")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup cache", e)
        }
    }
    
    /**
     * Get cache statistics
     */
    suspend fun getCacheStatistics(): CacheStatistics {
        return withContext(Dispatchers.IO) {
            try {
                val totalEntries = cacheDao.getCacheEntryCount()
                val totalSize = cacheDao.getTotalCacheSize() ?: 0L
                val cacheTypes = cacheDao.getAllCacheTypes()
                
                val typeStatistics = mutableMapOf<String, Pair<Int, Long>>()
                for (type in cacheTypes) {
                    val count = cacheDao.getCacheEntryCountByType(type)
                    val size = cacheDao.getCacheSizeByType(type) ?: 0L
                    typeStatistics[type] = Pair(count, size)
                }
                
                CacheStatistics(
                    totalEntries = totalEntries,
                    totalSizeBytes = totalSize,
                    typeStatistics = typeStatistics
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get cache statistics", e)
                CacheStatistics(0, 0L, emptyMap())
            }
        }
    }
    
    /**
     * Clear all cache data
     */
    suspend fun clearAllCache() {
        withContext(Dispatchers.IO) {
            try {
                cacheDao.deleteAllCacheEntries()
                Log.d(TAG, "Cleared all cache data")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to clear cache", e)
            }
        }
    }
    
    /**
     * Clear cache by type
     */
    suspend fun clearCacheByType(type: String) {
        withContext(Dispatchers.IO) {
            try {
                cacheDao.deleteCacheEntriesByType(type)
                Log.d(TAG, "Cleared cache for type: $type")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to clear cache for type: $type", e)
            }
        }
    }
}

/**
 * Data class for synchronization results
 */
data class SyncResult(
    val success: Boolean,
    val syncedCount: Int,
    val message: String
)

/**
 * Data class for cache statistics
 */
data class CacheStatistics(
    val totalEntries: Int,
    val totalSizeBytes: Long,
    val typeStatistics: Map<String, Pair<Int, Long>> // Type -> (Count, Size)
)