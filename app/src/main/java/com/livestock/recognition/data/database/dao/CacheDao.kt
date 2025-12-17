package com.livestock.recognition.data.database.dao

import androidx.room.*
import com.livestock.recognition.data.database.entities.CacheEntity
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for cache management
 */
@Dao
interface CacheDao {
    
    @Query("SELECT * FROM cache ORDER BY last_accessed DESC")
    fun getAllCacheEntries(): Flow<List<CacheEntity>>
    
    @Query("SELECT * FROM cache WHERE cache_key = :key")
    suspend fun getCacheEntry(key: String): CacheEntity?
    
    @Query("SELECT * FROM cache WHERE cache_type = :type ORDER BY last_accessed DESC")
    suspend fun getCacheEntriesByType(type: String): List<CacheEntity>
    
    @Query("SELECT * FROM cache WHERE expires_at IS NULL OR expires_at > :currentTime")
    suspend fun getValidCacheEntries(currentTime: Long = System.currentTimeMillis()): List<CacheEntity>
    
    @Query("SELECT * FROM cache WHERE expires_at IS NOT NULL AND expires_at <= :currentTime")
    suspend fun getExpiredCacheEntries(currentTime: Long = System.currentTimeMillis()): List<CacheEntity>
    
    @Query("SELECT COUNT(*) FROM cache")
    suspend fun getCacheEntryCount(): Int
    
    @Query("SELECT COUNT(*) FROM cache WHERE cache_type = :type")
    suspend fun getCacheEntryCountByType(type: String): Int
    
    @Query("SELECT SUM(size_bytes) FROM cache")
    suspend fun getTotalCacheSize(): Long?
    
    @Query("SELECT SUM(size_bytes) FROM cache WHERE cache_type = :type")
    suspend fun getCacheSizeByType(type: String): Long?
    
    @Query("SELECT DISTINCT cache_type FROM cache")
    suspend fun getAllCacheTypes(): List<String>
    
    @Query("SELECT * FROM cache ORDER BY access_count DESC LIMIT :limit")
    suspend fun getMostAccessedEntries(limit: Int): List<CacheEntity>
    
    @Query("SELECT * FROM cache ORDER BY last_accessed ASC LIMIT :limit")
    suspend fun getLeastRecentlyUsedEntries(limit: Int): List<CacheEntity>
    
    @Query("SELECT * FROM cache ORDER BY size_bytes DESC LIMIT :limit")
    suspend fun getLargestCacheEntries(limit: Int): List<CacheEntity>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCacheEntry(cacheEntry: CacheEntity)
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCacheEntries(cacheEntries: List<CacheEntity>)
    
    @Update
    suspend fun updateCacheEntry(cacheEntry: CacheEntity)
    
    @Query("UPDATE cache SET access_count = access_count + 1, last_accessed = :accessTime WHERE cache_key = :key")
    suspend fun incrementAccessCount(key: String, accessTime: Long = System.currentTimeMillis())
    
    @Query("UPDATE cache SET cache_value = :value, last_accessed = :accessTime WHERE cache_key = :key")
    suspend fun updateCacheValue(key: String, value: String, accessTime: Long = System.currentTimeMillis())
    
    @Delete
    suspend fun deleteCacheEntry(cacheEntry: CacheEntity)
    
    @Query("DELETE FROM cache WHERE cache_key = :key")
    suspend fun deleteCacheEntryByKey(key: String)
    
    @Query("DELETE FROM cache WHERE cache_type = :type")
    suspend fun deleteCacheEntriesByType(type: String)
    
    @Query("DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= :currentTime")
    suspend fun deleteExpiredEntries(currentTime: Long = System.currentTimeMillis()): Int
    
    @Query("DELETE FROM cache")
    suspend fun deleteAllCacheEntries()
    
    @Query("DELETE FROM cache WHERE cache_key IN (SELECT cache_key FROM cache ORDER BY last_accessed ASC LIMIT :count)")
    suspend fun deleteLeastRecentlyUsedEntries(count: Int): Int
    
    @Query("DELETE FROM cache WHERE size_bytes > :maxSize")
    suspend fun deleteLargeCacheEntries(maxSize: Long): Int
    
    @Query("SELECT * FROM cache WHERE created_at < :cutoffTime")
    suspend fun getOldCacheEntries(cutoffTime: Long): List<CacheEntity>
    
    @Query("DELETE FROM cache WHERE created_at < :cutoffTime")
    suspend fun deleteOldCacheEntries(cutoffTime: Long): Int
}