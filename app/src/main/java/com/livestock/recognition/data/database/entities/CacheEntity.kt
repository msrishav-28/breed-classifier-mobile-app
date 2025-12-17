package com.livestock.recognition.data.database.entities

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.ColumnInfo

/**
 * Entity for caching various data for offline operation
 */
@Entity(tableName = "cache")
data class CacheEntity(
    @PrimaryKey
    @ColumnInfo(name = "cache_key")
    val cacheKey: String,
    
    @ColumnInfo(name = "cache_value")
    val cacheValue: String, // JSON string of cached data
    
    @ColumnInfo(name = "cache_type")
    val cacheType: String, // Type of cached data (e.g., "model_result", "breed_info", "image_metadata")
    
    @ColumnInfo(name = "created_at")
    val createdAt: Long = System.currentTimeMillis(),
    
    @ColumnInfo(name = "expires_at")
    val expiresAt: Long? = null,
    
    @ColumnInfo(name = "size_bytes")
    val sizeBytes: Long = 0,
    
    @ColumnInfo(name = "access_count")
    val accessCount: Int = 0,
    
    @ColumnInfo(name = "last_accessed")
    val lastAccessed: Long = System.currentTimeMillis()
)