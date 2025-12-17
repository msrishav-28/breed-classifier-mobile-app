package com.livestock.recognition.data.database

import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import android.content.Context
import com.livestock.recognition.data.database.dao.ClassificationDao
import com.livestock.recognition.data.database.dao.BreedMappingDao
import com.livestock.recognition.data.database.dao.CacheDao
import com.livestock.recognition.data.database.entities.ClassificationEntity
import com.livestock.recognition.data.database.entities.BreedMappingEntity
import com.livestock.recognition.data.database.entities.CacheEntity
import com.livestock.recognition.data.database.converters.Converters

/**
 * Room database for livestock recognition app
 * Handles classifications, breed mapping, and offline caching
 */
@Database(
    entities = [
        ClassificationEntity::class,
        BreedMappingEntity::class,
        CacheEntity::class
    ],
    version = 1,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class LivestockDatabase : RoomDatabase() {
    
    abstract fun classificationDao(): ClassificationDao
    abstract fun breedMappingDao(): BreedMappingDao
    abstract fun cacheDao(): CacheDao
    
    companion object {
        @Volatile
        private var INSTANCE: LivestockDatabase? = null
        
        private const val DATABASE_NAME = "livestock_database"
        
        fun getDatabase(context: Context): LivestockDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    LivestockDatabase::class.java,
                    DATABASE_NAME
                )
                .fallbackToDestructiveMigration()
                .build()
                INSTANCE = instance
                instance
            }
        }
        
        /**
         * For testing purposes - allows providing a custom database instance
         */
        fun getTestDatabase(context: Context): LivestockDatabase {
            return Room.inMemoryDatabaseBuilder(
                context.applicationContext,
                LivestockDatabase::class.java
            ).build()
        }
    }
}