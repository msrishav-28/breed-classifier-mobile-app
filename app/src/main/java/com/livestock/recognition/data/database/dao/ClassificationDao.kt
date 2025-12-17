package com.livestock.recognition.data.database.dao

import androidx.room.*
import androidx.lifecycle.LiveData
import com.livestock.recognition.data.database.entities.ClassificationEntity
import com.livestock.recognition.data.models.AnimalType
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for classification results
 */
@Dao
interface ClassificationDao {
    
    @Query("SELECT * FROM classifications ORDER BY timestamp DESC")
    fun getAllClassifications(): Flow<List<ClassificationEntity>>
    
    @Query("SELECT * FROM classifications ORDER BY timestamp DESC")
    fun getAllClassificationsLiveData(): LiveData<List<ClassificationEntity>>
    
    @Query("SELECT * FROM classifications WHERE id = :id")
    suspend fun getClassificationById(id: Long): ClassificationEntity?
    
    @Query("SELECT * FROM classifications WHERE breed = :breed ORDER BY timestamp DESC")
    suspend fun getClassificationsByBreed(breed: String): List<ClassificationEntity>
    
    @Query("SELECT * FROM classifications WHERE animal_type = :animalType ORDER BY timestamp DESC")
    suspend fun getClassificationsByType(animalType: AnimalType): List<ClassificationEntity>
    
    @Query("SELECT * FROM classifications WHERE synced = 0 ORDER BY timestamp ASC")
    suspend fun getUnsyncedClassifications(): List<ClassificationEntity>
    
    @Query("SELECT * FROM classifications WHERE timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp DESC")
    suspend fun getClassificationsByTimeRange(startTime: Long, endTime: Long): List<ClassificationEntity>
    
    @Query("SELECT * FROM classifications WHERE breed_confidence >= :minConfidence ORDER BY timestamp DESC")
    suspend fun getHighConfidenceClassifications(minConfidence: Float): List<ClassificationEntity>
    
    @Query("SELECT COUNT(*) FROM classifications")
    suspend fun getClassificationCount(): Int
    
    @Query("SELECT COUNT(*) FROM classifications WHERE breed = :breed")
    suspend fun getClassificationCountByBreed(breed: String): Int
    
    @Query("SELECT COUNT(*) FROM classifications WHERE animal_type = :animalType")
    suspend fun getClassificationCountByType(animalType: AnimalType): Int
    
    @Query("SELECT COUNT(*) FROM classifications WHERE synced = 0")
    suspend fun getUnsyncedCount(): Int
    
    @Query("SELECT AVG(breed_confidence) FROM classifications WHERE breed = :breed")
    suspend fun getAverageConfidenceByBreed(breed: String): Float?
    
    @Query("SELECT AVG(processing_time) FROM classifications")
    suspend fun getAverageProcessingTime(): Float?
    
    @Query("SELECT breed, COUNT(*) as count FROM classifications GROUP BY breed ORDER BY count DESC")
    suspend fun getBreedStatistics(): List<BreedStatistic>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertClassification(classification: ClassificationEntity): Long
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertClassifications(classifications: List<ClassificationEntity>): List<Long>
    
    @Update
    suspend fun updateClassification(classification: ClassificationEntity)
    
    @Query("UPDATE classifications SET synced = 1 WHERE id = :id")
    suspend fun markAsSynced(id: Long)
    
    @Query("UPDATE classifications SET synced = 1 WHERE id IN (:ids)")
    suspend fun markAsSynced(ids: List<Long>)
    
    @Delete
    suspend fun deleteClassification(classification: ClassificationEntity)
    
    @Query("DELETE FROM classifications WHERE id = :id")
    suspend fun deleteClassificationById(id: Long)
    
    @Query("DELETE FROM classifications WHERE timestamp < :cutoffTime")
    suspend fun deleteOldClassifications(cutoffTime: Long): Int
    
    @Query("DELETE FROM classifications")
    suspend fun deleteAllClassifications()
    
    @Query("DELETE FROM classifications WHERE synced = 1 AND timestamp < :cutoffTime")
    suspend fun deleteSyncedOldClassifications(cutoffTime: Long): Int
    
    // Custom result class for breed statistics
    data class BreedStatistic(
        val breed: String,
        val count: Int
    )
}