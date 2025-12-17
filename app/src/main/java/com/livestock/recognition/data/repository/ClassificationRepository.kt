package com.livestock.recognition.data.repository

import androidx.lifecycle.LiveData
import com.livestock.recognition.data.database.dao.ClassificationDao
import com.livestock.recognition.data.database.entities.ClassificationEntity
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.models.ClassificationResult
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for managing classification data
 * Provides abstraction layer between database and UI/business logic
 */
@Singleton
class ClassificationRepository @Inject constructor(
    private val classificationDao: ClassificationDao
) {
    
    /**
     * Get all classifications as Flow
     */
    fun getAllClassifications(): Flow<List<ClassificationResult>> {
        return classificationDao.getAllClassifications().map { entities ->
            entities.map { it.toClassificationResult() }
        }
    }
    
    /**
     * Get all classifications as LiveData
     */
    fun getAllClassificationsLiveData(): LiveData<List<ClassificationEntity>> {
        return classificationDao.getAllClassificationsLiveData()
    }
    
    /**
     * Get classification by ID
     */
    suspend fun getClassificationById(id: Long): ClassificationResult? {
        return classificationDao.getClassificationById(id)?.toClassificationResult()
    }
    
    /**
     * Get classifications by breed
     */
    suspend fun getClassificationsByBreed(breed: String): List<ClassificationResult> {
        return classificationDao.getClassificationsByBreed(breed).map { it.toClassificationResult() }
    }
    
    /**
     * Get classifications by animal type
     */
    suspend fun getClassificationsByType(animalType: AnimalType): List<ClassificationResult> {
        return classificationDao.getClassificationsByType(animalType).map { it.toClassificationResult() }
    }
    
    /**
     * Get unsynced classifications for cloud synchronization
     */
    suspend fun getUnsyncedClassifications(): List<ClassificationResult> {
        return classificationDao.getUnsyncedClassifications().map { it.toClassificationResult() }
    }
    
    /**
     * Get classifications within time range
     */
    suspend fun getClassificationsByTimeRange(startTime: Long, endTime: Long): List<ClassificationResult> {
        return classificationDao.getClassificationsByTimeRange(startTime, endTime).map { it.toClassificationResult() }
    }
    
    /**
     * Get high confidence classifications
     */
    suspend fun getHighConfidenceClassifications(minConfidence: Float = 0.8f): List<ClassificationResult> {
        return classificationDao.getHighConfidenceClassifications(minConfidence).map { it.toClassificationResult() }
    }
    
    /**
     * Get total classification count
     */
    suspend fun getClassificationCount(): Int {
        return classificationDao.getClassificationCount()
    }
    
    /**
     * Get classification count by breed
     */
    suspend fun getClassificationCountByBreed(breed: String): Int {
        return classificationDao.getClassificationCountByBreed(breed)
    }
    
    /**
     * Get classification count by type
     */
    suspend fun getClassificationCountByType(animalType: AnimalType): Int {
        return classificationDao.getClassificationCountByType(animalType)
    }
    
    /**
     * Get unsynced count
     */
    suspend fun getUnsyncedCount(): Int {
        return classificationDao.getUnsyncedCount()
    }
    
    /**
     * Get average confidence by breed
     */
    suspend fun getAverageConfidenceByBreed(breed: String): Float {
        return classificationDao.getAverageConfidenceByBreed(breed) ?: 0.0f
    }
    
    /**
     * Get average processing time
     */
    suspend fun getAverageProcessingTime(): Float {
        return classificationDao.getAverageProcessingTime() ?: 0.0f
    }
    
    /**
     * Get breed statistics
     */
    suspend fun getBreedStatistics(): List<ClassificationDao.BreedStatistic> {
        return classificationDao.getBreedStatistics()
    }
    
    /**
     * Save classification result
     */
    suspend fun saveClassification(classificationResult: ClassificationResult): Long {
        val entity = classificationResult.toEntity()
        return classificationDao.insertClassification(entity)
    }
    
    /**
     * Save multiple classification results
     */
    suspend fun saveClassifications(classificationResults: List<ClassificationResult>): List<Long> {
        val entities = classificationResults.map { it.toEntity() }
        return classificationDao.insertClassifications(entities)
    }
    
    /**
     * Update classification
     */
    suspend fun updateClassification(classificationResult: ClassificationResult) {
        val entity = classificationResult.toEntity()
        classificationDao.updateClassification(entity)
    }
    
    /**
     * Mark classification as synced
     */
    suspend fun markAsSynced(id: Long) {
        classificationDao.markAsSynced(id)
    }
    
    /**
     * Mark multiple classifications as synced
     */
    suspend fun markAsSynced(ids: List<Long>) {
        classificationDao.markAsSynced(ids)
    }
    
    /**
     * Delete classification
     */
    suspend fun deleteClassification(classificationResult: ClassificationResult) {
        val entity = classificationResult.toEntity()
        classificationDao.deleteClassification(entity)
    }
    
    /**
     * Delete classification by ID
     */
    suspend fun deleteClassificationById(id: Long) {
        classificationDao.deleteClassificationById(id)
    }
    
    /**
     * Delete old classifications
     */
    suspend fun deleteOldClassifications(cutoffTime: Long): Int {
        return classificationDao.deleteOldClassifications(cutoffTime)
    }
    
    /**
     * Delete all classifications
     */
    suspend fun deleteAllClassifications() {
        classificationDao.deleteAllClassifications()
    }
    
    /**
     * Delete synced old classifications
     */
    suspend fun deleteSyncedOldClassifications(cutoffTime: Long): Int {
        return classificationDao.deleteSyncedOldClassifications(cutoffTime)
    }
    
    /**
     * Clean up old data based on retention policy
     */
    suspend fun cleanupOldData(retentionDays: Int = 30): Int {
        val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
        return deleteSyncedOldClassifications(cutoffTime)
    }
}

/**
 * Extension function to convert ClassificationEntity to ClassificationResult
 */
private fun ClassificationEntity.toClassificationResult(): ClassificationResult {
    return ClassificationResult(
        breed = this.breed,
        breedConfidence = this.breedConfidence,
        animalType = this.animalType,
        typeConfidence = this.typeConfidence,
        timestamp = this.timestamp,
        imageMetadata = null, // Would need to parse from stored data
        processingTime = this.processingTime
    )
}

/**
 * Extension function to convert ClassificationResult to ClassificationEntity
 */
private fun ClassificationResult.toEntity(): ClassificationEntity {
    return ClassificationEntity(
        breed = this.breed,
        breedConfidence = this.breedConfidence,
        animalType = this.animalType,
        typeConfidence = this.typeConfidence,
        imagePath = "", // Would need to be provided
        timestamp = this.timestamp,
        processingTime = this.processingTime,
        synced = false
    )
}