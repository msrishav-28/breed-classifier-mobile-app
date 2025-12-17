package com.livestock.recognition.data.database.dao

import androidx.room.*
import androidx.lifecycle.LiveData
import com.livestock.recognition.data.database.entities.BreedMappingEntity
import com.livestock.recognition.data.models.AnimalType
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for breed mapping data
 */
@Dao
interface BreedMappingDao {
    
    @Query("SELECT * FROM breed_mapping ORDER BY breed_name ASC")
    fun getAllBreedMappings(): Flow<List<BreedMappingEntity>>
    
    @Query("SELECT * FROM breed_mapping ORDER BY breed_name ASC")
    fun getAllBreedMappingsLiveData(): LiveData<List<BreedMappingEntity>>
    
    @Query("SELECT * FROM breed_mapping WHERE breed_name = :breedName")
    suspend fun getBreedMapping(breedName: String): BreedMappingEntity?
    
    @Query("SELECT * FROM breed_mapping WHERE animal_type = :animalType ORDER BY breed_name ASC")
    suspend fun getBreedMappingsByType(animalType: AnimalType): List<BreedMappingEntity>
    
    @Query("SELECT * FROM breed_mapping WHERE primary_use = :primaryUse ORDER BY breed_name ASC")
    suspend fun getBreedMappingsByUse(primaryUse: String): List<BreedMappingEntity>
    
    @Query("SELECT * FROM breed_mapping WHERE origin = :origin ORDER BY breed_name ASC")
    suspend fun getBreedMappingsByOrigin(origin: String): List<BreedMappingEntity>
    
    @Query("SELECT * FROM breed_mapping WHERE milk_yield_min >= :minYield AND milk_yield_max <= :maxYield ORDER BY breed_name ASC")
    suspend fun getBreedMappingsByMilkYieldRange(minYield: Int, maxYield: Int): List<BreedMappingEntity>
    
    @Query("SELECT * FROM breed_mapping WHERE average_weight BETWEEN :minWeight AND :maxWeight ORDER BY breed_name ASC")
    suspend fun getBreedMappingsByWeightRange(minWeight: Int, maxWeight: Int): List<BreedMappingEntity>
    
    @Query("SELECT DISTINCT animal_type FROM breed_mapping")
    suspend fun getAllAnimalTypes(): List<AnimalType>
    
    @Query("SELECT DISTINCT origin FROM breed_mapping WHERE origin IS NOT NULL")
    suspend fun getAllOrigins(): List<String>
    
    @Query("SELECT DISTINCT primary_use FROM breed_mapping WHERE primary_use IS NOT NULL")
    suspend fun getAllPrimaryUses(): List<String>
    
    @Query("SELECT COUNT(*) FROM breed_mapping")
    suspend fun getBreedMappingCount(): Int
    
    @Query("SELECT COUNT(*) FROM breed_mapping WHERE animal_type = :animalType")
    suspend fun getBreedMappingCountByType(animalType: AnimalType): Int
    
    @Query("SELECT breed_name FROM breed_mapping WHERE animal_type = :animalType ORDER BY breed_name ASC")
    suspend fun getBreedNamesByType(animalType: AnimalType): List<String>
    
    @Query("SELECT breed_name FROM breed_mapping ORDER BY breed_name ASC")
    suspend fun getAllBreedNames(): List<String>
    
    @Query("SELECT * FROM breed_mapping WHERE breed_name LIKE '%' || :searchTerm || '%' OR scientific_name LIKE '%' || :searchTerm || '%' ORDER BY breed_name ASC")
    suspend fun searchBreedMappings(searchTerm: String): List<BreedMappingEntity>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertBreedMapping(breedMapping: BreedMappingEntity)
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertBreedMappings(breedMappings: List<BreedMappingEntity>)
    
    @Update
    suspend fun updateBreedMapping(breedMapping: BreedMappingEntity)
    
    @Delete
    suspend fun deleteBreedMapping(breedMapping: BreedMappingEntity)
    
    @Query("DELETE FROM breed_mapping WHERE breed_name = :breedName")
    suspend fun deleteBreedMappingByName(breedName: String)
    
    @Query("DELETE FROM breed_mapping")
    suspend fun deleteAllBreedMappings()
    
    @Query("UPDATE breed_mapping SET last_updated = :timestamp WHERE breed_name = :breedName")
    suspend fun updateLastUpdated(breedName: String, timestamp: Long)
    
    @Query("SELECT * FROM breed_mapping WHERE last_updated < :cutoffTime")
    suspend fun getOutdatedBreedMappings(cutoffTime: Long): List<BreedMappingEntity>
}