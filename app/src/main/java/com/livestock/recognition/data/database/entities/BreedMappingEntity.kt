package com.livestock.recognition.data.database.entities

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.ColumnInfo
import com.livestock.recognition.data.models.AnimalType

/**
 * Entity for storing breed to type mapping information
 */
@Entity(tableName = "breed_mapping")
data class BreedMappingEntity(
    @PrimaryKey
    @ColumnInfo(name = "breed_name")
    val breedName: String,
    
    @ColumnInfo(name = "animal_type")
    val animalType: AnimalType,
    
    @ColumnInfo(name = "milk_yield_min")
    val milkYieldMin: Int?,
    
    @ColumnInfo(name = "milk_yield_max")
    val milkYieldMax: Int?,
    
    @ColumnInfo(name = "characteristics")
    val characteristics: String, // JSON string of characteristics list
    
    @ColumnInfo(name = "scientific_name")
    val scientificName: String?,
    
    @ColumnInfo(name = "origin")
    val origin: String?,
    
    @ColumnInfo(name = "primary_use")
    val primaryUse: String?,
    
    @ColumnInfo(name = "average_weight")
    val averageWeight: Int?,
    
    @ColumnInfo(name = "description")
    val description: String?,
    
    @ColumnInfo(name = "image_url")
    val imageUrl: String?,
    
    @ColumnInfo(name = "last_updated")
    val lastUpdated: Long = System.currentTimeMillis()
)