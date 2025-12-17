package com.livestock.recognition.data.database.entities

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.ColumnInfo
import com.livestock.recognition.data.models.AnimalType

/**
 * Entity for storing classification results in local database
 */
@Entity(tableName = "classifications")
data class ClassificationEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    
    @ColumnInfo(name = "breed")
    val breed: String,
    
    @ColumnInfo(name = "breed_confidence")
    val breedConfidence: Float,
    
    @ColumnInfo(name = "animal_type")
    val animalType: AnimalType,
    
    @ColumnInfo(name = "type_confidence")
    val typeConfidence: Float,
    
    @ColumnInfo(name = "image_path")
    val imagePath: String,
    
    @ColumnInfo(name = "timestamp")
    val timestamp: Long,
    
    @ColumnInfo(name = "processing_time")
    val processingTime: Long,
    
    @ColumnInfo(name = "synced")
    val synced: Boolean = false,
    
    @ColumnInfo(name = "device_info")
    val deviceInfo: String? = null,
    
    @ColumnInfo(name = "model_version")
    val modelVersion: String? = null,
    
    @ColumnInfo(name = "ensemble_predictions")
    val ensemblePredictions: String? = null // JSON string of individual model predictions
)