package com.livestock.recognition.data.database.converters

import androidx.room.TypeConverter
import com.livestock.recognition.data.models.AnimalType
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

/**
 * Type converters for Room database
 */
class Converters {
    
    private val gson = Gson()
    
    @TypeConverter
    fun fromAnimalType(animalType: AnimalType): String {
        return animalType.name
    }
    
    @TypeConverter
    fun toAnimalType(animalTypeString: String): AnimalType {
        return try {
            AnimalType.valueOf(animalTypeString)
        } catch (e: IllegalArgumentException) {
            AnimalType.DUAL_PURPOSE // Default fallback
        }
    }
    
    @TypeConverter
    fun fromStringList(value: List<String>): String {
        return gson.toJson(value)
    }
    
    @TypeConverter
    fun toStringList(value: String): List<String> {
        return try {
            val listType = object : TypeToken<List<String>>() {}.type
            gson.fromJson(value, listType) ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    @TypeConverter
    fun fromStringMap(value: Map<String, String>): String {
        return gson.toJson(value)
    }
    
    @TypeConverter
    fun toStringMap(value: String): Map<String, String> {
        return try {
            val mapType = object : TypeToken<Map<String, String>>() {}.type
            gson.fromJson(value, mapType) ?: emptyMap()
        } catch (e: Exception) {
            emptyMap()
        }
    }
    
    @TypeConverter
    fun fromFloatList(value: List<Float>): String {
        return gson.toJson(value)
    }
    
    @TypeConverter
    fun toFloatList(value: String): List<Float> {
        return try {
            val listType = object : TypeToken<List<Float>>() {}.type
            gson.fromJson(value, listType) ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }
}