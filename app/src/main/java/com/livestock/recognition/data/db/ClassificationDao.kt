package com.livestock.recognition.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ClassificationDao {

    @Query("SELECT * FROM classifications ORDER BY captured_at DESC")
    fun observeAll(): Flow<List<ClassificationEntity>>

    @Query("SELECT * FROM classifications WHERE id = :id")
    suspend fun getById(id: Long): ClassificationEntity?

    @Query("SELECT image_path FROM classifications")
    suspend fun getAllImagePaths(): List<String>

    @Insert
    suspend fun insert(entity: ClassificationEntity): Long

    @Query("DELETE FROM classifications WHERE id = :id")
    suspend fun deleteById(id: Long)

    @Query("DELETE FROM classifications")
    suspend fun deleteAll()
}
