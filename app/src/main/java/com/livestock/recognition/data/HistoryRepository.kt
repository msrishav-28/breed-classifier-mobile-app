package com.livestock.recognition.data

import com.livestock.recognition.core.model.ClassificationRecord
import com.livestock.recognition.data.db.ClassificationDao
import com.livestock.recognition.data.db.toEntity
import com.livestock.recognition.data.db.toRecord
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * A classification as stored on device, together with the photo it was made
 * from.
 */
data class SavedClassification(
    val id: Long,
    val imagePath: String,
    val record: ClassificationRecord,
)

class HistoryRepository(private val dao: ClassificationDao) {

    fun observeAll(): Flow<List<SavedClassification>> =
        dao.observeAll().map { entities ->
            entities.map { SavedClassification(it.id, it.imagePath, it.toRecord()) }
        }

    suspend fun get(id: Long): SavedClassification? =
        dao.getById(id)?.let { SavedClassification(it.id, it.imagePath, it.toRecord()) }

    suspend fun save(record: ClassificationRecord, imagePath: String): Long =
        dao.insert(record.toEntity(imagePath))

    /** Deletes the row and returns the orphaned image path, if any. */
    suspend fun delete(id: Long): String? {
        val imagePath = dao.getById(id)?.imagePath
        dao.deleteById(id)
        return imagePath
    }

    /** Deletes all rows and returns the orphaned image paths. */
    suspend fun clear(): List<String> {
        val imagePaths = dao.getAllImagePaths()
        dao.deleteAll()
        return imagePaths
    }
}
