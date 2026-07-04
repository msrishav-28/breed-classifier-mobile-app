package com.livestock.recognition.data.db

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey
import com.livestock.recognition.core.model.AnimalType
import com.livestock.recognition.core.model.ClassificationRecord
import com.livestock.recognition.core.model.Prediction

/**
 * Persisted classification. Alternatives are stored in a compact
 * `label=confidence;label=confidence` encoding to avoid dragging a JSON
 * library into the app for a two-element list.
 */
@Entity(tableName = "classifications")
data class ClassificationEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "breed_label")
    val breedLabel: String,

    @ColumnInfo(name = "confidence")
    val confidence: Float,

    @ColumnInfo(name = "animal_type")
    val animalType: String?,

    @ColumnInfo(name = "alternatives")
    val alternatives: String,

    @ColumnInfo(name = "image_path")
    val imagePath: String,

    @ColumnInfo(name = "captured_at")
    val capturedAt: Long,

    @ColumnInfo(name = "processing_time_ms")
    val processingTimeMs: Long,

    @ColumnInfo(name = "model_version")
    val modelVersion: String?,
)

fun ClassificationRecord.toEntity(imagePath: String): ClassificationEntity =
    ClassificationEntity(
        breedLabel = breedLabel,
        confidence = confidence,
        animalType = animalType?.name,
        alternatives = encodeAlternatives(alternatives),
        imagePath = imagePath,
        capturedAt = capturedAtEpochMillis,
        processingTimeMs = processingTimeMillis,
        modelVersion = modelVersion,
    )

fun ClassificationEntity.toRecord(): ClassificationRecord =
    ClassificationRecord(
        breedLabel = breedLabel,
        confidence = confidence.coerceIn(0f, 1f),
        animalType = animalType?.let { name -> AnimalType.entries.firstOrNull { it.name == name } },
        alternatives = decodeAlternatives(alternatives),
        capturedAtEpochMillis = capturedAt,
        processingTimeMillis = processingTimeMs,
        modelVersion = modelVersion,
    )

internal fun encodeAlternatives(alternatives: List<Prediction>): String =
    alternatives.joinToString(ENTRY_SEPARATOR) { "${it.label}$FIELD_SEPARATOR${it.confidence}" }

internal fun decodeAlternatives(encoded: String): List<Prediction> =
    encoded.split(ENTRY_SEPARATOR)
        .filter { it.isNotBlank() }
        .mapNotNull { entry ->
            val label = entry.substringBeforeLast(FIELD_SEPARATOR)
            val confidence = entry.substringAfterLast(FIELD_SEPARATOR).toFloatOrNull()
            if (label.isBlank() || confidence == null) {
                null
            } else {
                Prediction(label, confidence.coerceIn(0f, 1f))
            }
        }

private const val ENTRY_SEPARATOR = ";"
private const val FIELD_SEPARATOR = "="
