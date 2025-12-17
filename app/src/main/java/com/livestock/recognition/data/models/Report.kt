package com.livestock.recognition.data.models

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import java.io.File

/**
 * Data class representing a generated report with metadata.
 */
@Parcelize
data class Report(
    val filePath: String,
    val fileName: String,
    val fileSize: Long,
    val generationTime: Long,
    val classificationResult: ClassificationResult
) : Parcelable {
    /**
     * Validates the report data integrity.
     * @return ValidationResult indicating if the report data is valid
     */
    fun validate(): ValidationResult {
        val errors = mutableListOf<String>()

        if (filePath.isBlank()) {
            errors.add("File path cannot be empty")
        }

        if (fileName.isBlank()) {
            errors.add("File name cannot be empty")
        }

        if (fileSize < 0) {
            errors.add("File size cannot be negative")
        }

        if (generationTime <= 0) {
            errors.add("Generation time must be positive")
        }

        val resultValidation = classificationResult.validate()
        if (!resultValidation.isValid) {
            errors.addAll(resultValidation.errors.map { "Classification result: $it" })
        }

        return ValidationResult(errors.isEmpty(), errors)
    }

    /**
     * Checks if the report file exists on the filesystem.
     * @return true if file exists, false otherwise
     */
    fun fileExists(): Boolean {
        return File(filePath).exists()
    }

    /**
     * Gets the file extension from the filename.
     * @return file extension or empty string if none
     */
    fun getFileExtension(): String {
        return fileName.substringAfterLast('.', "")
    }
}