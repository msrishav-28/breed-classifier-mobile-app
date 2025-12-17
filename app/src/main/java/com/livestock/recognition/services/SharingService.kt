package com.livestock.recognition.services

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import com.livestock.recognition.data.models.Report
import com.livestock.recognition.data.models.ValidationResult
import java.io.File

/**
 * Service for sharing reports and files across multiple platforms.
 * Handles email, messaging apps, and file transfer with secure file provider.
 */
class SharingService(private val context: Context) {

    companion object {
        private const val FILE_PROVIDER_AUTHORITY = ".fileprovider"
    }

    /**
     * Shares a report file via the specified sharing method.
     * @param report The Report object to share
     * @param method The sharing method to use
     * @return ShareResult indicating success or failure with details
     */
    fun shareReport(report: Report, method: ShareMethod): ShareResult {
        return try {
            val file = File(report.filePath)
            if (!file.exists()) {
                return ShareResult(false, "Report file not found: ${report.filePath}", null)
            }

            val uri = getSecureFileUri(file)
            val intent = createShareIntent(uri, report, method)

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(Intent.createChooser(intent, getChooserTitle(method)))
                ShareResult(true, "Share intent created successfully", intent)
            } else {
                ShareResult(false, "No app available to handle ${method.displayName} sharing", null)
            }
        } catch (e: Exception) {
            ShareResult(false, "Failed to share report: ${e.message}", null)
        }
    }

    /**
     * Shares multiple files together.
     * @param files List of File objects to share
     * @param method The sharing method to use
     * @param subject Optional subject for the share
     * @param message Optional message body for the share
     * @return ShareResult indicating success or failure
     */
    fun shareMultipleFiles(
        files: List<File>,
        method: ShareMethod,
        subject: String? = null,
        message: String? = null
    ): ShareResult {
        return try {
            if (files.isEmpty()) {
                return ShareResult(false, "No files to share", null)
            }

            val existingFiles = files.filter { it.exists() }
            if (existingFiles.isEmpty()) {
                return ShareResult(false, "None of the specified files exist", null)
            }

            val uris = existingFiles.map { getSecureFileUri(it) }
            val intent = createMultipleShareIntent(uris, method, subject, message)

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(Intent.createChooser(intent, getChooserTitle(method)))
                ShareResult(true, "Multiple files share intent created successfully", intent)
            } else {
                ShareResult(false, "No app available to handle ${method.displayName} sharing", null)
            }
        } catch (e: Exception) {
            ShareResult(false, "Failed to share multiple files: ${e.message}", null)
        }
    }

    /**
     * Validates that a file can be shared.
     * @param file The file to validate
     * @return ValidationResult indicating if the file can be shared
     */
    fun validateFileForSharing(file: File): ValidationResult {
        val errors = mutableListOf<String>()

        if (!file.exists()) {
            errors.add("File does not exist: ${file.absolutePath}")
        }

        if (!file.canRead()) {
            errors.add("File is not readable: ${file.absolutePath}")
        }

        if (file.length() == 0L) {
            errors.add("File is empty: ${file.absolutePath}")
        }

        if (file.length() > 25 * 1024 * 1024) { // 25MB limit for most sharing platforms
            errors.add("File is too large for sharing (>25MB): ${file.absolutePath}")
        }

        return ValidationResult(errors.isEmpty(), errors)
    }

    /**
     * Gets available sharing methods on the device.
     * @return List of ShareMethod objects that are available
     */
    fun getAvailableSharingMethods(): List<ShareMethod> {
        val availableMethods = mutableListOf<ShareMethod>()

        ShareMethod.values().forEach { method ->
            val testIntent = createTestIntent(method)
            if (testIntent.resolveActivity(context.packageManager) != null) {
                availableMethods.add(method)
            }
        }

        return availableMethods
    }

    /**
     * Creates a secure file URI using FileProvider.
     * @param file The file to create URI for
     * @return Uri object for secure file access
     */
    private fun getSecureFileUri(file: File): Uri {
        return FileProvider.getUriForFile(
            context,
            context.packageName + FILE_PROVIDER_AUTHORITY,
            file
        )
    }

    /**
     * Creates a share intent for a single report.
     * @param uri The secure file URI
     * @param report The report being shared
     * @param method The sharing method
     * @return Intent configured for sharing
     */
    private fun createShareIntent(uri: Uri, report: Report, method: ShareMethod): Intent {
        val intent = Intent().apply {
            action = Intent.ACTION_SEND
            type = "application/pdf"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        // Add method-specific configurations
        when (method) {
            ShareMethod.EMAIL -> {
                intent.type = "message/rfc822"
                intent.putExtra(Intent.EXTRA_SUBJECT, "Livestock Recognition Report - ${report.classificationResult.breed}")
                intent.putExtra(Intent.EXTRA_TEXT, createEmailBody(report))
            }
            ShareMethod.MESSAGING -> {
                intent.putExtra(Intent.EXTRA_TEXT, "Livestock recognition report for ${report.classificationResult.breed}")
            }
            ShareMethod.FILE_TRANSFER -> {
                intent.putExtra(Intent.EXTRA_TITLE, report.fileName)
            }
            ShareMethod.GENERAL -> {
                intent.putExtra(Intent.EXTRA_SUBJECT, "Livestock Recognition Report")
                intent.putExtra(Intent.EXTRA_TEXT, "Livestock recognition report for ${report.classificationResult.breed}")
            }
        }

        return intent
    }

    /**
     * Creates a share intent for multiple files.
     * @param uris List of secure file URIs
     * @param method The sharing method
     * @param subject Optional subject
     * @param message Optional message
     * @return Intent configured for sharing multiple files
     */
    private fun createMultipleShareIntent(
        uris: List<Uri>,
        method: ShareMethod,
        subject: String?,
        message: String?
    ): Intent {
        val intent = Intent().apply {
            action = Intent.ACTION_SEND_MULTIPLE
            type = "application/pdf"
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        when (method) {
            ShareMethod.EMAIL -> {
                intent.type = "message/rfc822"
                intent.putExtra(Intent.EXTRA_SUBJECT, subject ?: "Livestock Recognition Reports")
                intent.putExtra(Intent.EXTRA_TEXT, message ?: "Multiple livestock recognition reports attached.")
            }
            ShareMethod.MESSAGING -> {
                intent.putExtra(Intent.EXTRA_TEXT, message ?: "Multiple livestock recognition reports")
            }
            ShareMethod.FILE_TRANSFER -> {
                intent.putExtra(Intent.EXTRA_TITLE, subject ?: "Livestock Reports")
            }
            ShareMethod.GENERAL -> {
                intent.putExtra(Intent.EXTRA_SUBJECT, subject ?: "Livestock Recognition Reports")
                intent.putExtra(Intent.EXTRA_TEXT, message ?: "Multiple livestock recognition reports")
            }
        }

        return intent
    }

    /**
     * Creates a test intent to check if sharing method is available.
     * @param method The sharing method to test
     * @return Intent for testing availability
     */
    private fun createTestIntent(method: ShareMethod): Intent {
        return when (method) {
            ShareMethod.EMAIL -> Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("mailto:")
            }
            ShareMethod.MESSAGING -> Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
            }
            ShareMethod.FILE_TRANSFER -> Intent(Intent.ACTION_SEND).apply {
                type = "application/pdf"
            }
            ShareMethod.GENERAL -> Intent(Intent.ACTION_SEND).apply {
                type = "*/*"
            }
        }
    }

    /**
     * Creates an email body for report sharing.
     * @param report The report to create email body for
     * @return Formatted email body string
     */
    private fun createEmailBody(report: Report): String {
        val result = report.classificationResult
        return """
            Livestock Recognition Report
            
            Classification Details:
            • Breed: ${result.breed}
            • Confidence: ${(result.breedConfidence * 100).toInt()}%
            • Animal Type: ${result.animalType.displayName}
            • Processing Time: ${result.processingTime}ms
            
            Image Information:
            • Resolution: ${result.imageMetadata.width} x ${result.imageMetadata.height}
            • Quality Score: ${(result.imageMetadata.qualityScore * 100).toInt()}%
            • File Size: ${result.imageMetadata.fileSize / 1024} KB
            
            This report was generated by the Comprehensive Cattle and Buffalo Recognition System.
            
            Please find the detailed PDF report attached.
        """.trimIndent()
    }

    /**
     * Gets the appropriate chooser title for the sharing method.
     * @param method The sharing method
     * @return String title for the share chooser
     */
    private fun getChooserTitle(method: ShareMethod): String {
        return when (method) {
            ShareMethod.EMAIL -> "Share report via email"
            ShareMethod.MESSAGING -> "Share report via messaging"
            ShareMethod.FILE_TRANSFER -> "Share report file"
            ShareMethod.GENERAL -> "Share report"
        }
    }
}

/**
 * Enumeration of available sharing methods.
 */
enum class ShareMethod(val displayName: String, val description: String) {
    EMAIL("Email", "Share via email applications"),
    MESSAGING("Messaging", "Share via messaging and chat applications"),
    FILE_TRANSFER("File Transfer", "Share via file transfer applications"),
    GENERAL("General", "Share via any compatible application")
}

/**
 * Result of a sharing operation.
 */
data class ShareResult(
    val success: Boolean,
    val message: String,
    val intent: Intent?
) {
    /**
     * Validates the share result data.
     * @return ValidationResult indicating if the data is valid
     */
    fun validate(): ValidationResult {
        val errors = mutableListOf<String>()

        if (message.isBlank()) {
            errors.add("Share result message cannot be empty")
        }

        if (success && intent == null) {
            errors.add("Successful share result should have an intent")
        }

        return ValidationResult(errors.isEmpty(), errors)
    }

    /**
     * Checks if the sharing was successful.
     * @return true if sharing was successful, false otherwise
     */
    fun isSuccessful(): Boolean = success

    /**
     * Gets a user-friendly message about the sharing result.
     * @return String message for display to user
     */
    fun getUserMessage(): String {
        return if (success) {
            "Report shared successfully"
        } else {
            "Failed to share report: $message"
        }
    }
}