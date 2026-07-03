package com.livestock.recognition.image

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * All captured and imported photos live in app-internal storage
 * (`filesDir/images`), which requires no permissions, is excluded from
 * backups and is removed on uninstall.
 */
object ImageFiles {

    private const val IMAGES_DIR = "images"
    private const val FILE_NAME_PATTERN = "yyyyMMdd_HHmmss_SSS"

    fun imagesDir(context: Context): File =
        File(context.filesDir, IMAGES_DIR).apply { mkdirs() }

    fun newImageFile(context: Context): File {
        val name = SimpleDateFormat(FILE_NAME_PATTERN, Locale.US).format(Date())
        return File(imagesDir(context), "IMG_$name.jpg")
    }

    /**
     * Copies a content Uri (e.g. from the photo picker) into app storage so
     * the app keeps access after the picker grant expires. Returns null when
     * the source cannot be read.
     */
    suspend fun copyToAppStorage(context: Context, uri: Uri): File? =
        withContext(Dispatchers.IO) {
            val target = newImageFile(context)
            try {
                context.contentResolver.openInputStream(uri)?.use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                } ?: return@withContext null
                target
            } catch (_: IOException) {
                target.delete()
                null
            } catch (_: SecurityException) {
                target.delete()
                null
            }
        }

    fun delete(path: String?) {
        if (path.isNullOrBlank()) return
        val file = File(path)
        if (file.exists()) file.delete()
    }
}
