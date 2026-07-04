package com.livestock.recognition.image

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.IOException

/**
 * Decodes image files with subsampling (to bound memory on large camera
 * captures) and EXIF orientation applied, so portrait photos never arrive
 * sideways at the classifier or the UI.
 */
object BitmapLoader {

    /**
     * Decodes [path] so that the longest edge does not greatly exceed
     * [maxDimension]. Returns null when the file is missing or not an image.
     */
    fun decode(path: String, maxDimension: Int): Bitmap? {
        if (!File(path).exists()) return null

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight, maxDimension)
        }
        val bitmap = BitmapFactory.decodeFile(path, options) ?: return null
        return applyExifRotation(path, bitmap)
    }

    internal fun sampleSize(width: Int, height: Int, maxDimension: Int): Int {
        var sample = 1
        var largest = maxOf(width, height)
        while (largest / 2 >= maxDimension) {
            sample *= 2
            largest /= 2
        }
        return sample
    }

    private fun applyExifRotation(path: String, bitmap: Bitmap): Bitmap {
        val degrees = try {
            ExifInterface(path).rotationDegrees
        } catch (_: IOException) {
            0
        }
        if (degrees == 0) return bitmap

        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }
}
