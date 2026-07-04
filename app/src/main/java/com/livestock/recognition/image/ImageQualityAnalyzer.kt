package com.livestock.recognition.image

import android.graphics.Bitmap
import com.livestock.recognition.core.quality.ImageQualityPolicy
import com.livestock.recognition.core.quality.LuminanceStatistics
import com.livestock.recognition.core.quality.QualityAssessment

/**
 * Bridges Android bitmaps to the pure quality analysis in :core. The bitmap
 * is downscaled before statistics run, which both bounds the cost and makes
 * the sharpness metric resolution-independent.
 */
object ImageQualityAnalyzer {

    private const val ANALYSIS_MAX_DIMENSION = 256

    fun analyze(bitmap: Bitmap): QualityAssessment {
        val scaled = downscale(bitmap)
        val width = scaled.width
        val height = scaled.height

        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)
        if (scaled !== bitmap) scaled.recycle()

        val luminance = IntArray(pixels.size) { i ->
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            (299 * r + 587 * g + 114 * b) / 1000
        }

        val metrics = LuminanceStatistics.compute(
            luminance = luminance,
            width = width,
            height = height,
            originalWidth = bitmap.width,
            originalHeight = bitmap.height,
        )
        return ImageQualityPolicy.assess(metrics)
    }

    private fun downscale(bitmap: Bitmap): Bitmap {
        val largest = maxOf(bitmap.width, bitmap.height)
        if (largest <= ANALYSIS_MAX_DIMENSION) return bitmap
        val scale = ANALYSIS_MAX_DIMENSION.toFloat() / largest
        return Bitmap.createScaledBitmap(
            bitmap,
            (bitmap.width * scale).toInt().coerceAtLeast(1),
            (bitmap.height * scale).toInt().coerceAtLeast(1),
            true,
        )
    }
}
