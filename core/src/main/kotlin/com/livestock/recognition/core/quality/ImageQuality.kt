package com.livestock.recognition.core.quality

import kotlin.math.sqrt

/**
 * Luminance-based statistics for a (possibly downscaled) image.
 *
 * @property width Width of the analysed image in pixels.
 * @property height Height of the analysed image in pixels.
 * @property meanLuminance Average luminance in [0, 255].
 * @property luminanceStdDev Standard deviation of luminance; a proxy for contrast.
 * @property sharpness Variance of the Laplacian; low values indicate blur.
 */
data class QualityMetrics(
    val width: Int,
    val height: Int,
    val meanLuminance: Double,
    val luminanceStdDev: Double,
    val sharpness: Double,
)

enum class QualityIssue {
    LOW_RESOLUTION,
    TOO_DARK,
    TOO_BRIGHT,
    LOW_CONTRAST,
    BLURRY,
}

data class QualityAssessment(
    val metrics: QualityMetrics,
    val issues: List<QualityIssue>,
) {
    val isAcceptable: Boolean get() = issues.isEmpty()
}

/**
 * Computes [QualityMetrics] from a row-major luminance buffer. Pure math so
 * it can be exhaustively tested off-device; the app extracts luminance from
 * a Bitmap and hands it over.
 */
object LuminanceStatistics {

    /**
     * @param luminance row-major luminance values in [0, 255], one per pixel
     * @param width analysed image width; `luminance.size` must equal `width * height`
     * @param originalWidth width of the full-size source image (metrics report
     *   the source resolution while statistics run on the downscaled buffer)
     */
    fun compute(
        luminance: IntArray,
        width: Int,
        height: Int,
        originalWidth: Int = width,
        originalHeight: Int = height,
    ): QualityMetrics {
        require(width > 0 && height > 0) { "Image dimensions must be positive" }
        require(luminance.size == width * height) {
            "Expected ${width * height} luminance values, got ${luminance.size}"
        }

        var sum = 0.0
        var sumSquares = 0.0
        for (value in luminance) {
            sum += value
            sumSquares += value.toDouble() * value
        }
        val count = luminance.size.toDouble()
        val mean = sum / count
        val variance = (sumSquares / count - mean * mean).coerceAtLeast(0.0)

        return QualityMetrics(
            width = originalWidth,
            height = originalHeight,
            meanLuminance = mean,
            luminanceStdDev = sqrt(variance),
            sharpness = laplacianVariance(luminance, width, height),
        )
    }

    /**
     * Variance of the 4-neighbour Laplacian, the standard cheap blur metric.
     * Border pixels are excluded. Images smaller than 3x3 return 0.
     */
    private fun laplacianVariance(luminance: IntArray, width: Int, height: Int): Double {
        if (width < 3 || height < 3) return 0.0

        var sum = 0.0
        var sumSquares = 0.0
        var count = 0
        for (y in 1 until height - 1) {
            val row = y * width
            for (x in 1 until width - 1) {
                val idx = row + x
                val lap = (
                    luminance[idx - 1] + luminance[idx + 1] +
                        luminance[idx - width] + luminance[idx + width] -
                        4 * luminance[idx]
                    ).toDouble()
                sum += lap
                sumSquares += lap * lap
                count++
            }
        }
        val mean = sum / count
        return (sumSquares / count - mean * mean).coerceAtLeast(0.0)
    }
}

/**
 * Product policy for when an image is good enough to classify. Warnings are
 * advisory: classification proceeds, but the user is told why the result may
 * be unreliable.
 */
object ImageQualityPolicy {

    const val MIN_WIDTH = 224
    const val MIN_HEIGHT = 224
    const val MIN_MEAN_LUMINANCE = 40.0
    const val MAX_MEAN_LUMINANCE = 220.0
    const val MIN_LUMINANCE_STD_DEV = 18.0
    const val MIN_SHARPNESS = 60.0

    fun assess(metrics: QualityMetrics): QualityAssessment {
        val issues = mutableListOf<QualityIssue>()

        if (metrics.width < MIN_WIDTH || metrics.height < MIN_HEIGHT) {
            issues.add(QualityIssue.LOW_RESOLUTION)
        }
        when {
            metrics.meanLuminance < MIN_MEAN_LUMINANCE -> issues.add(QualityIssue.TOO_DARK)
            metrics.meanLuminance > MAX_MEAN_LUMINANCE -> issues.add(QualityIssue.TOO_BRIGHT)
        }
        if (metrics.luminanceStdDev < MIN_LUMINANCE_STD_DEV) {
            issues.add(QualityIssue.LOW_CONTRAST)
        }
        if (metrics.sharpness < MIN_SHARPNESS) {
            issues.add(QualityIssue.BLURRY)
        }

        return QualityAssessment(metrics, issues)
    }
}
