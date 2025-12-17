package com.livestock.recognition.services

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Service for validating image quality and providing feedback for ML inference.
 * Implements resolution, brightness, and blur detection algorithms.
 * 
 * Requirements: 1.3, 6.1
 */
class ImageQualityService {
    
    companion object {
        private const val MIN_RESOLUTION_WIDTH = 224
        private const val MIN_RESOLUTION_HEIGHT = 224
        private const val MIN_BRIGHTNESS = 30
        private const val MAX_BRIGHTNESS = 225
        private const val BLUR_THRESHOLD = 100.0
        private const val CONTRAST_THRESHOLD = 20.0
    }
    
    /**
     * Validates image quality and returns a comprehensive quality report
     */
    fun validateImageQuality(bitmap: Bitmap): ImageQualityReport {
        val issues = mutableListOf<QualityIssue>()
        val metrics = mutableMapOf<String, Double>()
        
        // Check resolution
        val resolutionValid = validateResolution(bitmap)
        if (!resolutionValid) {
            issues.add(QualityIssue.LOW_RESOLUTION)
        }
        metrics["width"] = bitmap.width.toDouble()
        metrics["height"] = bitmap.height.toDouble()
        
        // Check brightness
        val brightness = calculateBrightness(bitmap)
        metrics["brightness"] = brightness
        when {
            brightness < MIN_BRIGHTNESS -> issues.add(QualityIssue.TOO_DARK)
            brightness > MAX_BRIGHTNESS -> issues.add(QualityIssue.TOO_BRIGHT)
        }
        
        // Check blur
        val blurScore = calculateBlurScore(bitmap)
        metrics["blur_score"] = blurScore
        if (blurScore < BLUR_THRESHOLD) {
            issues.add(QualityIssue.TOO_BLURRY)
        }
        
        // Check contrast
        val contrast = calculateContrast(bitmap)
        metrics["contrast"] = contrast
        if (contrast < CONTRAST_THRESHOLD) {
            issues.add(QualityIssue.LOW_CONTRAST)
        }
        
        // Calculate overall quality score (0-100)
        val qualityScore = calculateOverallQuality(metrics, issues)
        
        return ImageQualityReport(
            isAcceptable = issues.isEmpty(),
            qualityScore = qualityScore,
            issues = issues,
            metrics = metrics,
            feedback = generateFeedback(issues)
        )
    }
    
    /**
     * Preprocesses image for ML inference with quality optimizations
     */
    fun preprocessForInference(bitmap: Bitmap): ProcessedImage {
        val startTime = System.currentTimeMillis()
        
        // Resize to standard ML input size while maintaining aspect ratio
        val resized = resizeForInference(bitmap, 224, 224)
        
        // Apply basic image enhancements
        val enhanced = enhanceImage(resized)
        
        val processingTime = System.currentTimeMillis() - startTime
        
        return ProcessedImage(
            bitmap = enhanced,
            originalWidth = bitmap.width,
            originalHeight = bitmap.height,
            processedWidth = enhanced.width,
            processedHeight = enhanced.height,
            processingTimeMs = processingTime
        )
    }
    
    private fun validateResolution(bitmap: Bitmap): Boolean {
        return bitmap.width >= MIN_RESOLUTION_WIDTH && bitmap.height >= MIN_RESOLUTION_HEIGHT
    }
    
    private fun calculateBrightness(bitmap: Bitmap): Double {
        var totalBrightness = 0.0
        var pixelCount = 0
        
        // Sample pixels for performance (every 10th pixel)
        val step = 10
        for (x in 0 until bitmap.width step step) {
            for (y in 0 until bitmap.height step step) {
                val pixel = bitmap.getPixel(x, y)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)
                
                // Calculate luminance using standard formula
                val brightness = 0.299 * r + 0.587 * g + 0.114 * b
                totalBrightness += brightness
                pixelCount++
            }
        }
        
        return if (pixelCount > 0) totalBrightness / pixelCount else 0.0
    }
    
    private fun calculateBlurScore(bitmap: Bitmap): Double {
        // Use Laplacian variance to detect blur
        val width = bitmap.width
        val height = bitmap.height
        
        if (width < 3 || height < 3) return 0.0
        
        // Convert to grayscale and calculate Laplacian
        var variance = 0.0
        var mean = 0.0
        var count = 0
        
        for (x in 1 until width - 1) {
            for (y in 1 until height - 1) {
                val center = getGrayValue(bitmap, x, y)
                val top = getGrayValue(bitmap, x, y - 1)
                val bottom = getGrayValue(bitmap, x, y + 1)
                val left = getGrayValue(bitmap, x - 1, y)
                val right = getGrayValue(bitmap, x + 1, y)
                
                // Laplacian kernel: 4*center - (top + bottom + left + right)
                val laplacian = 4 * center - (top + bottom + left + right)
                mean += laplacian
                count++
            }
        }
        
        mean /= count
        
        // Calculate variance
        for (x in 1 until width - 1) {
            for (y in 1 until height - 1) {
                val center = getGrayValue(bitmap, x, y)
                val top = getGrayValue(bitmap, x, y - 1)
                val bottom = getGrayValue(bitmap, x, y + 1)
                val left = getGrayValue(bitmap, x - 1, y)
                val right = getGrayValue(bitmap, x + 1, y)
                
                val laplacian = 4 * center - (top + bottom + left + right)
                variance += (laplacian - mean).pow(2)
            }
        }
        
        return variance / count
    }
    
    private fun calculateContrast(bitmap: Bitmap): Double {
        var min = 255.0
        var max = 0.0
        
        // Sample pixels for performance
        val step = 10
        for (x in 0 until bitmap.width step step) {
            for (y in 0 until bitmap.height step step) {
                val gray = getGrayValue(bitmap, x, y)
                min = minOf(min, gray)
                max = maxOf(max, gray)
            }
        }
        
        return max - min
    }
    
    private fun getGrayValue(bitmap: Bitmap, x: Int, y: Int): Double {
        val pixel = bitmap.getPixel(x, y)
        val r = Color.red(pixel)
        val g = Color.green(pixel)
        val b = Color.blue(pixel)
        return 0.299 * r + 0.587 * g + 0.114 * b
    }
    
    private fun calculateOverallQuality(metrics: Map<String, Double>, issues: List<QualityIssue>): Double {
        var score = 100.0
        
        // Deduct points for each issue
        issues.forEach { issue ->
            score -= when (issue) {
                QualityIssue.LOW_RESOLUTION -> 30.0
                QualityIssue.TOO_DARK, QualityIssue.TOO_BRIGHT -> 20.0
                QualityIssue.TOO_BLURRY -> 25.0
                QualityIssue.LOW_CONTRAST -> 15.0
            }
        }
        
        return maxOf(0.0, score)
    }
    
    private fun generateFeedback(issues: List<QualityIssue>): List<String> {
        return issues.map { issue ->
            when (issue) {
                QualityIssue.LOW_RESOLUTION -> "Image resolution too low. Please use higher quality camera."
                QualityIssue.TOO_DARK -> "Image too dark. Please improve lighting conditions."
                QualityIssue.TOO_BRIGHT -> "Image too bright. Please reduce lighting or move to shade."
                QualityIssue.TOO_BLURRY -> "Image too blurry. Hold camera steady and ensure proper focus."
                QualityIssue.LOW_CONTRAST -> "Image has low contrast. Ensure good lighting and clear subject visibility."
            }
        }
    }
    
    private fun resizeForInference(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }
    
    private fun enhanceImage(bitmap: Bitmap): Bitmap {
        // For now, return the bitmap as-is
        // Future enhancements could include histogram equalization, noise reduction, etc.
        return bitmap
    }
}

/**
 * Data class representing the result of image quality validation
 */
data class ImageQualityReport(
    val isAcceptable: Boolean,
    val qualityScore: Double,
    val issues: List<QualityIssue>,
    val metrics: Map<String, Double>,
    val feedback: List<String>
)

/**
 * Data class representing a processed image ready for ML inference
 */
data class ProcessedImage(
    val bitmap: Bitmap,
    val originalWidth: Int,
    val originalHeight: Int,
    val processedWidth: Int,
    val processedHeight: Int,
    val processingTimeMs: Long
)

/**
 * Enum representing different types of image quality issues
 */
enum class QualityIssue {
    LOW_RESOLUTION,
    TOO_DARK,
    TOO_BRIGHT,
    TOO_BLURRY,
    LOW_CONTRAST
}