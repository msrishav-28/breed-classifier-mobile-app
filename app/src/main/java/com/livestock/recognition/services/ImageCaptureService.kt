package com.livestock.recognition.services

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File

/**
 * Service interface for image capture operations with quality validation integration.
 * Provides methods for loading, validating, and preprocessing images.
 * 
 * Requirements: 1.1, 1.3, 6.1
 */
interface ImageCaptureService {
    fun loadImageFromPath(imagePath: String): ImageResult
    fun validateImageQuality(bitmap: Bitmap): ImageQualityReport
    fun preprocessImage(bitmap: Bitmap): ProcessedImage
}

/**
 * Default implementation of ImageCaptureService
 */
class DefaultImageCaptureService(
    private val qualityService: ImageQualityService = ImageQualityService()
) : ImageCaptureService {
    
    override fun loadImageFromPath(imagePath: String): ImageResult {
        return try {
            val file = File(imagePath)
            if (!file.exists()) {
                return ImageResult.Error("Image file not found: $imagePath")
            }
            
            val bitmap = BitmapFactory.decodeFile(imagePath)
            if (bitmap == null) {
                return ImageResult.Error("Failed to decode image: $imagePath")
            }
            
            ImageResult.Success(bitmap, imagePath)
        } catch (e: Exception) {
            ImageResult.Error("Error loading image: ${e.message}")
        }
    }
    
    override fun validateImageQuality(bitmap: Bitmap): ImageQualityReport {
        return qualityService.validateImageQuality(bitmap)
    }
    
    override preprocessImage(bitmap: Bitmap): ProcessedImage {
        return qualityService.preprocessForInference(bitmap)
    }
}

/**
 * Sealed class representing the result of image loading operations
 */
sealed class ImageResult {
    data class Success(val bitmap: Bitmap, val imagePath: String) : ImageResult()
    data class Error(val message: String) : ImageResult()
}