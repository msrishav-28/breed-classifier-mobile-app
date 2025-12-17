package com.livestock.recognition.training

import android.content.Context
import android.graphics.*
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import kotlin.math.*
import kotlin.random.Random

/**
 * Advanced augmentation pipeline with 15+ techniques including weather simulation
 * Implements Mixup, CutMix, and geometric transformations
 * Targets 30,000-40,000 total images after augmentation
 */
class AdvancedAugmentationPipeline(private val context: Context) {
    
    companion object {
        private const val TAG = "AugmentationPipeline"
        private const val TARGET_MIN_IMAGES = 30000
        private const val TARGET_MAX_IMAGES = 40000
        private const val AUGMENTED_DIR = "augmented_dataset"
    }

    data class AugmentationConfig(
        val enableGeometric: Boolean = true,
        val enablePhotometric: Boolean = true,
        val enableWeatherSimulation: Boolean = true,
        val enableNoise: Boolean = true,
        val enableMixup: Boolean = true,
        val enableCutMix: Boolean = true,
        val enableAdvancedTechniques: Boolean = true,
        val augmentationFactor: Int = 3 // How many augmented versions per original
    )

    data class AugmentationStats(
        val originalImages: Int,
        val augmentedImages: Int,
        val totalImages: Int,
        val techniquesApplied: Map<String, Int>
    )

    private val augmentationTechniques = listOf(
        "rotation", "scaling", "horizontal_flip", "shift_scale_rotate",
        "brightness", "contrast", "clahe", "color_jitter", "hue_shift",
        "rain_simulation", "fog_simulation", "lighting_variation",
        "gaussian_noise", "coarse_dropout", "cutout",
        "mixup", "cutmix"
    )

    /**
     * Applies comprehensive augmentation pipeline to dataset
     */
    suspend fun augmentDataset(
        inputDatasetPath: String,
        config: AugmentationConfig = AugmentationConfig()
    ): AugmentationStats = withContext(Dispatchers.IO) {
        
        Log.i(TAG, "Starting advanced augmentation pipeline...")
        
        val inputDir = File(inputDatasetPath)
        val outputDir = File(context.filesDir, AUGMENTED_DIR)
        
        if (outputDir.exists()) {
            outputDir.deleteRecursively()
        }
        outputDir.mkdirs()

        val techniqueStats = mutableMapOf<String, Int>()
        var originalCount = 0
        var augmentedCount = 0

        // Process each split (train, validation, test)
        val splits = listOf("train", "validation", "test")
        
        for (split in splits) {
            val splitInputDir = File(inputDir, split)
            val splitOutputDir = File(outputDir, split)
            splitOutputDir.mkdirs()
            
            if (!splitInputDir.exists()) continue
            
            val breedDirs = splitInputDir.listFiles()?.filter { it.isDirectory } ?: continue
            
            for (breedDir in breedDirs) {
                val breedOutputDir = File(splitOutputDir, breedDir.name)
                breedOutputDir.mkdirs()
                
                val imageFiles = breedDir.listFiles()?.filter { 
                    it.extension.lowercase() in listOf("jpg", "jpeg", "png") 
                } ?: continue
                
                originalCount += imageFiles.size
                
                // Copy original images
                for (imageFile in imageFiles) {
                    val outputFile = File(breedOutputDir, imageFile.name)
                    imageFile.copyTo(outputFile)
                }
                
                // Generate augmented versions
                for (imageFile in imageFiles) {
                    val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                    if (bitmap == null) continue
                    
                    // Apply multiple augmentation techniques
                    for (i in 1..config.augmentationFactor) {
                        val augmentedBitmap = applyRandomAugmentations(bitmap, config, techniqueStats)
                        
                        val outputFile = File(breedOutputDir, 
                            "${imageFile.nameWithoutExtension}_aug_$i.jpg")
                        
                        saveBitmap(augmentedBitmap, outputFile)
                        augmentedCount++
                    }
                    
                    bitmap.recycle()
                }
                
                Log.i(TAG, "Processed breed ${breedDir.name} in $split: ${imageFiles.size} -> ${imageFiles.size * (1 + config.augmentationFactor)}")
            }
        }

        // Apply advanced techniques like Mixup and CutMix
        if (config.enableMixup || config.enableCutMix) {
            val advancedCount = applyAdvancedTechniques(outputDir, config, techniqueStats)
            augmentedCount += advancedCount
        }

        Log.i(TAG, "Augmentation complete: $originalCount original -> ${originalCount + augmentedCount} total")
        
        AugmentationStats(
            originalImages = originalCount,
            augmentedImages = augmentedCount,
            totalImages = originalCount + augmentedCount,
            techniquesApplied = techniqueStats
        )
    }

    /**
     * Applies random combination of augmentation techniques to a single image
     */
    private fun applyRandomAugmentations(
        bitmap: Bitmap,
        config: AugmentationConfig,
        stats: MutableMap<String, Int>
    ): Bitmap {
        
        var result = bitmap.copy(bitmap.config, true)
        
        // Geometric transformations
        if (config.enableGeometric) {
            if (Random.nextFloat() < 0.7f) {
                result = applyGeometricTransformations(result, stats)
            }
        }
        
        // Photometric adjustments
        if (config.enablePhotometric) {
            if (Random.nextFloat() < 0.8f) {
                result = applyPhotometricAdjustments(result, stats)
            }
        }
        
        // Weather simulation
        if (config.enableWeatherSimulation) {
            if (Random.nextFloat() < 0.3f) {
                result = applyWeatherSimulation(result, stats)
            }
        }
        
        // Noise and occlusion
        if (config.enableNoise) {
            if (Random.nextFloat() < 0.4f) {
                result = applyNoiseAndOcclusion(result, stats)
            }
        }
        
        return result
    }

    /**
     * Applies geometric transformations (rotation, scaling, flipping, etc.)
     */
    private fun applyGeometricTransformations(
        bitmap: Bitmap,
        stats: MutableMap<String, Int>
    ): Bitmap {
        
        val matrix = Matrix()
        var transformed = false
        
        // Rotation (-15 to +15 degrees)
        if (Random.nextFloat() < 0.5f) {
            val angle = Random.nextFloat() * 30f - 15f
            matrix.postRotate(angle, bitmap.width / 2f, bitmap.height / 2f)
            stats["rotation"] = stats.getOrDefault("rotation", 0) + 1
            transformed = true
        }
        
        // Scaling (0.8 to 1.2)
        if (Random.nextFloat() < 0.4f) {
            val scale = Random.nextFloat() * 0.4f + 0.8f
            matrix.postScale(scale, scale, bitmap.width / 2f, bitmap.height / 2f)
            stats["scaling"] = stats.getOrDefault("scaling", 0) + 1
            transformed = true
        }
        
        // Horizontal flip
        if (Random.nextFloat() < 0.5f) {
            matrix.postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            stats["horizontal_flip"] = stats.getOrDefault("horizontal_flip", 0) + 1
            transformed = true
        }
        
        return if (transformed) {
            try {
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } catch (e: Exception) {
                Log.w(TAG, "Geometric transformation failed", e)
                bitmap
            }
        } else {
            bitmap
        }
    }

    /**
     * Applies photometric adjustments (brightness, contrast, color)
     */
    private fun applyPhotometricAdjustments(
        bitmap: Bitmap,
        stats: MutableMap<String, Int>
    ): Bitmap {
        
        val result = bitmap.copy(bitmap.config, true)
        val canvas = Canvas(result)
        val paint = Paint()
        
        // Brightness adjustment (-50 to +50)
        if (Random.nextFloat() < 0.6f) {
            val brightness = Random.nextInt(-50, 51)
            val colorMatrix = ColorMatrix(floatArrayOf(
                1f, 0f, 0f, 0f, brightness.toFloat(),
                0f, 1f, 0f, 0f, brightness.toFloat(),
                0f, 0f, 1f, 0f, brightness.toFloat(),
                0f, 0f, 0f, 1f, 0f
            ))
            paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
            stats["brightness"] = stats.getOrDefault("brightness", 0) + 1
        }
        
        // Contrast adjustment (0.7 to 1.3)
        if (Random.nextFloat() < 0.5f) {
            val contrast = Random.nextFloat() * 0.6f + 0.7f
            val translate = (1f - contrast) * 127.5f
            val colorMatrix = ColorMatrix(floatArrayOf(
                contrast, 0f, 0f, 0f, translate,
                0f, contrast, 0f, 0f, translate,
                0f, 0f, contrast, 0f, translate,
                0f, 0f, 0f, 1f, 0f
            ))
            paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
            stats["contrast"] = stats.getOrDefault("contrast", 0) + 1
        }
        
        // Color jitter (hue shift)
        if (Random.nextFloat() < 0.3f) {
            val hueShift = Random.nextFloat() * 60f - 30f
            val colorMatrix = ColorMatrix()
            colorMatrix.setRotate(0, hueShift) // Red
            colorMatrix.setRotate(1, hueShift) // Green  
            colorMatrix.setRotate(2, hueShift) // Blue
            paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
            stats["hue_shift"] = stats.getOrDefault("hue_shift", 0) + 1
        }
        
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return result
    }

    /**
     * Applies weather simulation (rain, fog, lighting variations)
     */
    private fun applyWeatherSimulation(
        bitmap: Bitmap,
        stats: MutableMap<String, Int>
    ): Bitmap {
        
        val result = bitmap.copy(bitmap.config, true)
        val canvas = Canvas(result)
        val paint = Paint()
        
        when (Random.nextInt(3)) {
            0 -> {
                // Rain simulation
                paint.color = Color.argb(30, 200, 200, 255)
                paint.strokeWidth = 2f
                
                for (i in 0 until Random.nextInt(50, 150)) {
                    val x = Random.nextFloat() * bitmap.width
                    val y = Random.nextFloat() * bitmap.height
                    canvas.drawLine(x, y, x + 5, y + 20, paint)
                }
                stats["rain_simulation"] = stats.getOrDefault("rain_simulation", 0) + 1
            }
            1 -> {
                // Fog simulation
                paint.color = Color.argb(40, 220, 220, 220)
                canvas.drawRect(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat(), paint)
                stats["fog_simulation"] = stats.getOrDefault("fog_simulation", 0) + 1
            }
            2 -> {
                // Lighting variation (shadow overlay)
                val gradient = RadialGradient(
                    bitmap.width * Random.nextFloat(),
                    bitmap.height * Random.nextFloat(),
                    bitmap.width * 0.7f,
                    Color.TRANSPARENT,
                    Color.argb(60, 0, 0, 0),
                    Shader.TileMode.CLAMP
                )
                paint.shader = gradient
                canvas.drawRect(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat(), paint)
                stats["lighting_variation"] = stats.getOrDefault("lighting_variation", 0) + 1
            }
        }
        
        return result
    }

    /**
     * Applies noise and occlusion techniques
     */
    private fun applyNoiseAndOcclusion(
        bitmap: Bitmap,
        stats: MutableMap<String, Int>
    ): Bitmap {
        
        val result = bitmap.copy(bitmap.config, true)
        val canvas = Canvas(result)
        val paint = Paint()
        
        when (Random.nextInt(3)) {
            0 -> {
                // Gaussian noise
                val pixels = IntArray(bitmap.width * bitmap.height)
                bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                
                for (i in pixels.indices) {
                    if (Random.nextFloat() < 0.1f) {
                        val noise = Random.nextInt(-30, 31)
                        val r = Color.red(pixels[i]) + noise
                        val g = Color.green(pixels[i]) + noise
                        val b = Color.blue(pixels[i]) + noise
                        
                        pixels[i] = Color.rgb(
                            r.coerceIn(0, 255),
                            g.coerceIn(0, 255),
                            b.coerceIn(0, 255)
                        )
                    }
                }
                
                result.setPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                stats["gaussian_noise"] = stats.getOrDefault("gaussian_noise", 0) + 1
            }
            1 -> {
                // Coarse dropout (random rectangles)
                paint.color = Color.BLACK
                for (i in 0 until Random.nextInt(3, 8)) {
                    val x = Random.nextFloat() * bitmap.width
                    val y = Random.nextFloat() * bitmap.height
                    val w = Random.nextFloat() * bitmap.width * 0.1f
                    val h = Random.nextFloat() * bitmap.height * 0.1f
                    canvas.drawRect(x, y, x + w, y + h, paint)
                }
                stats["coarse_dropout"] = stats.getOrDefault("coarse_dropout", 0) + 1
            }
            2 -> {
                // Cutout (single large rectangle)
                paint.color = Color.GRAY
                val size = min(bitmap.width, bitmap.height) * 0.15f
                val x = Random.nextFloat() * (bitmap.width - size)
                val y = Random.nextFloat() * (bitmap.height - size)
                canvas.drawRect(x, y, x + size, y + size, paint)
                stats["cutout"] = stats.getOrDefault("cutout", 0) + 1
            }
        }
        
        return result
    }

    /**
     * Applies advanced techniques like Mixup and CutMix
     */
    private suspend fun applyAdvancedTechniques(
        datasetDir: File,
        config: AugmentationConfig,
        stats: MutableMap<String, Int>
    ): Int = withContext(Dispatchers.IO) {
        
        var advancedCount = 0
        
        val trainDir = File(datasetDir, "train")
        if (!trainDir.exists()) return@withContext 0
        
        val breedDirs = trainDir.listFiles()?.filter { it.isDirectory } ?: return@withContext 0
        val allImages = mutableListOf<Pair<File, String>>()
        
        // Collect all training images
        for (breedDir in breedDirs) {
            val imageFiles = breedDir.listFiles()?.filter { 
                it.extension.lowercase() in listOf("jpg", "jpeg", "png") 
            } ?: continue
            
            for (imageFile in imageFiles) {
                allImages.add(imageFile to breedDir.name)
            }
        }
        
        if (allImages.size < 2) return@withContext 0
        
        // Generate Mixup examples
        if (config.enableMixup) {
            val mixupCount = (allImages.size * 0.1).toInt() // 10% mixup examples
            
            for (i in 0 until mixupCount) {
                val (img1, breed1) = allImages.random()
                val (img2, breed2) = allImages.random()
                
                if (img1 != img2) {
                    val mixupResult = createMixupImage(img1, img2, breed1, breed2)
                    if (mixupResult != null) {
                        val outputDir = File(trainDir, breed1) // Use first breed's directory
                        val outputFile = File(outputDir, "mixup_${i}.jpg")
                        saveBitmap(mixupResult, outputFile)
                        advancedCount++
                        stats["mixup"] = stats.getOrDefault("mixup", 0) + 1
                    }
                }
            }
        }
        
        // Generate CutMix examples
        if (config.enableCutMix) {
            val cutmixCount = (allImages.size * 0.1).toInt() // 10% cutmix examples
            
            for (i in 0 until cutmixCount) {
                val (img1, breed1) = allImages.random()
                val (img2, breed2) = allImages.random()
                
                if (img1 != img2) {
                    val cutmixResult = createCutMixImage(img1, img2, breed1, breed2)
                    if (cutmixResult != null) {
                        val outputDir = File(trainDir, breed1) // Use first breed's directory
                        val outputFile = File(outputDir, "cutmix_${i}.jpg")
                        saveBitmap(cutmixResult, outputFile)
                        advancedCount++
                        stats["cutmix"] = stats.getOrDefault("cutmix", 0) + 1
                    }
                }
            }
        }
        
        advancedCount
    }

    /**
     * Creates Mixup image by blending two images
     */
    private fun createMixupImage(
        img1: File,
        img2: File,
        breed1: String,
        breed2: String
    ): Bitmap? {
        
        try {
            val bitmap1 = BitmapFactory.decodeFile(img1.absolutePath) ?: return null
            val bitmap2 = BitmapFactory.decodeFile(img2.absolutePath) ?: return null
            
            // Resize to same dimensions
            val targetWidth = min(bitmap1.width, bitmap2.width)
            val targetHeight = min(bitmap1.height, bitmap2.height)
            
            val resized1 = Bitmap.createScaledBitmap(bitmap1, targetWidth, targetHeight, true)
            val resized2 = Bitmap.createScaledBitmap(bitmap2, targetWidth, targetHeight, true)
            
            // Blend with random alpha
            val alpha = Random.nextFloat() * 0.6f + 0.2f // 0.2 to 0.8
            val result = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(result)
            
            val paint1 = Paint()
            paint1.alpha = (255 * alpha).toInt()
            canvas.drawBitmap(resized1, 0f, 0f, paint1)
            
            val paint2 = Paint()
            paint2.alpha = (255 * (1f - alpha)).toInt()
            canvas.drawBitmap(resized2, 0f, 0f, paint2)
            
            bitmap1.recycle()
            bitmap2.recycle()
            resized1.recycle()
            resized2.recycle()
            
            return result
            
        } catch (e: Exception) {
            Log.w(TAG, "Mixup creation failed", e)
            return null
        }
    }

    /**
     * Creates CutMix image by combining regions from two images
     */
    private fun createCutMixImage(
        img1: File,
        img2: File,
        breed1: String,
        breed2: String
    ): Bitmap? {
        
        try {
            val bitmap1 = BitmapFactory.decodeFile(img1.absolutePath) ?: return null
            val bitmap2 = BitmapFactory.decodeFile(img2.absolutePath) ?: return null
            
            // Resize to same dimensions
            val targetWidth = min(bitmap1.width, bitmap2.width)
            val targetHeight = min(bitmap1.height, bitmap2.height)
            
            val resized1 = Bitmap.createScaledBitmap(bitmap1, targetWidth, targetHeight, true)
            val resized2 = Bitmap.createScaledBitmap(bitmap2, targetWidth, targetHeight, true)
            
            val result = resized1.copy(resized1.config, true)
            val canvas = Canvas(result)
            
            // Cut random rectangular region from second image
            val cutWidth = (targetWidth * Random.nextFloat() * 0.5f).toInt()
            val cutHeight = (targetHeight * Random.nextFloat() * 0.5f).toInt()
            val cutX = Random.nextInt(0, targetWidth - cutWidth)
            val cutY = Random.nextInt(0, targetHeight - cutHeight)
            
            val cutRegion = Bitmap.createBitmap(resized2, cutX, cutY, cutWidth, cutHeight)
            canvas.drawBitmap(cutRegion, cutX.toFloat(), cutY.toFloat(), null)
            
            bitmap1.recycle()
            bitmap2.recycle()
            resized1.recycle()
            resized2.recycle()
            cutRegion.recycle()
            
            return result
            
        } catch (e: Exception) {
            Log.w(TAG, "CutMix creation failed", e)
            return null
        }
    }

    /**
     * Saves bitmap to file with JPEG compression
     */
    private fun saveBitmap(bitmap: Bitmap, file: File) {
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save bitmap to ${file.absolutePath}", e)
        }
    }

    /**
     * Validates augmentation results
     */
    suspend fun validateAugmentation(): Boolean = withContext(Dispatchers.IO) {
        val augmentedDir = File(context.filesDir, AUGMENTED_DIR)
        
        if (!augmentedDir.exists()) {
            Log.e(TAG, "Augmented dataset directory does not exist")
            return@withContext false
        }

        var totalImages = 0
        val splits = listOf("train", "validation", "test")
        
        for (split in splits) {
            val splitDir = File(augmentedDir, split)
            if (!splitDir.exists()) continue
            
            val breedDirs = splitDir.listFiles()?.filter { it.isDirectory } ?: continue
            
            for (breedDir in breedDirs) {
                val imageFiles = breedDir.listFiles()?.filter { 
                    it.extension.lowercase() in listOf("jpg", "jpeg", "png") 
                } ?: continue
                
                totalImages += imageFiles.size
                Log.d(TAG, "Split $split, Breed ${breedDir.name}: ${imageFiles.size} images")
            }
        }
        
        Log.i(TAG, "Augmentation validation complete. Total images: $totalImages")
        
        // Check if we reached target range
        val targetMet = totalImages in TARGET_MIN_IMAGES..TARGET_MAX_IMAGES
        if (!targetMet) {
            Log.w(TAG, "Target range not met: $totalImages (target: $TARGET_MIN_IMAGES-$TARGET_MAX_IMAGES)")
        }
        
        return@withContext targetMet
    }
}