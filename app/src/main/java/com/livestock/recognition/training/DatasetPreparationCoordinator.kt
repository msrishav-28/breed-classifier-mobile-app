package com.livestock.recognition.training

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Coordinates the complete dataset preparation pipeline
 * Manages acquisition, cleaning, augmentation, and validation
 */
class DatasetPreparationCoordinator(private val context: Context) {
    
    companion object {
        private const val TAG = "DatasetPreparation"
    }

    private val acquisitionService = DatasetAcquisitionService(context)
    private val augmentationPipeline = AdvancedAugmentationPipeline(context)

    data class PreparationResult(
        val acquisitionStats: DatasetAcquisitionService.DatasetStats,
        val augmentationStats: AdvancedAugmentationPipeline.AugmentationStats,
        val finalImageCount: Int,
        val success: Boolean,
        val processingTimeMs: Long
    )

    /**
     * Executes complete dataset preparation pipeline
     */
    suspend fun prepareCompleteDataset(
        augmentationConfig: AdvancedAugmentationPipeline.AugmentationConfig = 
            AdvancedAugmentationPipeline.AugmentationConfig()
    ): PreparationResult = withContext(Dispatchers.IO) {
        
        val startTime = System.currentTimeMillis()
        Log.i(TAG, "Starting complete dataset preparation pipeline...")

        try {
            // Step 1: Acquire and organize dataset
            Log.i(TAG, "Phase 1: Dataset acquisition and cleaning...")
            val acquisitionStats = acquisitionService.acquireCompleteDataset()
            
            Log.i(TAG, "Acquisition complete:")
            Log.i(TAG, "  Total images: ${acquisitionStats.totalImages}")
            Log.i(TAG, "  Duplicates removed: ${acquisitionStats.duplicatesRemoved}")
            Log.i(TAG, "  Train/Val/Test: ${acquisitionStats.trainCount}/${acquisitionStats.validationCount}/${acquisitionStats.testCount}")
            
            // Validate acquisition
            val acquisitionValid = acquisitionService.validateDataset()
            if (!acquisitionValid) {
                Log.e(TAG, "Dataset acquisition validation failed")
                return@withContext PreparationResult(
                    acquisitionStats = acquisitionStats,
                    augmentationStats = AdvancedAugmentationPipeline.AugmentationStats(0, 0, 0, emptyMap()),
                    finalImageCount = 0,
                    success = false,
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }

            // Step 2: Apply advanced augmentation pipeline
            Log.i(TAG, "Phase 2: Advanced augmentation pipeline...")
            val datasetPath = File(context.filesDir, "livestock_dataset").absolutePath
            val augmentationStats = augmentationPipeline.augmentDataset(datasetPath, augmentationConfig)
            
            Log.i(TAG, "Augmentation complete:")
            Log.i(TAG, "  Original images: ${augmentationStats.originalImages}")
            Log.i(TAG, "  Augmented images: ${augmentationStats.augmentedImages}")
            Log.i(TAG, "  Total images: ${augmentationStats.totalImages}")
            Log.i(TAG, "  Techniques applied: ${augmentationStats.techniquesApplied}")

            // Validate augmentation
            val augmentationValid = augmentationPipeline.validateAugmentation()
            if (!augmentationValid) {
                Log.w(TAG, "Augmentation validation failed - target range not met")
            }

            val processingTime = System.currentTimeMillis() - startTime
            Log.i(TAG, "Dataset preparation complete in ${processingTime}ms")

            PreparationResult(
                acquisitionStats = acquisitionStats,
                augmentationStats = augmentationStats,
                finalImageCount = augmentationStats.totalImages,
                success = acquisitionValid && augmentationValid,
                processingTimeMs = processingTime
            )

        } catch (e: Exception) {
            Log.e(TAG, "Dataset preparation failed", e)
            PreparationResult(
                acquisitionStats = DatasetAcquisitionService.DatasetStats(0, 0, emptyMap(), 0, 0, 0),
                augmentationStats = AdvancedAugmentationPipeline.AugmentationStats(0, 0, 0, emptyMap()),
                finalImageCount = 0,
                success = false,
                processingTimeMs = System.currentTimeMillis() - startTime
            )
        }
    }

    /**
     * Gets current dataset statistics
     */
    suspend fun getDatasetStatistics(): Map<String, Any> = withContext(Dispatchers.IO) {
        val stats = mutableMapOf<String, Any>()
        
        try {
            // Check original dataset
            val originalDir = File(context.filesDir, "livestock_dataset")
            if (originalDir.exists()) {
                val originalCount = countImagesInDirectory(originalDir)
                stats["original_images"] = originalCount
            }
            
            // Check augmented dataset
            val augmentedDir = File(context.filesDir, "augmented_dataset")
            if (augmentedDir.exists()) {
                val augmentedCount = countImagesInDirectory(augmentedDir)
                stats["augmented_images"] = augmentedCount
                
                // Count by split
                val splits = listOf("train", "validation", "test")
                for (split in splits) {
                    val splitDir = File(augmentedDir, split)
                    if (splitDir.exists()) {
                        stats["${split}_images"] = countImagesInDirectory(splitDir)
                    }
                }
                
                // Count by breed
                val breedCounts = mutableMapOf<String, Int>()
                for (split in splits) {
                    val splitDir = File(augmentedDir, split)
                    if (splitDir.exists()) {
                        val breedDirs = splitDir.listFiles()?.filter { it.isDirectory } ?: continue
                        for (breedDir in breedDirs) {
                            val count = countImagesInDirectory(breedDir)
                            breedCounts[breedDir.name] = breedCounts.getOrDefault(breedDir.name, 0) + count
                        }
                    }
                }
                stats["breed_counts"] = breedCounts
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get dataset statistics", e)
            stats["error"] = e.message ?: "Unknown error"
        }
        
        stats
    }

    /**
     * Cleans up dataset directories
     */
    suspend fun cleanupDatasets(
        cleanOriginal: Boolean = false,
        cleanAugmented: Boolean = false
    ): Boolean = withContext(Dispatchers.IO) {
        
        try {
            if (cleanOriginal) {
                val originalDir = File(context.filesDir, "livestock_dataset")
                if (originalDir.exists()) {
                    originalDir.deleteRecursively()
                    Log.i(TAG, "Cleaned original dataset directory")
                }
            }
            
            if (cleanAugmented) {
                val augmentedDir = File(context.filesDir, "augmented_dataset")
                if (augmentedDir.exists()) {
                    augmentedDir.deleteRecursively()
                    Log.i(TAG, "Cleaned augmented dataset directory")
                }
            }
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup datasets", e)
            false
        }
    }

    /**
     * Counts total images in a directory recursively
     */
    private fun countImagesInDirectory(directory: File): Int {
        if (!directory.exists() || !directory.isDirectory) return 0
        
        var count = 0
        val files = directory.listFiles() ?: return 0
        
        for (file in files) {
            if (file.isDirectory) {
                count += countImagesInDirectory(file)
            } else if (file.extension.lowercase() in listOf("jpg", "jpeg", "png")) {
                count++
            }
        }
        
        return count
    }

    /**
     * Validates that dataset meets minimum requirements
     */
    suspend fun validateDatasetRequirements(): Boolean = withContext(Dispatchers.IO) {
        val stats = getDatasetStatistics()
        
        val augmentedImages = stats["augmented_images"] as? Int ?: 0
        val trainImages = stats["train_images"] as? Int ?: 0
        val validationImages = stats["validation_images"] as? Int ?: 0
        val testImages = stats["test_images"] as? Int ?: 0
        
        val breedCounts = stats["breed_counts"] as? Map<String, Int> ?: emptyMap()
        
        // Check minimum requirements
        val meetsMinimumImages = augmentedImages >= 30000 // Target minimum
        val hasBalancedSplits = trainImages > 0 && validationImages > 0 && testImages > 0
        val hasMinimumBreeds = breedCounts.size >= 20 // 15 cattle + 8 buffalo minimum
        val hasBalancedBreeds = breedCounts.values.all { it >= 50 } // Minimum per breed
        
        Log.i(TAG, "Dataset validation:")
        Log.i(TAG, "  Total images: $augmentedImages (min: 30000) - ${if (meetsMinimumImages) "✓" else "✗"}")
        Log.i(TAG, "  Balanced splits: ${if (hasBalancedSplits) "✓" else "✗"}")
        Log.i(TAG, "  Breed count: ${breedCounts.size} (min: 20) - ${if (hasMinimumBreeds) "✓" else "✗"}")
        Log.i(TAG, "  Balanced breeds: ${if (hasBalancedBreeds) "✓" else "✗"}")
        
        meetsMinimumImages && hasBalancedSplits && hasMinimumBreeds && hasBalancedBreeds
    }
}