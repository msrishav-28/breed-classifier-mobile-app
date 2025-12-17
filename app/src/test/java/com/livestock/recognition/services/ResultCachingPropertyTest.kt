package com.livestock.recognition.services

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.livestock.recognition.data.database.dao.CacheDao
import com.livestock.recognition.data.database.entities.CacheEntity
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.repository.CacheStatistics
import com.livestock.recognition.data.repository.ClassificationRepository
import com.livestock.recognition.data.repository.OfflineDataManager
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.runner.RunWith
import org.mockito.kotlin.*
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Property-based test for result caching in offline mode
 * **Feature: cattle-buffalo-recognition, Property 8: Result caching in offline mode**
 * **Validates: Requirements 4.3**
 * 
 * Tests that classification results generated while offline are cached locally
 * for later synchronization across all valid inputs
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ResultCachingPropertyTest : StringSpec({
    
    "Property 8: Result caching in offline mode - classification results should be cached locally when offline" {
        
        // Arrange - Create test dependencies
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockCacheDao = mock<CacheDao>()
        val mockClassificationRepository = mock<ClassificationRepository>()
        
        // Configure mocks for offline operation
        whenever(mockCacheDao.insertCacheEntry(any())).thenReturn(Unit)
        whenever(mockCacheDao.getCacheEntry(any())).thenReturn(null) // No existing cache
        whenever(mockCacheDao.incrementAccessCount(any(), any())).thenReturn(Unit)
        whenever(mockCacheDao.getCacheEntryCount()).thenReturn(0)
        whenever(mockCacheDao.getTotalCacheSize()).thenReturn(0L)
        whenever(mockCacheDao.getAllCacheTypes()).thenReturn(emptyList())
        
        val offlineDataManager = OfflineDataManager(
            context,
            mockCacheDao,
            mockClassificationRepository
        )
        
        // Property test generators
        val classificationResultGen = Arb.bind(
            Arb.element(listOf(
                "Gir", "Sahiwal", "Red_Sindhi", "Tharparkar", "Rathi",
                "Hariana", "Ongole", "Krishna_Valley", "Deoni", "Khillari",
                "Murrah", "Mehsana", "Surti", "Jaffarabadi", "Nili_Ravi"
            )),
            Arb.float(0.3f, 1.0f),
            Arb.enum<AnimalType>(),
            Arb.float(0.3f, 1.0f),
            Arb.long(1000000000L, 2000000000L), // Timestamp
            Arb.long(100L, 5000L) // Processing time
        ) { breed, breedConfidence, animalType, typeConfidence, timestamp, processingTime ->
            ClassificationResult(
                breed = breed,
                breedConfidence = breedConfidence,
                animalType = animalType,
                typeConfidence = typeConfidence,
                timestamp = timestamp,
                imageMetadata = null,
                processingTime = processingTime
            )
        }
        
        val cacheKeyGen = Arb.string(10..50, Arb.char('a'..'z', '0'..'9'))
        
        checkAll(100, classificationResultGen, cacheKeyGen) { classificationResult, cacheKey ->
            
            // Act - Cache classification result
            runBlocking {
                offlineDataManager.cacheClassificationResult(cacheKey, classificationResult)
            }
            
            // Assert - Verify caching behavior
            verify(mockCacheDao).insertCacheEntry(argThat { cacheEntity ->
                // Verify cache entry properties
                cacheEntity.cacheKey == cacheKey &&
                cacheEntity.cacheType == "classification" &&
                cacheEntity.cacheValue.isNotEmpty() &&
                cacheEntity.sizeBytes > 0 &&
                cacheEntity.expiresAt != null &&
                cacheEntity.expiresAt!! > System.currentTimeMillis()
            })
            
            // Reset mock for next iteration
            reset(mockCacheDao)
            whenever(mockCacheDao.insertCacheEntry(any())).thenReturn(Unit)
            whenever(mockCacheDao.getCacheEntry(any())).thenReturn(null)
        }
    }
    
    "Property 8a: Cache retrieval consistency - cached results should be retrievable and consistent" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockCacheDao = mock<CacheDao>()
        val mockClassificationRepository = mock<ClassificationRepository>()
        
        val offlineDataManager = OfflineDataManager(
            context,
            mockCacheDao,
            mockClassificationRepository
        )
        
        // Property test for cache retrieval
        val classificationResultGen = Arb.bind(
            Arb.element(listOf("Gir", "Sahiwal", "Murrah", "Mehsana")),
            Arb.float(0.5f, 1.0f),
            Arb.enum<AnimalType>()
        ) { breed, confidence, animalType ->
            ClassificationResult(
                breed = breed,
                breedConfidence = confidence,
                animalType = animalType,
                typeConfidence = confidence,
                timestamp = System.currentTimeMillis(),
                imageMetadata = null,
                processingTime = 1000L
            )
        }
        
        val cacheKeyGen = Arb.string(10..30, Arb.char('a'..'z', '0'..'9'))
        
        checkAll(50, classificationResultGen, cacheKeyGen) { originalResult, cacheKey ->
            
            // Configure mock to return cached entry
            val jsonValue = """{"breed":"${originalResult.breed}","breedConfidence":${originalResult.breedConfidence},"animalType":"${originalResult.animalType}","typeConfidence":${originalResult.typeConfidence},"timestamp":${originalResult.timestamp},"processingTime":${originalResult.processingTime}}"""
            
            val cachedEntity = CacheEntity(
                cacheKey = cacheKey,
                cacheValue = jsonValue,
                cacheType = "classification",
                expiresAt = System.currentTimeMillis() + 24 * 60 * 60 * 1000L, // 24 hours
                sizeBytes = jsonValue.length.toLong()
            )
            
            whenever(mockCacheDao.getCacheEntry(cacheKey)).thenReturn(cachedEntity)
            whenever(mockCacheDao.incrementAccessCount(any(), any())).thenReturn(Unit)
            
            // Act - Retrieve cached result
            val retrievedResult = runBlocking {
                offlineDataManager.getCachedClassificationResult(cacheKey)
            }
            
            // Assert - Verify consistency
            retrievedResult shouldNotBe null
            retrievedResult!!.breed shouldBe originalResult.breed
            retrievedResult.breedConfidence shouldBe originalResult.breedConfidence
            retrievedResult.animalType shouldBe originalResult.animalType
            retrievedResult.typeConfidence shouldBe originalResult.typeConfidence
            retrievedResult.timestamp shouldBe originalResult.timestamp
            retrievedResult.processingTime shouldBe originalResult.processingTime
            
            // Verify access count was incremented
            verify(mockCacheDao).incrementAccessCount(cacheKey)
            
            // Reset mock
            reset(mockCacheDao)
        }
    }
    
    "Property 8b: Cache expiration handling - expired cache entries should be handled correctly" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockCacheDao = mock<CacheDao>()
        val mockClassificationRepository = mock<ClassificationRepository>()
        
        val offlineDataManager = OfflineDataManager(
            context,
            mockCacheDao,
            mockClassificationRepository
        )
        
        // Property test for cache expiration
        val cacheKeyGen = Arb.string(10..30, Arb.char('a'..'z', '0'..'9'))
        val expiredTimeGen = Arb.long(1000000000L, System.currentTimeMillis() - 1000L) // Past timestamps
        val validTimeGen = Arb.long(System.currentTimeMillis() + 1000L, System.currentTimeMillis() + 86400000L) // Future timestamps
        
        checkAll(50, cacheKeyGen, expiredTimeGen, validTimeGen) { cacheKey, expiredTime, validTime ->
            
            // Test with expired cache entry
            val expiredEntity = CacheEntity(
                cacheKey = cacheKey,
                cacheValue = """{"breed":"Gir","breedConfidence":0.9}""",
                cacheType = "classification",
                expiresAt = expiredTime,
                sizeBytes = 100L
            )
            
            whenever(mockCacheDao.getCacheEntry(cacheKey)).thenReturn(expiredEntity)
            whenever(mockCacheDao.deleteCacheEntryByKey(cacheKey)).thenReturn(Unit)
            
            // Act - Try to retrieve expired cache
            val expiredResult = runBlocking {
                offlineDataManager.getCachedClassificationResult(cacheKey)
            }
            
            // Assert - Expired cache should return null and be deleted
            expiredResult shouldBe null
            verify(mockCacheDao).deleteCacheEntryByKey(cacheKey)
            
            // Reset and test with valid cache entry
            reset(mockCacheDao)
            
            val validEntity = CacheEntity(
                cacheKey = cacheKey,
                cacheValue = """{"breed":"Sahiwal","breedConfidence":0.8,"animalType":"DAIRY","typeConfidence":0.8,"timestamp":${System.currentTimeMillis()},"processingTime":1500}""",
                cacheType = "classification",
                expiresAt = validTime,
                sizeBytes = 150L
            )
            
            whenever(mockCacheDao.getCacheEntry(cacheKey)).thenReturn(validEntity)
            whenever(mockCacheDao.incrementAccessCount(any(), any())).thenReturn(Unit)
            
            // Act - Retrieve valid cache
            val validResult = runBlocking {
                offlineDataManager.getCachedClassificationResult(cacheKey)
            }
            
            // Assert - Valid cache should be returned
            validResult shouldNotBe null
            validResult!!.breed shouldBe "Sahiwal"
            validResult.breedConfidence shouldBe 0.8f
            
            // Verify cache was not deleted
            verify(mockCacheDao, never()).deleteCacheEntryByKey(any())
            
            // Reset for next iteration
            reset(mockCacheDao)
        }
    }
    
    "Property 8c: Cache size management - cache should respect size limits and cleanup when needed" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockCacheDao = mock<CacheDao>()
        val mockClassificationRepository = mock<ClassificationRepository>()
        
        val offlineDataManager = OfflineDataManager(
            context,
            mockCacheDao,
            mockClassificationRepository
        )
        
        // Property test for cache size management
        val largeCacheSizeGen = Arb.long(100_000_000L, 200_000_000L) // 100-200 MB
        val smallCacheSizeGen = Arb.long(1_000_000L, 50_000_000L) // 1-50 MB
        val entrySizeGen = Arb.long(1000L, 10_000_000L) // 1KB - 10MB
        
        checkAll(30, largeCacheSizeGen, smallCacheSizeGen, entrySizeGen) { largeCacheSize, smallCacheSize, entrySize ->
            
            // Test scenario where cache size exceeds limit
            whenever(mockCacheDao.getTotalCacheSize()).thenReturn(largeCacheSize)
            whenever(mockCacheDao.deleteLeastRecentlyUsedEntries(any())).thenReturn(5)
            whenever(mockCacheDao.insertCacheEntry(any())).thenReturn(Unit)
            
            val classificationResult = ClassificationResult(
                breed = "TestBreed",
                breedConfidence = 0.9f,
                animalType = AnimalType.DAIRY,
                typeConfidence = 0.9f,
                timestamp = System.currentTimeMillis(),
                imageMetadata = null,
                processingTime = 1000L
            )
            
            // Act - Cache result when cache is large
            runBlocking {
                offlineDataManager.cacheClassificationResult("test_key", classificationResult)
            }
            
            // Assert - Cleanup should be triggered
            verify(mockCacheDao).getTotalCacheSize()
            verify(mockCacheDao).deleteLeastRecentlyUsedEntries(any())
            
            // Reset and test with small cache size
            reset(mockCacheDao)
            whenever(mockCacheDao.getTotalCacheSize()).thenReturn(smallCacheSize)
            whenever(mockCacheDao.insertCacheEntry(any())).thenReturn(Unit)
            whenever(mockCacheDao.getCacheEntry(any())).thenReturn(null)
            
            // Act - Cache result when cache is small
            runBlocking {
                offlineDataManager.cacheClassificationResult("test_key_2", classificationResult)
            }
            
            // Assert - No cleanup should be triggered
            verify(mockCacheDao).getTotalCacheSize()
            verify(mockCacheDao, never()).deleteLeastRecentlyUsedEntries(any())
            
            // Reset for next iteration
            reset(mockCacheDao)
        }
    }
    
    "Property 8d: Cache statistics accuracy - cache statistics should accurately reflect cached data" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockCacheDao = mock<CacheDao>()
        val mockClassificationRepository = mock<ClassificationRepository>()
        
        val offlineDataManager = OfflineDataManager(
            context,
            mockCacheDao,
            mockClassificationRepository
        )
        
        // Property test for cache statistics
        val entryCountGen = Arb.int(0, 1000)
        val totalSizeGen = Arb.long(0L, 100_000_000L)
        val cacheTypesGen = Arb.list(
            Arb.element(listOf("classification", "breed_info", "model_result", "image_metadata")),
            0..5
        )
        
        checkAll(50, entryCountGen, totalSizeGen, cacheTypesGen) { entryCount, totalSize, cacheTypes ->
            
            // Configure mock responses
            whenever(mockCacheDao.getCacheEntryCount()).thenReturn(entryCount)
            whenever(mockCacheDao.getTotalCacheSize()).thenReturn(totalSize)
            whenever(mockCacheDao.getAllCacheTypes()).thenReturn(cacheTypes)
            
            // Configure type-specific statistics
            val typeStatistics = mutableMap<String, Pair<Int, Long>>()
            for (type in cacheTypes) {
                val typeCount = (entryCount * Math.random()).toInt()
                val typeSize = (totalSize * Math.random()).toLong()
                typeStatistics[type] = Pair(typeCount, typeSize)
                
                whenever(mockCacheDao.getCacheEntryCountByType(type)).thenReturn(typeCount)
                whenever(mockCacheDao.getCacheSizeByType(type)).thenReturn(typeSize)
            }
            
            // Act - Get cache statistics
            val statistics = runBlocking {
                offlineDataManager.getCacheStatistics()
            }
            
            // Assert - Verify statistics accuracy
            statistics.totalEntries shouldBe entryCount
            statistics.totalSizeBytes shouldBe totalSize
            statistics.typeStatistics.size shouldBe cacheTypes.size
            
            for (type in cacheTypes) {
                val expectedStats = typeStatistics[type]!!
                val actualStats = statistics.typeStatistics[type]!!
                actualStats.first shouldBe expectedStats.first
                actualStats.second shouldBe expectedStats.second
            }
            
            // Verify all DAO methods were called
            verify(mockCacheDao).getCacheEntryCount()
            verify(mockCacheDao).getTotalCacheSize()
            verify(mockCacheDao).getAllCacheTypes()
            
            for (type in cacheTypes) {
                verify(mockCacheDao).getCacheEntryCountByType(type)
                verify(mockCacheDao).getCacheSizeByType(type)
            }
            
            // Reset for next iteration
            reset(mockCacheDao)
        }
    }
})