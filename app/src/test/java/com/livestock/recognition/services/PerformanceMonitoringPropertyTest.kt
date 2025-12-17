package com.livestock.recognition.services

import android.content.Context
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.junit.MockitoJUnitRunner
import kotlin.random.Random

/**
 * Property-based test for performance monitoring functionality.
 * **Feature: cattle-buffalo-recognition, Property 11: Performance metrics logging**
 * **Validates: Requirements 6.2**
 * 
 * Tests that performance metrics are consistently logged and tracked across all operations.
 */
@RunWith(MockitoJUnitRunner::class)
class PerformanceMonitoringPropertyTest {
    
    @Mock
    private lateinit var context: Context
    
    private lateinit var performanceMetricsService: PerformanceMetricsService
    
    @Before
    fun setup() {
        performanceMetricsService = PerformanceMetricsService(context)
    }
    
    @Test
    fun `Property 11 - Performance metrics logging - For any operation timing, metrics should be recorded and retrievable`() {
        // Run property-based test with 100 iterations
        repeat(100) {
            val operationId = generateRandomOperationId()
            val operationType = generateRandomOperationType()
            
            // Start timing
            performanceMetricsService.startTiming(operationId)
            
            // Simulate some processing time
            val processingTime = Random.nextLong(10, 5000)
            Thread.sleep(processingTime.coerceAtMost(50)) // Limit actual sleep for test speed
            
            // End timing
            val recordedTime = performanceMetricsService.endTiming(operationId)
            
            // Verify timing was recorded
            assertTrue("Recorded time should be positive", recordedTime > 0)
            
            // Record as specific operation type
            performanceMetricsService.recordInferenceTime(operationType, recordedTime)
            
            // Verify stats can be retrieved
            val stats = performanceMetricsService.getPerformanceStats(operationType)
            assertNotNull("Stats should be available for recorded operation", stats)
            assertEquals("Operation type should match", operationType, stats!!.operationType)
            assertTrue("Count should be positive", stats.count > 0)
            assertTrue("Total time should be positive", stats.totalTimeMs > 0)
            assertTrue("Average time should be positive", stats.averageTimeMs > 0)
        }
    }
    
    @Test
    fun `Property 11a - Timing consistency - Multiple timings for same operation should accumulate correctly`() {
        repeat(50) {
            val operationType = generateRandomOperationType()
            val timingCount = Random.nextInt(1, 20)
            val recordedTimes = mutableListOf<Long>()
            
            // Record multiple timings for the same operation type
            repeat(timingCount) {
                val time = Random.nextLong(100, 3000)
                performanceMetricsService.recordInferenceTime(operationType, time)
                recordedTimes.add(time)
            }
            
            // Verify accumulated stats
            val stats = performanceMetricsService.getPerformanceStats(operationType)
            assertNotNull("Stats should be available", stats)
            assertEquals("Count should match recorded timings", timingCount, stats!!.count)
            assertEquals("Total time should match sum", recordedTimes.sum(), stats.totalTimeMs)
            
            val expectedAverage = recordedTimes.average()
            assertEquals("Average should be correct", expectedAverage, stats.averageTimeMs, 0.01)
            
            assertEquals("Min should be correct", recordedTimes.minOrNull()!!, stats.minTimeMs)
            assertEquals("Max should be correct", recordedTimes.maxOrNull()!!, stats.maxTimeMs)
        }
    }
    
    @Test
    fun `Property 11b - Memory usage tracking - Memory snapshots should be captured and stored`() {
        repeat(100) {
            val initialSnapshotCount = performanceMetricsService.getRecentMemoryUsage().size
            
            // Capture memory usage
            performanceMetricsService.captureMemoryUsage()
            
            // Verify snapshot was added
            val newSnapshotCount = performanceMetricsService.getRecentMemoryUsage().size
            assertTrue("Memory snapshot count should increase", newSnapshotCount > initialSnapshotCount)
            
            // Verify snapshot data
            val recentSnapshots = performanceMetricsService.getRecentMemoryUsage(1)
            assertFalse("Should have at least one snapshot", recentSnapshots.isEmpty())
            
            val snapshot = recentSnapshots.first()
            assertTrue("Timestamp should be recent", snapshot.timestamp > 0)
            assertTrue("Total memory should be positive", snapshot.totalMemoryMB >= 0)
            assertTrue("Available memory should be non-negative", snapshot.availableMemoryMB >= 0)
            assertTrue("Used memory should be non-negative", snapshot.usedMemoryMB >= 0)
        }
    }
    
    @Test
    fun `Property 11c - Battery usage tracking - Battery snapshots should be captured and stored`() {
        repeat(100) {
            val initialSnapshotCount = performanceMetricsService.getRecentBatteryUsage().size
            
            // Capture battery usage
            performanceMetricsService.captureBatteryUsage()
            
            // Verify snapshot was added
            val newSnapshotCount = performanceMetricsService.getRecentBatteryUsage().size
            assertTrue("Battery snapshot count should increase", newSnapshotCount > initialSnapshotCount)
            
            // Verify snapshot data
            val recentSnapshots = performanceMetricsService.getRecentBatteryUsage(1)
            assertFalse("Should have at least one snapshot", recentSnapshots.isEmpty())
            
            val snapshot = recentSnapshots.first()
            assertTrue("Timestamp should be recent", snapshot.timestamp > 0)
            assertTrue("Battery level should be valid", snapshot.batteryLevel in 0..100)
            assertTrue("Battery temperature should be reasonable", snapshot.batteryTemperature > -50 && snapshot.batteryTemperature < 100)
        }
    }
    
    @Test
    fun `Property 11d - Overall stats consistency - Overall stats should reflect individual operation stats`() {
        repeat(50) {
            // Clear previous metrics
            performanceMetricsService.clearMetrics()
            
            val operationTypes = generateRandomOperationTypes()
            var totalExpectedInferences = 0L
            var totalExpectedTime = 0L
            
            // Record various operations
            operationTypes.forEach { operationType ->
                val operationCount = Random.nextInt(1, 10)
                repeat(operationCount) {
                    val time = Random.nextLong(100, 2000)
                    performanceMetricsService.recordInferenceTime(operationType, time)
                    totalExpectedInferences++
                    totalExpectedTime += time
                }
            }
            
            // Verify overall stats
            val overallStats = performanceMetricsService.getOverallPerformanceStats()
            assertEquals("Total inferences should match", totalExpectedInferences, overallStats.totalInferences)
            assertEquals("Total processing time should match", totalExpectedTime, overallStats.totalProcessingTimeMs)
            
            if (totalExpectedInferences > 0) {
                val expectedAverage = totalExpectedTime.toDouble() / totalExpectedInferences
                assertEquals("Average processing time should be correct", expectedAverage, overallStats.averageProcessingTimeMs, 0.01)
            }
            
            assertEquals("Operation types should match", operationTypes.toSet(), overallStats.operationTypes.toSet())
        }
    }
    
    @Test
    fun `Property 11e - Percentile calculations - P95 and P99 should be correctly calculated`() {
        repeat(50) {
            val operationType = generateRandomOperationType()
            val timingCount = Random.nextInt(20, 100) // Need enough data for percentiles
            val timings = mutableListOf<Long>()
            
            // Generate and record timings
            repeat(timingCount) {
                val time = Random.nextLong(100, 5000)
                timings.add(time)
                performanceMetricsService.recordInferenceTime(operationType, time)
            }
            
            val stats = performanceMetricsService.getPerformanceStats(operationType)
            assertNotNull("Stats should be available", stats)
            
            // Verify percentiles are within expected range
            val sortedTimings = timings.sorted()
            assertTrue("P95 should be >= median", stats!!.p95TimeMs >= stats.medianTimeMs.toLong())
            assertTrue("P99 should be >= P95", stats.p99TimeMs >= stats.p95TimeMs)
            assertTrue("P99 should be <= max", stats.p99TimeMs <= stats.maxTimeMs)
            
            // Verify percentiles are reasonable
            val p95Index = (timingCount * 0.95).toInt().coerceAtMost(timingCount - 1)
            val p99Index = (timingCount * 0.99).toInt().coerceAtMost(timingCount - 1)
            
            assertEquals("P95 should match calculated value", sortedTimings[p95Index], stats.p95TimeMs)
            assertEquals("P99 should match calculated value", sortedTimings[p99Index], stats.p99TimeMs)
        }
    }
    
    // Helper functions for generating random test data
    
    private fun generateRandomOperationId(): String {
        return "operation_${Random.nextInt(1000, 9999)}_${System.currentTimeMillis()}"
    }
    
    private fun generateRandomOperationType(): String {
        val types = listOf(
            "breed_classification",
            "type_classification", 
            "image_preprocessing",
            "model_inference",
            "result_processing",
            "annotation_generation"
        )
        return types[Random.nextInt(types.size)]
    }
    
    private fun generateRandomOperationTypes(): List<String> {
        val allTypes = listOf(
            "breed_classification",
            "type_classification", 
            "image_preprocessing",
            "model_inference",
            "result_processing",
            "annotation_generation"
        )
        val count = Random.nextInt(1, allTypes.size + 1)
        return allTypes.shuffled().take(count)
    }
}