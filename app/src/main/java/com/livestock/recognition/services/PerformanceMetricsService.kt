package com.livestock.recognition.services

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/**
 * Service for collecting and monitoring performance metrics including inference time,
 * memory usage, and battery consumption tracking.
 * 
 * Requirements: 6.2, 6.5
 */
class PerformanceMetricsService(private val context: Context) {
    
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    
    // Metrics storage
    private val inferenceTimings = ConcurrentHashMap<String, MutableList<Long>>()
    private val memoryUsageHistory = mutableListOf<MemoryUsageSnapshot>()
    private val batteryUsageHistory = mutableListOf<BatteryUsageSnapshot>()
    
    // Counters
    private val totalInferences = AtomicLong(0)
    private val totalProcessingTime = AtomicLong(0)
    
    // Timing tracking
    private val activeTimings = ConcurrentHashMap<String, Long>()
    
    companion object {
        private const val TAG = "PerformanceMetrics"
        private const val MAX_HISTORY_SIZE = 1000
        private const val MEMORY_SAMPLING_INTERVAL = 5000L // 5 seconds
    }
    
    /**
     * Starts timing for a specific operation.
     * @param operationId Unique identifier for the operation
     */
    fun startTiming(operationId: String) {
        val startTime = SystemClock.elapsedRealtime()
        activeTimings[operationId] = startTime
        Log.d(TAG, "Started timing for operation: $operationId")
    }
    
    /**
     * Ends timing for a specific operation and records the duration.
     * @param operationId Unique identifier for the operation
     * @return The elapsed time in milliseconds, or -1 if operation was not started
     */
    fun endTiming(operationId: String): Long {
        val endTime = SystemClock.elapsedRealtime()
        val startTime = activeTimings.remove(operationId)
        
        return if (startTime != null) {
            val duration = endTime - startTime
            recordInferenceTime(operationId, duration)
            totalInferences.incrementAndGet()
            totalProcessingTime.addAndGet(duration)
            Log.d(TAG, "Completed timing for operation: $operationId, duration: ${duration}ms")
            duration
        } else {
            Log.w(TAG, "No start time found for operation: $operationId")
            -1L
        }
    }
    
    /**
     * Records inference time for a specific operation type.
     * @param operationType Type of operation (e.g., "breed_classification", "type_classification")
     * @param timeMs Time taken in milliseconds
     */
    fun recordInferenceTime(operationType: String, timeMs: Long) {
        inferenceTimings.computeIfAbsent(operationType) { mutableListOf() }.add(timeMs)
        
        // Keep only recent timings to prevent memory leaks
        val timings = inferenceTimings[operationType]!!
        if (timings.size > MAX_HISTORY_SIZE) {
            timings.removeAt(0)
        }
    }
    
    /**
     * Captures current memory usage snapshot.
     */
    fun captureMemoryUsage() {
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        val runtime = Runtime.getRuntime()
        val nativeHeapSize = Debug.getNativeHeapSize()
        val nativeHeapAllocated = Debug.getNativeHeapAllocatedSize()
        val nativeHeapFree = Debug.getNativeHeapFreeSize()
        
        val snapshot = MemoryUsageSnapshot(
            timestamp = System.currentTimeMillis(),
            totalMemoryMB = memoryInfo.totalMem / (1024 * 1024),
            availableMemoryMB = memoryInfo.availMem / (1024 * 1024),
            usedMemoryMB = (memoryInfo.totalMem - memoryInfo.availMem) / (1024 * 1024),
            jvmHeapSizeMB = runtime.totalMemory() / (1024 * 1024),
            jvmHeapUsedMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024),
            jvmHeapFreeMB = runtime.freeMemory() / (1024 * 1024),
            nativeHeapSizeMB = nativeHeapSize / (1024 * 1024),
            nativeHeapUsedMB = nativeHeapAllocated / (1024 * 1024),
            nativeHeapFreeMB = nativeHeapFree / (1024 * 1024),
            isLowMemory = memoryInfo.lowMemory
        )
        
        synchronized(memoryUsageHistory) {
            memoryUsageHistory.add(snapshot)
            if (memoryUsageHistory.size > MAX_HISTORY_SIZE) {
                memoryUsageHistory.removeAt(0)
            }
        }
        
        Log.d(TAG, "Memory usage captured: ${snapshot.usedMemoryMB}MB used, ${snapshot.availableMemoryMB}MB available")
    }
    
    /**
     * Captures current battery usage snapshot.
     */
    fun captureBatteryUsage() {
        val batteryLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val batteryStatus = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_STATUS)
        val batteryHealth = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_HEALTH)
        val batteryTemperature = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_TEMPERATURE) / 10.0f // Convert to Celsius
        
        val snapshot = BatteryUsageSnapshot(
            timestamp = System.currentTimeMillis(),
            batteryLevel = batteryLevel,
            batteryStatus = batteryStatus,
            batteryHealth = batteryHealth,
            batteryTemperature = batteryTemperature,
            isCharging = batteryStatus == BatteryManager.BATTERY_STATUS_CHARGING
        )
        
        synchronized(batteryUsageHistory) {
            batteryUsageHistory.add(snapshot)
            if (batteryUsageHistory.size > MAX_HISTORY_SIZE) {
                batteryUsageHistory.removeAt(0)
            }
        }
        
        Log.d(TAG, "Battery usage captured: ${batteryLevel}%, temp: ${batteryTemperature}°C")
    }
    
    /**
     * Gets performance statistics for a specific operation type.
     * @param operationType Type of operation to get stats for
     * @return PerformanceStats object with timing statistics
     */
    fun getPerformanceStats(operationType: String): PerformanceStats? {
        val timings = inferenceTimings[operationType] ?: return null
        
        if (timings.isEmpty()) return null
        
        val sortedTimings = timings.sorted()
        val count = timings.size
        val sum = timings.sum()
        val average = sum.toDouble() / count
        val median = if (count % 2 == 0) {
            (sortedTimings[count / 2 - 1] + sortedTimings[count / 2]) / 2.0
        } else {
            sortedTimings[count / 2].toDouble()
        }
        val p95 = sortedTimings[(count * 0.95).toInt().coerceAtMost(count - 1)]
        val p99 = sortedTimings[(count * 0.99).toInt().coerceAtMost(count - 1)]
        
        return PerformanceStats(
            operationType = operationType,
            count = count,
            totalTimeMs = sum,
            averageTimeMs = average,
            medianTimeMs = median,
            minTimeMs = sortedTimings.first(),
            maxTimeMs = sortedTimings.last(),
            p95TimeMs = p95,
            p99TimeMs = p99
        )
    }
    
    /**
     * Gets overall performance summary across all operations.
     */
    fun getOverallPerformanceStats(): OverallPerformanceStats {
        val allTimings = inferenceTimings.values.flatten()
        val totalCount = totalInferences.get()
        val totalTime = totalProcessingTime.get()
        
        val averageTime = if (totalCount > 0) totalTime.toDouble() / totalCount else 0.0
        
        return OverallPerformanceStats(
            totalInferences = totalCount,
            totalProcessingTimeMs = totalTime,
            averageProcessingTimeMs = averageTime,
            operationTypes = inferenceTimings.keys.toList(),
            memorySnapshotCount = memoryUsageHistory.size,
            batterySnapshotCount = batteryUsageHistory.size
        )
    }
    
    /**
     * Gets recent memory usage snapshots.
     * @param count Number of recent snapshots to return
     */
    fun getRecentMemoryUsage(count: Int = 10): List<MemoryUsageSnapshot> {
        synchronized(memoryUsageHistory) {
            return memoryUsageHistory.takeLast(count)
        }
    }
    
    /**
     * Gets recent battery usage snapshots.
     * @param count Number of recent snapshots to return
     */
    fun getRecentBatteryUsage(count: Int = 10): List<BatteryUsageSnapshot> {
        synchronized(batteryUsageHistory) {
            return batteryUsageHistory.takeLast(count)
        }
    }
    
    /**
     * Starts continuous monitoring of memory and battery usage.
     */
    fun startContinuousMonitoring() {
        CoroutineScope(Dispatchers.IO).launch {
            while (true) {
                captureMemoryUsage()
                captureBatteryUsage()
                kotlinx.coroutines.delay(MEMORY_SAMPLING_INTERVAL)
            }
        }
        Log.i(TAG, "Started continuous performance monitoring")
    }
    
    /**
     * Clears all collected metrics data.
     */
    fun clearMetrics() {
        inferenceTimings.clear()
        memoryUsageHistory.clear()
        batteryUsageHistory.clear()
        activeTimings.clear()
        totalInferences.set(0)
        totalProcessingTime.set(0)
        Log.i(TAG, "Cleared all performance metrics")
    }
    
    /**
     * Logs current performance summary to console.
     */
    fun logPerformanceSummary() {
        val overallStats = getOverallPerformanceStats()
        Log.i(TAG, "=== Performance Summary ===")
        Log.i(TAG, "Total inferences: ${overallStats.totalInferences}")
        Log.i(TAG, "Total processing time: ${overallStats.totalProcessingTimeMs}ms")
        Log.i(TAG, "Average processing time: ${"%.2f".format(overallStats.averageProcessingTimeMs)}ms")
        Log.i(TAG, "Operation types: ${overallStats.operationTypes.joinToString()}")
        Log.i(TAG, "Memory snapshots: ${overallStats.memorySnapshotCount}")
        Log.i(TAG, "Battery snapshots: ${overallStats.batterySnapshotCount}")
        
        // Log individual operation stats
        overallStats.operationTypes.forEach { operationType ->
            val stats = getPerformanceStats(operationType)
            stats?.let {
                Log.i(TAG, "$operationType: avg=${"%.2f".format(it.averageTimeMs)}ms, " +
                        "median=${"%.2f".format(it.medianTimeMs)}ms, " +
                        "p95=${it.p95TimeMs}ms, count=${it.count}")
            }
        }
    }
}

/**
 * Data class representing performance statistics for a specific operation type.
 */
data class PerformanceStats(
    val operationType: String,
    val count: Int,
    val totalTimeMs: Long,
    val averageTimeMs: Double,
    val medianTimeMs: Double,
    val minTimeMs: Long,
    val maxTimeMs: Long,
    val p95TimeMs: Long,
    val p99TimeMs: Long
)

/**
 * Data class representing overall performance statistics.
 */
data class OverallPerformanceStats(
    val totalInferences: Long,
    val totalProcessingTimeMs: Long,
    val averageProcessingTimeMs: Double,
    val operationTypes: List<String>,
    val memorySnapshotCount: Int,
    val batterySnapshotCount: Int
)

/**
 * Data class representing a memory usage snapshot.
 */
data class MemoryUsageSnapshot(
    val timestamp: Long,
    val totalMemoryMB: Long,
    val availableMemoryMB: Long,
    val usedMemoryMB: Long,
    val jvmHeapSizeMB: Long,
    val jvmHeapUsedMB: Long,
    val jvmHeapFreeMB: Long,
    val nativeHeapSizeMB: Long,
    val nativeHeapUsedMB: Long,
    val nativeHeapFreeMB: Long,
    val isLowMemory: Boolean
)

/**
 * Data class representing a battery usage snapshot.
 */
data class BatteryUsageSnapshot(
    val timestamp: Long,
    val batteryLevel: Int,
    val batteryStatus: Int,
    val batteryHealth: Int,
    val batteryTemperature: Float,
    val isCharging: Boolean
)