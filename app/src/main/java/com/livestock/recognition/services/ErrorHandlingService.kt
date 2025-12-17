package com.livestock.recognition.services

import android.content.Context
import android.util.Log
import kotlinx.coroutines.delay
import kotlin.random.Random

/**
 * Service for implementing graceful degradation, retry mechanisms, and user-friendly error handling.
 * Provides error recovery strategies and actionable guidance for users.
 * 
 * Requirements: 6.3, 7.4
 */
class ErrorHandlingService(private val context: Context) {
    
    companion object {
        private const val TAG = "ErrorHandling"
        private const val MAX_RETRY_ATTEMPTS = 3
        private const val BASE_RETRY_DELAY_MS = 1000L
        private const val MAX_RETRY_DELAY_MS = 8000L
    }
    
    /**
     * Executes an operation with retry logic and exponential backoff.
     * @param operation The operation to execute
     * @param maxAttempts Maximum number of retry attempts
     * @param onRetry Callback called before each retry attempt
     * @return Result of the operation or throws the last exception
     */
    suspend fun <T> executeWithRetry(
        operation: suspend () -> T,
        maxAttempts: Int = MAX_RETRY_ATTEMPTS,
        onRetry: ((attempt: Int, exception: Exception) -> Unit)? = null
    ): T {
        var lastException: Exception? = null
        
        repeat(maxAttempts) { attempt ->
            try {
                return operation()
            } catch (e: Exception) {
                lastException = e
                Log.w(TAG, "Operation failed on attempt ${attempt + 1}/$maxAttempts", e)
                
                if (attempt < maxAttempts - 1) {
                    onRetry?.invoke(attempt + 1, e)
                    val delayMs = calculateRetryDelay(attempt)
                    Log.d(TAG, "Retrying in ${delayMs}ms...")
                    delay(delayMs)
                }
            }
        }
        
        throw lastException ?: Exception("Operation failed after $maxAttempts attempts")
    }
    
    /**
     * Implements graceful degradation for ML model failures.
     * Falls back to simpler models or cached results when primary models fail.
     */
    suspend fun <T> executeWithGracefulDegradation(
        primaryOperation: suspend () -> T,
        fallbackOperation: suspend () -> T,
        onFallback: ((exception: Exception) -> Unit)? = null
    ): T {
        return try {
            primaryOperation()
        } catch (e: Exception) {
            Log.w(TAG, "Primary operation failed, attempting fallback", e)
            onFallback?.invoke(e)
            
            try {
                fallbackOperation()
            } catch (fallbackException: Exception) {
                Log.e(TAG, "Fallback operation also failed", fallbackException)
                throw GracefulDegradationException(
                    "Both primary and fallback operations failed",
                    e,
                    fallbackException
                )
            }
        }
    }
    
    /**
     * Converts technical exceptions into user-friendly error messages with actionable guidance.
     */
    fun getUserFriendlyErrorMessage(exception: Exception): UserFriendlyError {
        return when (exception) {
            is OutOfMemoryError -> UserFriendlyError(
                title = "Memory Error",
                message = "The app is running low on memory.",
                actionableGuidance = listOf(
                    "Close other apps to free up memory",
                    "Restart the app",
                    "Try using a smaller image"
                ),
                isRetryable = true,
                errorCode = "MEMORY_ERROR"
            )
            
            is SecurityException -> UserFriendlyError(
                title = "Permission Error",
                message = "The app needs permission to access your camera or storage.",
                actionableGuidance = listOf(
                    "Go to Settings > Apps > Cattle Recognition",
                    "Enable Camera and Storage permissions",
                    "Restart the app"
                ),
                isRetryable = true,
                errorCode = "PERMISSION_ERROR"
            )
            
            is java.io.FileNotFoundException -> UserFriendlyError(
                title = "File Not Found",
                message = "The image file could not be found or accessed.",
                actionableGuidance = listOf(
                    "Make sure the image file exists",
                    "Try capturing a new image",
                    "Check if storage is full"
                ),
                isRetryable = true,
                errorCode = "FILE_NOT_FOUND"
            )
            
            is java.net.UnknownHostException, is java.net.ConnectException -> UserFriendlyError(
                title = "Network Error",
                message = "Unable to connect to the internet.",
                actionableGuidance = listOf(
                    "Check your internet connection",
                    "Try switching between WiFi and mobile data",
                    "The app can work offline for basic features"
                ),
                isRetryable = true,
                errorCode = "NETWORK_ERROR"
            )
            
            is IllegalArgumentException -> UserFriendlyError(
                title = "Invalid Input",
                message = "The provided image or data is not valid.",
                actionableGuidance = listOf(
                    "Try capturing a new image",
                    "Make sure the image shows a clear view of the animal",
                    "Check that the image is not corrupted"
                ),
                isRetryable = true,
                errorCode = "INVALID_INPUT"
            )
            
            is ModelLoadException -> UserFriendlyError(
                title = "Model Loading Error",
                message = "Unable to load the AI recognition model.",
                actionableGuidance = listOf(
                    "Restart the app",
                    "Check if you have enough storage space",
                    "Update the app if available"
                ),
                isRetryable = true,
                errorCode = "MODEL_LOAD_ERROR"
            )
            
            is InferenceException -> UserFriendlyError(
                title = "Recognition Error",
                message = "Unable to analyze the image.",
                actionableGuidance = listOf(
                    "Try with a clearer image",
                    "Ensure good lighting conditions",
                    "Make sure the animal is clearly visible"
                ),
                isRetryable = true,
                errorCode = "INFERENCE_ERROR"
            )
            
            is GracefulDegradationException -> UserFriendlyError(
                title = "Service Degraded",
                message = "Some features may not work as expected.",
                actionableGuidance = listOf(
                    "Basic recognition is still available",
                    "Try again later for full features",
                    "Restart the app if problems persist"
                ),
                isRetryable = true,
                errorCode = "DEGRADED_SERVICE"
            )
            
            else -> UserFriendlyError(
                title = "Unexpected Error",
                message = "An unexpected error occurred.",
                actionableGuidance = listOf(
                    "Try the operation again",
                    "Restart the app if the problem persists",
                    "Contact support if the issue continues"
                ),
                isRetryable = true,
                errorCode = "UNKNOWN_ERROR"
            )
        }
    }
    
    /**
     * Determines if an error is transient and worth retrying.
     */
    fun isTransientError(exception: Exception): Boolean {
        return when (exception) {
            is java.net.SocketTimeoutException,
            is java.net.ConnectException,
            is java.io.IOException,
            is java.util.concurrent.TimeoutException -> true
            
            is OutOfMemoryError -> false // Usually not transient
            is SecurityException -> false // Requires user action
            is IllegalArgumentException -> false // Input validation error
            
            else -> {
                // For unknown exceptions, consider them potentially transient
                // but log for investigation
                Log.w(TAG, "Unknown exception type, treating as potentially transient", exception)
                true
            }
        }
    }
    
    /**
     * Calculates retry delay using exponential backoff with jitter.
     */
    private fun calculateRetryDelay(attempt: Int): Long {
        val exponentialDelay = BASE_RETRY_DELAY_MS * (1L shl attempt) // 2^attempt
        val jitter = Random.nextLong(0, BASE_RETRY_DELAY_MS / 2) // Add randomness
        return (exponentialDelay + jitter).coerceAtMost(MAX_RETRY_DELAY_MS)
    }
    
    /**
     * Logs error details for debugging and analytics.
     */
    fun logError(
        exception: Exception,
        context: String,
        additionalData: Map<String, Any> = emptyMap()
    ) {
        Log.e(TAG, "Error in $context", exception)
        
        // Log additional context data
        additionalData.forEach { (key, value) ->
            Log.d(TAG, "Context[$context] $key: $value")
        }
        
        // In a production app, this would also send to crash reporting service
        // like Firebase Crashlytics or Bugsnag
    }
    
    /**
     * Creates a recovery strategy based on the type of error.
     */
    fun createRecoveryStrategy(exception: Exception): RecoveryStrategy {
        return when (exception) {
            is OutOfMemoryError -> RecoveryStrategy(
                type = RecoveryType.RESTART_REQUIRED,
                description = "Memory cleanup required",
                automaticRecovery = false,
                userAction = "Please restart the app"
            )
            
            is SecurityException -> RecoveryStrategy(
                type = RecoveryType.PERMISSION_REQUIRED,
                description = "Permission needs to be granted",
                automaticRecovery = false,
                userAction = "Grant required permissions in settings"
            )
            
            is java.net.UnknownHostException -> RecoveryStrategy(
                type = RecoveryType.RETRY_WITH_DELAY,
                description = "Network connectivity issue",
                automaticRecovery = true,
                userAction = "Check internet connection"
            )
            
            is ModelLoadException -> RecoveryStrategy(
                type = RecoveryType.FALLBACK_AVAILABLE,
                description = "Primary model failed, using backup",
                automaticRecovery = true,
                userAction = "Some features may be limited"
            )
            
            else -> RecoveryStrategy(
                type = RecoveryType.RETRY_IMMEDIATE,
                description = "Transient error",
                automaticRecovery = true,
                userAction = "Please try again"
            )
        }
    }
}

/**
 * Data class representing a user-friendly error message with actionable guidance.
 */
data class UserFriendlyError(
    val title: String,
    val message: String,
    val actionableGuidance: List<String>,
    val isRetryable: Boolean,
    val errorCode: String
)

/**
 * Data class representing a recovery strategy for different types of errors.
 */
data class RecoveryStrategy(
    val type: RecoveryType,
    val description: String,
    val automaticRecovery: Boolean,
    val userAction: String
)

/**
 * Enum representing different types of recovery strategies.
 */
enum class RecoveryType {
    RETRY_IMMEDIATE,
    RETRY_WITH_DELAY,
    FALLBACK_AVAILABLE,
    PERMISSION_REQUIRED,
    RESTART_REQUIRED,
    NO_RECOVERY
}

/**
 * Custom exception for model loading failures.
 */
class ModelLoadException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Custom exception for inference failures.
 */
class InferenceException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Custom exception for graceful degradation scenarios.
 */
class GracefulDegradationException(
    message: String,
    val primaryException: Exception,
    val fallbackException: Exception
) : Exception(message)