package com.livestock.recognition.services

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Monitors network connectivity state and provides reactive updates
 * Handles automatic synchronization when network becomes available
 */
@Singleton
class NetworkStateMonitor @Inject constructor(
    private val context: Context,
    private val offlineProcessingService: OfflineProcessingService
) {
    
    companion object {
        private const val TAG = "NetworkStateMonitor"
    }
    
    private val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    
    private val _networkState = MutableStateFlow(NetworkState.UNKNOWN)
    val networkState: StateFlow<NetworkState> = _networkState.asStateFlow()
    
    private val _isConnected = MutableStateFlow(false)
    val isConnected: StateFlow<Boolean> = _isConnected.asStateFlow()
    
    private var networkCallback: ConnectivityManager.NetworkCallback? = null
    private var wasOffline = false
    
    /**
     * Start monitoring network state
     */
    fun startMonitoring() {
        Log.d(TAG, "Starting network state monitoring...")
        
        // Check initial state
        updateNetworkState()
        
        // Register network callback
        val networkRequest = NetworkRequest.Builder()
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .addCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
            .build()
        
        networkCallback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                Log.d(TAG, "Network available: $network")
                handleNetworkAvailable()
            }
            
            override fun onLost(network: Network) {
                Log.d(TAG, "Network lost: $network")
                handleNetworkLost()
            }
            
            override fun onCapabilitiesChanged(network: Network, networkCapabilities: NetworkCapabilities) {
                Log.d(TAG, "Network capabilities changed: $network")
                updateNetworkState()
            }
            
            override fun onUnavailable() {
                Log.d(TAG, "Network unavailable")
                handleNetworkLost()
            }
        }
        
        connectivityManager.registerNetworkCallback(networkRequest, networkCallback!!)
        Log.d(TAG, "Network monitoring started")
    }
    
    /**
     * Stop monitoring network state
     */
    fun stopMonitoring() {
        Log.d(TAG, "Stopping network state monitoring...")
        
        networkCallback?.let { callback ->
            try {
                connectivityManager.unregisterNetworkCallback(callback)
                Log.d(TAG, "Network callback unregistered")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to unregister network callback", e)
            }
        }
        
        networkCallback = null
        Log.d(TAG, "Network monitoring stopped")
    }
    
    /**
     * Handle network becoming available
     */
    private fun handleNetworkAvailable() {
        updateNetworkState()
        
        if (wasOffline && _isConnected.value) {
            Log.d(TAG, "Network restored, triggering synchronization...")
            wasOffline = false
            
            // Trigger automatic synchronization
            // This would typically be done in a background coroutine
            // offlineProcessingService.synchronizeWhenOnline()
        }
    }
    
    /**
     * Handle network becoming unavailable
     */
    private fun handleNetworkLost() {
        wasOffline = true
        updateNetworkState()
        Log.d(TAG, "Switched to offline mode")
    }
    
    /**
     * Update current network state
     */
    private fun updateNetworkState() {
        val activeNetwork = connectivityManager.activeNetwork
        val networkCapabilities = activeNetwork?.let { connectivityManager.getNetworkCapabilities(it) }
        
        val newState = when {
            activeNetwork == null -> NetworkState.DISCONNECTED
            networkCapabilities == null -> NetworkState.DISCONNECTED
            !networkCapabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) -> NetworkState.DISCONNECTED
            !networkCapabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED) -> NetworkState.CONNECTING
            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> NetworkState.CONNECTED_WIFI
            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> NetworkState.CONNECTED_CELLULAR
            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> NetworkState.CONNECTED_ETHERNET
            else -> NetworkState.CONNECTED_OTHER
        }
        
        val isConnected = newState != NetworkState.DISCONNECTED && newState != NetworkState.CONNECTING
        
        if (_networkState.value != newState) {
            Log.d(TAG, "Network state changed: ${_networkState.value} -> $newState")
            _networkState.value = newState
        }
        
        if (_isConnected.value != isConnected) {
            Log.d(TAG, "Connection state changed: ${_isConnected.value} -> $isConnected")
            _isConnected.value = isConnected
        }
    }
    
    /**
     * Get current network type
     */
    fun getCurrentNetworkType(): NetworkType {
        val activeNetwork = connectivityManager.activeNetwork
        val networkCapabilities = activeNetwork?.let { connectivityManager.getNetworkCapabilities(it) }
        
        return when {
            networkCapabilities == null -> NetworkType.NONE
            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> NetworkType.WIFI
            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> NetworkType.CELLULAR
            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> NetworkType.ETHERNET
            else -> NetworkType.OTHER
        }
    }
    
    /**
     * Check if network is metered (limited data)
     */
    fun isNetworkMetered(): Boolean {
        val activeNetwork = connectivityManager.activeNetwork
        val networkCapabilities = activeNetwork?.let { connectivityManager.getNetworkCapabilities(it) }
        
        return networkCapabilities?.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) == false
    }
    
    /**
     * Get network connection quality
     */
    fun getConnectionQuality(): ConnectionQuality {
        val activeNetwork = connectivityManager.activeNetwork
        val networkCapabilities = activeNetwork?.let { connectivityManager.getNetworkCapabilities(it) }
        
        return when {
            networkCapabilities == null -> ConnectionQuality.NONE
            !networkCapabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) -> ConnectionQuality.NONE
            !networkCapabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED) -> ConnectionQuality.POOR
            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> {
                // Could check signal strength for WiFi if available
                ConnectionQuality.GOOD
            }
            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> {
                // Could check cellular signal strength if available
                ConnectionQuality.FAIR
            }
            else -> ConnectionQuality.FAIR
        }
    }
    
    /**
     * Force network state update
     */
    fun refreshNetworkState() {
        Log.d(TAG, "Forcing network state refresh...")
        updateNetworkState()
    }
    
    /**
     * Check if suitable for large data operations
     */
    fun isSuitableForLargeOperations(): Boolean {
        return _isConnected.value && 
               !isNetworkMetered() && 
               getConnectionQuality() != ConnectionQuality.POOR
    }
    
    /**
     * Check if suitable for synchronization
     */
    fun isSuitableForSync(): Boolean {
        return _isConnected.value && getConnectionQuality() != ConnectionQuality.NONE
    }
}

/**
 * Network connection states
 */
enum class NetworkState {
    UNKNOWN,
    DISCONNECTED,
    CONNECTING,
    CONNECTED_WIFI,
    CONNECTED_CELLULAR,
    CONNECTED_ETHERNET,
    CONNECTED_OTHER
}

/**
 * Network types
 */
enum class NetworkType {
    NONE,
    WIFI,
    CELLULAR,
    ETHERNET,
    OTHER
}

/**
 * Connection quality levels
 */
enum class ConnectionQuality {
    NONE,
    POOR,
    FAIR,
    GOOD,
    EXCELLENT
}