package com.livestock.recognition.ui.camera

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

/**
 * ViewModel for managing camera state and operations.
 * Handles camera lifecycle states and error management.
 */
class CameraViewModel : ViewModel() {
    
    private val _cameraState = MutableLiveData<CameraState>()
    val cameraState: LiveData<CameraState> = _cameraState
    
    init {
        _cameraState.value = CameraState.Initializing
    }
    
    fun setCameraReady() {
        _cameraState.value = CameraState.Ready
    }
    
    fun setCameraCapturing() {
        _cameraState.value = CameraState.Capturing
    }
    
    fun setCameraError(message: String) {
        _cameraState.value = CameraState.Error(message)
    }
}

/**
 * Sealed class representing different camera states
 */
sealed class CameraState {
    object Initializing : CameraState()
    object Ready : CameraState()
    object Capturing : CameraState()
    data class Error(val message: String) : CameraState()
}