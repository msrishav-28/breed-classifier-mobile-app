package com.livestock.recognition.ui.camera

import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test

/**
 * Unit tests for CameraViewModel
 * Tests camera state management and lifecycle handling
 * 
 * Requirements: 1.1, 1.3, 1.5
 */
class CameraViewModelTest {
    
    @get:Rule
    val instantTaskExecutorRule = InstantTaskExecutorRule()
    
    private lateinit var viewModel: CameraViewModel
    
    @Before
    fun setUp() {
        viewModel = CameraViewModel()
    }
    
    @Test
    fun `initial state should be Initializing`() {
        val initialState = viewModel.cameraState.value
        assertTrue("Initial state should be Initializing", initialState is CameraState.Initializing)
    }
    
    @Test
    fun `setCameraReady should update state to Ready`() {
        viewModel.setCameraReady()
        
        val currentState = viewModel.cameraState.value
        assertTrue("State should be Ready after setCameraReady", currentState is CameraState.Ready)
    }
    
    @Test
    fun `setCameraCapturing should update state to Capturing`() {
        viewModel.setCameraCapturing()
        
        val currentState = viewModel.cameraState.value
        assertTrue("State should be Capturing after setCameraCapturing", currentState is CameraState.Capturing)
    }
    
    @Test
    fun `setCameraError should update state to Error with message`() {
        val errorMessage = "Camera initialization failed"
        viewModel.setCameraError(errorMessage)
        
        val currentState = viewModel.cameraState.value
        assertTrue("State should be Error after setCameraError", currentState is CameraState.Error)
        assertEquals("Error message should match", errorMessage, (currentState as CameraState.Error).message)
    }
    
    @Test
    fun `state transitions should work correctly`() {
        // Test typical flow: Initializing -> Ready -> Capturing -> Ready
        viewModel.setCameraReady()
        assertTrue("Should transition to Ready", viewModel.cameraState.value is CameraState.Ready)
        
        viewModel.setCameraCapturing()
        assertTrue("Should transition to Capturing", viewModel.cameraState.value is CameraState.Capturing)
        
        viewModel.setCameraReady()
        assertTrue("Should transition back to Ready", viewModel.cameraState.value is CameraState.Ready)
    }
    
    @Test
    fun `error state should preserve error message`() {
        val errorMessage = "Permission denied"
        viewModel.setCameraError(errorMessage)
        
        val errorState = viewModel.cameraState.value as CameraState.Error
        assertEquals("Error message should be preserved", errorMessage, errorState.message)
    }
    
    @Test
    fun `multiple state changes should update correctly`() {
        // Test rapid state changes
        viewModel.setCameraReady()
        viewModel.setCameraCapturing()
        viewModel.setCameraError("Test error")
        
        val finalState = viewModel.cameraState.value
        assertTrue("Final state should be Error", finalState is CameraState.Error)
        assertEquals("Error message should be correct", "Test error", (finalState as CameraState.Error).message)
    }
}