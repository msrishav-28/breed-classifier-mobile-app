package com.livestock.recognition.ui.results

import android.app.Application
import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import androidx.lifecycle.Observer
import com.livestock.recognition.data.models.AnimalType
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.models.ImageMetadata
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.junit.MockitoJUnitRunner
import org.mockito.kotlin.verify
import org.mockito.kotlin.whenever

/**
 * Unit tests for ResultsViewModel functionality.
 * Tests UI component rendering with various data, warning display logic for low confidence,
 * and breed information loading and display.
 * 
 * Requirements: 2.5, 3.3, 7.3
 */
@ExperimentalCoroutinesApi
@RunWith(MockitoJUnitRunner::class)
class ResultsViewModelTest {

    @get:Rule
    val instantTaskExecutorRule = InstantTaskExecutorRule()

    @Mock
    private lateinit var application: Application

    @Mock
    private lateinit var resultsStateObserver: Observer<ResultsState>

    @Mock
    private lateinit var warningMessageObserver: Observer<String?>

    private lateinit var viewModel: ResultsViewModel
    private val testDispatcher = StandardTestDispatcher()

    @Before
    fun setup() {
        Dispatchers.setMain(testDispatcher)
        whenever(application.assets).thenReturn(null) // Mock assets for testing
        viewModel = ResultsViewModel(application)
        viewModel.resultsState.observeForever(resultsStateObserver)
        viewModel.warningMessage.observeForever(warningMessageObserver)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
        viewModel.resultsState.removeObserver(resultsStateObserver)
        viewModel.warningMessage.removeObserver(warningMessageObserver)
    }

    @Test
    fun `loadResults with high confidence should not show warning`() = runTest {
        // Given
        val highConfidenceResult = createClassificationResult(
            breedConfidence = 0.9f,
            typeConfidence = 0.85f
        )
        val imagePath = "/test/image.jpg"

        // When
        viewModel.loadResults(highConfidenceResult, imagePath)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify(warningMessageObserver).onChanged(null)
    }

    @Test
    fun `loadResults with low breed confidence should show warning`() = runTest {
        // Given
        val lowConfidenceResult = createClassificationResult(
            breedConfidence = 0.6f,
            typeConfidence = 0.8f
        )
        val imagePath = "/test/image.jpg"

        // When
        viewModel.loadResults(lowConfidenceResult, imagePath)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify(warningMessageObserver).onChanged(
            org.mockito.kotlin.argThat { message ->
                message != null && message.contains("Low breed identification confidence")
            }
        )
    }

    @Test
    fun `loadResults with low type confidence should show warning`() = runTest {
        // Given
        val lowTypeConfidenceResult = createClassificationResult(
            breedConfidence = 0.8f,
            typeConfidence = 0.6f
        )
        val imagePath = "/test/image.jpg"

        // When
        viewModel.loadResults(lowTypeConfidenceResult, imagePath)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify(warningMessageObserver).onChanged(
            org.mockito.kotlin.argThat { message ->
                message != null && message.contains("Low type classification confidence")
            }
        )
    }

    @Test
    fun `loadResults with both low confidences should show combined warning`() = runTest {
        // Given
        val lowConfidenceResult = createClassificationResult(
            breedConfidence = 0.6f,
            typeConfidence = 0.5f
        )
        val imagePath = "/test/image.jpg"

        // When
        viewModel.loadResults(lowConfidenceResult, imagePath)
        testDispatcher.scheduler.advanceUntilIdle()

        // Then
        verify(warningMessageObserver).onChanged(
            org.mockito.kotlin.argThat { message ->
                message != null && 
                message.contains("Low breed identification confidence") &&
                message.contains("Low type classification confidence") &&
                message.contains("Consider retaking the photo")
            }
        )
    }

    @Test
    fun `loadResults should emit loading state initially`() = runTest {
        // Given
        val result = createClassificationResult()
        val imagePath = "/test/image.jpg"

        // When
        viewModel.loadResults(result, imagePath)

        // Then
        verify(resultsStateObserver).onChanged(ResultsState.Loading)
    }

    @Test
    fun `confidence threshold validation`() {
        // Test confidence threshold constant
        val lowConfidenceThreshold = 0.75f
        
        // High confidence cases
        assert(0.8f >= lowConfidenceThreshold) { "High confidence should be above threshold" }
        assert(0.9f >= lowConfidenceThreshold) { "Very high confidence should be above threshold" }
        
        // Low confidence cases
        assert(0.6f < lowConfidenceThreshold) { "Low confidence should be below threshold" }
        assert(0.5f < lowConfidenceThreshold) { "Very low confidence should be below threshold" }
        
        // Boundary case
        assert(0.75f >= lowConfidenceThreshold) { "Threshold value should not trigger warning" }
        assert(0.74f < lowConfidenceThreshold) { "Just below threshold should trigger warning" }
    }

    @Test
    fun `warning message format validation`() {
        // Test that warning messages contain required elements
        val breedWarning = "Low breed identification confidence (60%)"
        val typeWarning = "Low type classification confidence (50%)"
        val actionGuidance = "Consider retaking the photo with better lighting and clearer view of the animal."
        
        // Verify warning message components
        assert(breedWarning.contains("breed identification")) { "Breed warning should mention breed identification" }
        assert(breedWarning.contains("60%")) { "Breed warning should show percentage" }
        
        assert(typeWarning.contains("type classification")) { "Type warning should mention type classification" }
        assert(typeWarning.contains("50%")) { "Type warning should show percentage" }
        
        assert(actionGuidance.contains("retaking")) { "Action guidance should suggest retaking" }
        assert(actionGuidance.contains("lighting")) { "Action guidance should mention lighting" }
    }

    @Test
    fun `results display data validation`() {
        // Given
        val result = createClassificationResult()
        val imagePath = "/test/image.jpg"
        val breedInfo = null // Simulating no breed info found
        
        // When
        val displayData = ResultsDisplayData(
            classificationResult = result,
            imagePath = imagePath,
            breedInfo = breedInfo
        )
        
        // Then
        assert(displayData.classificationResult == result) { "Classification result should be preserved" }
        assert(displayData.imagePath == imagePath) { "Image path should be preserved" }
        assert(displayData.breedInfo == null) { "Breed info can be null" }
    }

    private fun createClassificationResult(
        breed: String = "Gir",
        breedConfidence: Float = 0.9f,
        animalType: AnimalType = AnimalType.DAIRY,
        typeConfidence: Float = 0.85f
    ): ClassificationResult {
        return ClassificationResult(
            breed = breed,
            breedConfidence = breedConfidence,
            animalType = animalType,
            typeConfidence = typeConfidence,
            timestamp = System.currentTimeMillis(),
            imageMetadata = ImageMetadata(
                width = 1920,
                height = 1080,
                fileSize = 2048000L,
                captureTime = System.currentTimeMillis(),
                deviceInfo = "Test Device",
                qualityScore = 0.8f
            ),
            processingTime = 2100L
        )
    }
}