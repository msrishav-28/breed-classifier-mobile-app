package com.livestock.recognition.ui.results

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.livestock.recognition.data.models.BreedInfo
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.services.TypeClassificationService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * ViewModel for managing results display state and breed information loading.
 * Handles low confidence warnings and breed characteristic retrieval.
 */
class ResultsViewModel(application: Application) : AndroidViewModel(application) {
    
    private val typeClassificationService = TypeClassificationService(application)
    
    private val _resultsState = MutableLiveData<ResultsState>()
    val resultsState: LiveData<ResultsState> = _resultsState
    
    private val _warningMessage = MutableLiveData<String?>()
    val warningMessage: LiveData<String?> = _warningMessage
    
    companion object {
        private const val LOW_CONFIDENCE_THRESHOLD = 0.75f
    }
    
    fun loadResults(classificationResult: ClassificationResult, imagePath: String) {
        _resultsState.value = ResultsState.Loading
        
        viewModelScope.launch {
            try {
                val resultsData = withContext(Dispatchers.IO) {
                    // Initialize type classification service if needed
                    val initResult = typeClassificationService.initialize()
                    if (!initResult.isValid) {
                        throw Exception("Failed to initialize breed database: ${initResult.errors.joinToString()}")
                    }
                    
                    // Get breed information
                    val breedInfo = typeClassificationService.getBreedCharacteristics(classificationResult.breed)
                    
                    ResultsDisplayData(
                        classificationResult = classificationResult,
                        imagePath = imagePath,
                        breedInfo = breedInfo
                    )
                }
                
                _resultsState.value = ResultsState.Success(resultsData)
                
                // Check for low confidence warnings
                checkConfidenceWarnings(classificationResult)
                
            } catch (e: Exception) {
                _resultsState.value = ResultsState.Error("Failed to load results: ${e.message}")
            }
        }
    }
    
    private fun checkConfidenceWarnings(result: ClassificationResult) {
        val warnings = mutableListOf<String>()
        
        if (result.breedConfidence < LOW_CONFIDENCE_THRESHOLD) {
            warnings.add("Low breed identification confidence (${(result.breedConfidence * 100).toInt()}%)")
        }
        
        if (result.typeConfidence < LOW_CONFIDENCE_THRESHOLD) {
            warnings.add("Low type classification confidence (${(result.typeConfidence * 100).toInt()}%)")
        }
        
        if (warnings.isNotEmpty()) {
            val warningText = warnings.joinToString("\n") + 
                "\n\nConsider retaking the photo with better lighting and clearer view of the animal."
            _warningMessage.value = warningText
        } else {
            _warningMessage.value = null
        }
    }
    
    fun shareResults() {
        // This would trigger sharing functionality
        // Implementation would depend on integration with ReportGenerationService
    }
}

/**
 * Sealed class representing different results display states
 */
sealed class ResultsState {
    object Loading : ResultsState()
    data class Success(val data: ResultsDisplayData) : ResultsState()
    data class Error(val message: String) : ResultsState()
}

/**
 * Data class containing all information needed for results display
 */
data class ResultsDisplayData(
    val classificationResult: ClassificationResult,
    val imagePath: String,
    val breedInfo: BreedInfo?
)