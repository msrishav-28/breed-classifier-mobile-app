package com.livestock.recognition.ui.results

import android.app.Application
import android.graphics.Bitmap
import androidx.annotation.StringRes
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.livestock.recognition.LivestockApp
import com.livestock.recognition.R
import com.livestock.recognition.core.classify.ConfidencePolicy
import com.livestock.recognition.core.model.BreedInfo
import com.livestock.recognition.core.model.ClassificationRecord
import com.livestock.recognition.core.quality.QualityIssue
import com.livestock.recognition.core.report.ReportContentBuilder
import com.livestock.recognition.image.BitmapLoader
import com.livestock.recognition.image.ImageQualityAnalyzer
import com.livestock.recognition.ml.ClassificationException
import com.livestock.recognition.ml.ClassifierProvider
import com.livestock.recognition.ui.common.formatDateTime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Drives the results screen for both flows: classifying a fresh image and
 * re-opening a stored history entry.
 */
class ResultsViewModel(application: Application) : AndroidViewModel(application) {

    sealed interface UiState {
        data object Loading : UiState

        data class Ready(
            val record: ClassificationRecord,
            val breedInfo: BreedInfo?,
            val qualityIssues: List<QualityIssue>,
            val imagePath: String,
            val showConfidenceWarning: Boolean,
        ) : UiState

        data class ModelUnavailable(val detail: String) : UiState

        data class Error(@StringRes val messageRes: Int) : UiState
    }

    sealed interface ShareEvent {
        data class Ready(val report: File) : ShareEvent
        data object Failed : ShareEvent
        data object InProgress : ShareEvent
    }

    private val container get() = getApplication<LivestockApp>().container

    private val _state = MutableLiveData<UiState>()
    val state: LiveData<UiState> = _state

    private val _shareEvent = MutableLiveData<ShareEvent?>()
    val shareEvent: LiveData<ShareEvent?> = _shareEvent

    private var started = false

    /** Classifies a freshly captured or picked image and stores the result. */
    fun startNewClassification(imagePath: String) {
        if (started) return
        started = true
        _state.value = UiState.Loading

        viewModelScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                BitmapLoader.decode(imagePath, DECODE_MAX_DIMENSION)
            }
            if (bitmap == null) {
                _state.value = UiState.Error(R.string.error_image_load)
                return@launch
            }

            val quality = withContext(Dispatchers.Default) {
                ImageQualityAnalyzer.analyze(bitmap)
            }

            when (val classifierState = container.classifierProvider.get()) {
                is ClassifierProvider.State.Unavailable -> {
                    _state.value = UiState.ModelUnavailable(classifierState.reason)
                }
                is ClassifierProvider.State.Ready -> {
                    try {
                        val output = classifierState.classifier.classify(bitmap)
                        val best = output.predictions.first()
                        val catalog = container.breedCatalogProvider.catalog
                        val record = ClassificationRecord(
                            breedLabel = best.label,
                            confidence = best.confidence,
                            animalType = catalog?.find(best.label)?.type,
                            alternatives = output.predictions.drop(1),
                            capturedAtEpochMillis = System.currentTimeMillis(),
                            processingTimeMillis = output.processingTimeMillis,
                            modelVersion = output.modelVersion,
                        )
                        container.historyRepository.save(record, imagePath)
                        _state.value = readyState(record, imagePath, quality.issues)
                    } catch (e: ClassificationException) {
                        _state.value = UiState.Error(R.string.error_classification)
                    }
                }
            }
        }
    }

    /** Re-opens a previously stored classification. */
    fun loadSavedClassification(id: Long) {
        if (started) return
        started = true
        _state.value = UiState.Loading

        viewModelScope.launch {
            val saved = container.historyRepository.get(id)
            if (saved == null) {
                _state.value = UiState.Error(R.string.error_record_missing)
            } else {
                _state.value = readyState(saved.record, saved.imagePath, emptyList())
            }
        }
    }

    fun shareReport() {
        val current = _state.value as? UiState.Ready ?: return
        if (_shareEvent.value == ShareEvent.InProgress) return
        _shareEvent.value = ShareEvent.InProgress

        viewModelScope.launch {
            try {
                val photo: Bitmap? = withContext(Dispatchers.IO) {
                    BitmapLoader.decode(current.imagePath, REPORT_IMAGE_MAX_DIMENSION)
                }
                val content = ReportContentBuilder.build(
                    record = current.record,
                    breedInfo = current.breedInfo,
                    generatedAt = formatDateTime(System.currentTimeMillis()),
                    capturedAt = formatDateTime(current.record.capturedAtEpochMillis),
                )
                val file = container.reportGenerator.generate(content, photo)
                photo?.recycle()
                _shareEvent.value = ShareEvent.Ready(file)
            } catch (e: Exception) {
                _shareEvent.value = ShareEvent.Failed
            }
        }
    }

    fun consumeShareEvent() {
        _shareEvent.value = null
    }

    private fun readyState(
        record: ClassificationRecord,
        imagePath: String,
        qualityIssues: List<QualityIssue>,
    ): UiState.Ready = UiState.Ready(
        record = record,
        breedInfo = container.breedCatalogProvider.catalog?.find(record.breedLabel),
        qualityIssues = qualityIssues,
        imagePath = imagePath,
        showConfidenceWarning = ConfidencePolicy.requiresWarning(record.confidence),
    )

    private companion object {
        const val DECODE_MAX_DIMENSION = 1280
        const val REPORT_IMAGE_MAX_DIMENSION = 800
    }
}
