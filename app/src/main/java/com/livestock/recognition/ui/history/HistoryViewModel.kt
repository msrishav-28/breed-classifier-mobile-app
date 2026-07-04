package com.livestock.recognition.ui.history

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.asLiveData
import androidx.lifecycle.viewModelScope
import com.livestock.recognition.LivestockApp
import com.livestock.recognition.data.SavedClassification
import com.livestock.recognition.image.ImageFiles
import kotlinx.coroutines.launch

class HistoryViewModel(application: Application) : AndroidViewModel(application) {

    private val container get() = getApplication<LivestockApp>().container

    val entries: LiveData<List<SavedClassification>> =
        container.historyRepository.observeAll().asLiveData()

    /** Removes one entry and its stored photo. */
    fun delete(id: Long) {
        viewModelScope.launch {
            val orphanedImage = container.historyRepository.delete(id)
            ImageFiles.delete(orphanedImage)
        }
    }

    /** Clears the entire history including stored photos. */
    fun clearAll() {
        viewModelScope.launch {
            container.historyRepository.clear().forEach { ImageFiles.delete(it) }
        }
    }
}
