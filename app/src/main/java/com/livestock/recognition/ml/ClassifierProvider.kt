package com.livestock.recognition.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Lazily initialises the classifier exactly once and exposes whether a model
 * is bundled at all, so the UI can degrade gracefully in builds that ship
 * without a trained model, such as local or test variants.
 */
class ClassifierProvider(context: Context) {

    sealed interface State {
        data class Ready(val classifier: BreedClassifier) : State
        data class Unavailable(val reason: String) : State
    }

    private val appContext = context.applicationContext
    private val mutex = Mutex()
    @Volatile
    private var state: State? = null

    /** Fast, non-blocking check used by the home screen. */
    fun isModelBundled(): Boolean =
        TfLiteBreedClassifier.isModelBundled(appContext.assets)

    suspend fun get(): State = state ?: mutex.withLock {
        state ?: withContext(Dispatchers.IO) {
            createState().also { state = it }
        }
    }

    private fun createState(): State = try {
        State.Ready(TfLiteBreedClassifier.create(appContext))
    } catch (e: ClassificationException) {
        Log.w(TAG, "Classifier unavailable: ${e.message}")
        State.Unavailable(e.message ?: "Model could not be loaded")
    }

    private companion object {
        const val TAG = "ClassifierProvider"
    }
}
