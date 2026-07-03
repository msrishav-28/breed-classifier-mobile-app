package com.livestock.recognition

import android.app.Application
import com.livestock.recognition.di.AppContainer

/**
 * Application entry point. Owns the [AppContainer], the single composition
 * root through which every activity and view model obtains dependencies.
 */
class LivestockApp : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}
