package com.livestock.recognition.di

import android.app.Application
import androidx.room.Room
import com.livestock.recognition.data.BreedCatalogProvider
import com.livestock.recognition.data.HistoryRepository
import com.livestock.recognition.data.db.AppDatabase
import com.livestock.recognition.ml.ClassifierProvider
import com.livestock.recognition.report.PdfReportGenerator

/**
 * Manual dependency container. The app is intentionally small enough that a
 * hand-rolled composition root is simpler and more transparent than a DI
 * framework; everything here is lazy so cold start stays cheap.
 */
class AppContainer(private val app: Application) {

    val database: AppDatabase by lazy {
        Room.databaseBuilder(app, AppDatabase::class.java, AppDatabase.NAME).build()
    }

    val historyRepository: HistoryRepository by lazy {
        HistoryRepository(database.classificationDao())
    }

    val breedCatalogProvider: BreedCatalogProvider by lazy {
        BreedCatalogProvider(app)
    }

    val classifierProvider: ClassifierProvider by lazy {
        ClassifierProvider(app)
    }

    val reportGenerator: PdfReportGenerator by lazy {
        PdfReportGenerator(app)
    }
}
