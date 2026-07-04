package com.livestock.recognition.data.db

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(
    entities = [ClassificationEntity::class],
    version = 1,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun classificationDao(): ClassificationDao

    companion object {
        /** Kept in sync with the backup exclusion rules in res/xml. */
        const val NAME = "classifications.db"
    }
}
