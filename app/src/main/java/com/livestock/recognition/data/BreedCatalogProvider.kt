package com.livestock.recognition.data

import android.content.Context
import android.util.Log
import com.livestock.recognition.core.catalog.BreedCatalog
import java.io.IOException

/**
 * Loads the bundled breed catalog once and caches it. A missing or corrupt
 * catalog asset degrades to an empty lookup (breed info panels simply don't
 * render) instead of failing classification.
 */
class BreedCatalogProvider(context: Context) {

    private val appContext = context.applicationContext

    val catalog: BreedCatalog? by lazy { load() }

    private fun load(): BreedCatalog? {
        val result = try {
            appContext.assets.open(CATALOG_ASSET).bufferedReader().use { reader ->
                BreedCatalog.parse(reader)
            }
        } catch (e: IOException) {
            Log.e(TAG, "Breed catalog asset missing", e)
            return null
        }

        return when (result) {
            is BreedCatalog.ParseResult.Success -> {
                result.warnings.forEach { Log.w(TAG, "Catalog: $it") }
                result.catalog
            }
            is BreedCatalog.ParseResult.Failure -> {
                Log.e(TAG, "Breed catalog unusable: ${result.reason}")
                null
            }
        }
    }

    private companion object {
        const val TAG = "BreedCatalogProvider"
        const val CATALOG_ASSET = "data/breed_mapping.csv"
    }
}
