package com.livestock.recognition.data

import com.livestock.recognition.core.catalog.BreedCatalog
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File

/**
 * Validates the real catalog asset that ships in the APK, so a bad edit to
 * the CSV fails the build rather than degrading the app silently.
 */
class BundledCatalogTest {

    private fun catalogFile(): File {
        val candidates = listOf(
            File("src/main/assets/data/breed_mapping.csv"),
            File("app/src/main/assets/data/breed_mapping.csv"),
        )
        return candidates.firstOrNull { it.exists() }
            ?: error("Bundled catalog asset not found from ${File(".").absolutePath}")
    }

    @Test
    fun `bundled catalog parses without warnings`() {
        val result = catalogFile().bufferedReader().use { BreedCatalog.parse(it) }

        assertTrue(result is BreedCatalog.ParseResult.Success, "Catalog failed to parse: $result")
        result as BreedCatalog.ParseResult.Success
        assertTrue(result.warnings.isEmpty(), "Catalog has data problems: ${result.warnings}")
        assertTrue(result.catalog.size >= 20, "Catalog unexpectedly small: ${result.catalog.size}")
    }

    @Test
    fun `well-known breeds resolve by model label`() {
        val result = catalogFile().bufferedReader().use { BreedCatalog.parse(it) }
        result as BreedCatalog.ParseResult.Success

        listOf("Gir", "Red_Sindhi", "Nili_Ravi", "Murrah", "Krishna_Valley").forEach { label ->
            assertNotNull(result.catalog.find(label), "Label '$label' has no catalog entry")
        }
    }
}
