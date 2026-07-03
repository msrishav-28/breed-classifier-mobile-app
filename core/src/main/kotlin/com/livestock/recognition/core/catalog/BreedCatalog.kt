package com.livestock.recognition.core.catalog

import com.livestock.recognition.core.model.AnimalType
import com.livestock.recognition.core.model.BreedInfo
import java.io.Reader

/**
 * In-memory breed reference catalog, keyed by normalised breed name so that
 * model labels ("Red_Sindhi") and display names ("Red Sindhi") both resolve.
 */
class BreedCatalog private constructor(
    private val breedsByKey: Map<String, BreedInfo>,
) {

    /** All breeds in catalog order. */
    val breeds: List<BreedInfo> = breedsByKey.values.toList()

    val size: Int get() = breedsByKey.size

    /** Looks a breed up by display name or model label; null when unknown. */
    fun find(name: String): BreedInfo? = breedsByKey[BreedNames.normalize(name)]

    fun contains(name: String): Boolean = find(name) != null

    fun byType(type: AnimalType): List<BreedInfo> = breeds.filter { it.type == type }

    companion object {

        private const val EXPECTED_HEADER =
            "breed_name,scientific_name,origin,animal_type,milk_yield_min,milk_yield_max,characteristics"
        private const val FIELD_COUNT = 7
        private const val CHARACTERISTIC_SEPARATOR = "|"

        /**
         * Parses catalog CSV data.
         *
         * Structural problems (missing or wrong header) fail the parse;
         * individual malformed rows are skipped and reported as warnings so
         * one bad row cannot take the whole catalog down.
         */
        fun parse(reader: Reader): ParseResult {
            val lines = reader.readLines()
            val header = lines.firstOrNull()?.trim()
                ?: return ParseResult.Failure("Catalog is empty")
            if (header != EXPECTED_HEADER) {
                return ParseResult.Failure(
                    "Unexpected catalog header: '$header' (expected '$EXPECTED_HEADER')"
                )
            }

            val warnings = mutableListOf<String>()
            val breedsByKey = LinkedHashMap<String, BreedInfo>()

            lines.drop(1).forEachIndexed { index, rawLine ->
                val lineNumber = index + 2
                if (rawLine.isBlank()) return@forEachIndexed

                val breed = parseRow(rawLine, lineNumber, warnings) ?: return@forEachIndexed
                val key = BreedNames.normalize(breed.name)
                if (breedsByKey.containsKey(key)) {
                    warnings.add("Line $lineNumber: duplicate breed '${breed.name}' ignored")
                } else {
                    breedsByKey[key] = breed
                }
            }

            if (breedsByKey.isEmpty()) {
                return ParseResult.Failure("Catalog contains no valid breed rows")
            }
            return ParseResult.Success(BreedCatalog(breedsByKey), warnings)
        }

        private fun parseRow(
            rawLine: String,
            lineNumber: Int,
            warnings: MutableList<String>,
        ): BreedInfo? {
            val fields = try {
                CsvParser.splitLine(rawLine)
            } catch (e: IllegalArgumentException) {
                warnings.add("Line $lineNumber: ${e.message}")
                return null
            }
            if (fields.size != FIELD_COUNT) {
                warnings.add("Line $lineNumber: expected $FIELD_COUNT fields, found ${fields.size}")
                return null
            }

            val name = fields[0].trim()
            val type = AnimalType.fromToken(fields[3])
            if (type == null) {
                warnings.add("Line $lineNumber: unknown animal type '${fields[3].trim()}'")
                return null
            }
            val yieldMin = fields[4].trim().toIntOrNull()
            val yieldMax = fields[5].trim().toIntOrNull()
            if (yieldMin == null || yieldMax == null) {
                warnings.add("Line $lineNumber: milk yield is not numeric")
                return null
            }

            return try {
                BreedInfo(
                    name = name,
                    species = fields[1].trim(),
                    origin = fields[2].trim(),
                    type = type,
                    milkYieldMinLitresPerDay = yieldMin,
                    milkYieldMaxLitresPerDay = yieldMax,
                    characteristics = fields[6]
                        .split(CHARACTERISTIC_SEPARATOR)
                        .map { it.trim() }
                        .filter { it.isNotEmpty() },
                )
            } catch (e: IllegalArgumentException) {
                warnings.add("Line $lineNumber: ${e.message}")
                null
            }
        }
    }

    sealed interface ParseResult {
        data class Success(val catalog: BreedCatalog, val warnings: List<String>) : ParseResult
        data class Failure(val reason: String) : ParseResult
    }
}
