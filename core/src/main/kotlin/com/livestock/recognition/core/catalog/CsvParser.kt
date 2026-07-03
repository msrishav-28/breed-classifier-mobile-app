package com.livestock.recognition.core.catalog

/**
 * Minimal CSV field splitter supporting double-quoted fields with embedded
 * commas and escaped quotes (RFC 4180 style), which is all the bundled
 * catalog requires. Kept internal to the catalog package on purpose: this is
 * not a general-purpose CSV library.
 */
internal object CsvParser {

    /**
     * Splits one CSV line into fields.
     *
     * @throws IllegalArgumentException if a quoted field is left unterminated.
     */
    fun splitLine(line: String): List<String> {
        val fields = mutableListOf<String>()
        val current = StringBuilder()
        var inQuotes = false
        var i = 0

        while (i < line.length) {
            val ch = line[i]
            when {
                inQuotes && ch == '"' && i + 1 < line.length && line[i + 1] == '"' -> {
                    current.append('"')
                    i++
                }
                ch == '"' -> inQuotes = !inQuotes
                ch == ',' && !inQuotes -> {
                    fields.add(current.toString())
                    current.setLength(0)
                }
                else -> current.append(ch)
            }
            i++
        }
        require(!inQuotes) { "Unterminated quoted field in line: $line" }
        fields.add(current.toString())
        return fields
    }
}
