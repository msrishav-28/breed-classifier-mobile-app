package com.livestock.recognition.core.catalog

/**
 * Canonicalises breed names so that model labels and catalog entries can be
 * matched reliably. Model labels conventionally use underscores
 * ("Red_Sindhi", "Nili_Ravi") while the catalog uses display names
 * ("Red Sindhi", "Nili-Ravi"); both normalise to the same key.
 */
object BreedNames {

    private val SEPARATORS = Regex("[_\\-\\s]+")

    /** Lower-cased, separator-insensitive lookup key for a breed name. */
    fun normalize(name: String): String =
        name.replace(SEPARATORS, " ").trim().lowercase()

    /** Turns a raw model label into a human-readable display name. */
    fun displayName(label: String): String =
        label.trim()
            .replace(SEPARATORS, " ")
            .split(" ")
            .filter { it.isNotEmpty() }
            .joinToString(" ") { word ->
                word.replaceFirstChar { it.uppercaseChar() }
            }
}
