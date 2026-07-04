package com.livestock.recognition.core.model

/**
 * Primary economic use of a breed.
 */
enum class AnimalType {
    DAIRY,
    DRAUGHT,
    DUAL_PURPOSE;

    companion object {
        /**
         * Parses a catalog token such as `dairy`, `draught` or `dual_purpose`.
         * Returns null for unknown tokens so callers can decide how to handle
         * bad data instead of silently defaulting.
         */
        fun fromToken(token: String): AnimalType? = when (token.trim().lowercase()) {
            "dairy" -> DAIRY
            "draught", "draft" -> DRAUGHT
            "dual_purpose", "dual-purpose", "dual purpose" -> DUAL_PURPOSE
            else -> null
        }
    }
}
