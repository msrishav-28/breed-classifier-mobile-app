package com.livestock.recognition.core.model

/**
 * Reference information about a single breed from the bundled catalog.
 *
 * @property name Human-readable breed name, e.g. "Red Sindhi".
 * @property species Zoological species, e.g. "Bos indicus" or "Bubalus bubalis".
 * @property origin Region the breed originates from.
 * @property type Primary economic use.
 * @property milkYieldMinLitresPerDay Lower bound of typical daily milk yield.
 * @property milkYieldMaxLitresPerDay Upper bound of typical daily milk yield.
 * @property characteristics Notable traits, already split into separate entries.
 */
data class BreedInfo(
    val name: String,
    val species: String,
    val origin: String,
    val type: AnimalType,
    val milkYieldMinLitresPerDay: Int,
    val milkYieldMaxLitresPerDay: Int,
    val characteristics: List<String>,
) {
    init {
        require(name.isNotBlank()) { "Breed name must not be blank" }
        require(milkYieldMinLitresPerDay >= 0) { "Milk yield must not be negative" }
        require(milkYieldMaxLitresPerDay >= milkYieldMinLitresPerDay) {
            "Milk yield range is inverted for breed '$name'"
        }
    }

    val hasMilkYield: Boolean
        get() = milkYieldMaxLitresPerDay > 0
}
