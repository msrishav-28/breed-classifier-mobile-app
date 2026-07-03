package com.livestock.recognition.ui.common

import androidx.annotation.StringRes
import com.livestock.recognition.R
import com.livestock.recognition.core.model.AnimalType
import com.livestock.recognition.core.quality.QualityIssue
import java.text.DateFormat
import java.util.Date
import kotlin.math.roundToInt

/**
 * Mapping between domain values and localisable UI resources. Keeping the
 * domain enums free of Android types happens here and nowhere else.
 */

@StringRes
fun AnimalType.displayNameRes(): Int = when (this) {
    AnimalType.DAIRY -> R.string.animal_type_dairy
    AnimalType.DRAUGHT -> R.string.animal_type_draught
    AnimalType.DUAL_PURPOSE -> R.string.animal_type_dual_purpose
}

@StringRes
fun AnimalType.descriptionRes(): Int = when (this) {
    AnimalType.DAIRY -> R.string.animal_type_dairy_description
    AnimalType.DRAUGHT -> R.string.animal_type_draught_description
    AnimalType.DUAL_PURPOSE -> R.string.animal_type_dual_purpose_description
}

@StringRes
fun QualityIssue.messageRes(): Int = when (this) {
    QualityIssue.LOW_RESOLUTION -> R.string.quality_issue_low_resolution
    QualityIssue.TOO_DARK -> R.string.quality_issue_too_dark
    QualityIssue.TOO_BRIGHT -> R.string.quality_issue_too_bright
    QualityIssue.LOW_CONTRAST -> R.string.quality_issue_low_contrast
    QualityIssue.BLURRY -> R.string.quality_issue_blurry
}

fun formatDateTime(epochMillis: Long): String =
    DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
        .format(Date(epochMillis))

fun confidencePercent(confidence: Float): Int =
    (confidence * 100).roundToInt().coerceIn(0, 100)
