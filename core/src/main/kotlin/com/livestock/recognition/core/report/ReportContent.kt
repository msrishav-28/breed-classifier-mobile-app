package com.livestock.recognition.core.report

import com.livestock.recognition.core.catalog.BreedNames
import com.livestock.recognition.core.classify.ConfidencePolicy
import com.livestock.recognition.core.model.BreedInfo
import com.livestock.recognition.core.model.ClassificationRecord
import java.util.Locale

data class ReportField(val label: String, val value: String)

data class ReportSection(val title: String, val fields: List<ReportField>)

/**
 * Renderer-agnostic report structure. The app renders this to PDF; tests
 * assert on it directly without touching any drawing code.
 */
data class ReportContent(
    val title: String,
    val generatedAt: String,
    val sections: List<ReportSection>,
)

object ReportContentBuilder {

    private const val TITLE = "Livestock Breed Classification Report"

    /**
     * @param generatedAt pre-formatted local timestamp; formatting stays in
     *   the app layer to keep this module free of platform date APIs
     */
    fun build(
        record: ClassificationRecord,
        breedInfo: BreedInfo?,
        generatedAt: String,
        capturedAt: String,
    ): ReportContent {
        val sections = mutableListOf(
            ReportSection(
                title = "Classification",
                fields = buildList {
                    add(ReportField("Breed", BreedNames.displayName(record.breedLabel)))
                    add(ReportField("Confidence", formatPercent(record.confidence)))
                    add(
                        ReportField(
                            "Confidence level",
                            ConfidencePolicy.levelFor(record.confidence).name
                                .lowercase().replaceFirstChar { it.uppercase() },
                        )
                    )
                    record.animalType?.let {
                        add(ReportField("Primary use", formatEnum(it.name)))
                    }
                    if (record.alternatives.isNotEmpty()) {
                        add(
                            ReportField(
                                "Other candidates",
                                record.alternatives.joinToString(", ") {
                                    "${BreedNames.displayName(it.label)} (${formatPercent(it.confidence)})"
                                },
                            )
                        )
                    }
                },
            ),
        )

        breedInfo?.let { info ->
            sections.add(
                ReportSection(
                    title = "Breed information",
                    fields = buildList {
                        add(ReportField("Species", info.species))
                        add(ReportField("Origin", info.origin))
                        add(ReportField("Primary use", formatEnum(info.type.name)))
                        if (info.hasMilkYield) {
                            add(
                                ReportField(
                                    "Typical milk yield",
                                    "${info.milkYieldMinLitresPerDay}-${info.milkYieldMaxLitresPerDay} litres/day",
                                )
                            )
                        }
                        if (info.characteristics.isNotEmpty()) {
                            add(ReportField("Characteristics", info.characteristics.joinToString(", ")))
                        }
                    },
                )
            )
        }

        sections.add(
            ReportSection(
                title = "Processing details",
                fields = buildList {
                    add(ReportField("Captured", capturedAt))
                    add(ReportField("Processing time", "${record.processingTimeMillis} ms"))
                    record.modelVersion?.let { add(ReportField("Model", it)) }
                    add(ReportField("Analysis", "Performed entirely on device"))
                },
            )
        )

        return ReportContent(title = TITLE, generatedAt = generatedAt, sections = sections)
    }

    private fun formatPercent(fraction: Float): String =
        String.format(Locale.ROOT, "%.1f%%", fraction * 100)

    private fun formatEnum(name: String): String =
        name.lowercase().replace('_', '-').replaceFirstChar { it.uppercase() }
}
