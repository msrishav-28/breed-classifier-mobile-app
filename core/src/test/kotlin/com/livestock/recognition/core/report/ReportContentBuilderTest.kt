package com.livestock.recognition.core.report

import com.livestock.recognition.core.model.AnimalType
import com.livestock.recognition.core.model.BreedInfo
import com.livestock.recognition.core.model.ClassificationRecord
import com.livestock.recognition.core.model.Prediction
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.collections.shouldContain
import io.kotest.matchers.collections.shouldNotContain
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain

private fun record(
    breed: String = "Red_Sindhi",
    confidence: Float = 0.87f,
    alternatives: List<Prediction> = listOf(Prediction("Gir", 0.08f)),
) = ClassificationRecord(
    breedLabel = breed,
    confidence = confidence,
    animalType = AnimalType.DAIRY,
    alternatives = alternatives,
    capturedAtEpochMillis = 1_700_000_000_000,
    processingTimeMillis = 420,
    modelVersion = "breed_classifier v1",
)

private fun breedInfo() = BreedInfo(
    name = "Red Sindhi",
    species = "Bos indicus",
    origin = "Sindh",
    type = AnimalType.DAIRY,
    milkYieldMinLitresPerDay = 8,
    milkYieldMaxLitresPerDay = 14,
    characteristics = listOf("Heat tolerant", "Hardy"),
)

class ReportContentBuilderTest : FunSpec({

    test("builds all three sections when breed info is available") {
        val content = ReportContentBuilder.build(record(), breedInfo(), "2026-07-03 10:00", "2026-07-03 09:58")

        content.sections.map { it.title } shouldBe
            listOf("Classification", "Breed information", "Processing details")
        content.generatedAt shouldBe "2026-07-03 10:00"
    }

    test("formats the breed label as a display name with percent confidence") {
        val content = ReportContentBuilder.build(record(), null, "now", "then")
        val classification = content.sections.first()

        classification.fields.first { it.label == "Breed" }.value shouldBe "Red Sindhi"
        classification.fields.first { it.label == "Confidence" }.value shouldBe "87.0%"
        classification.fields.first { it.label == "Other candidates" }.value shouldContain "Gir (8.0%)"
    }

    test("omits the breed information section when the catalog has no entry") {
        val content = ReportContentBuilder.build(record(), null, "now", "then")
        content.sections.map { it.title } shouldNotContain "Breed information"
    }

    test("omits milk yield when the breed reports none") {
        val info = breedInfo().copy(milkYieldMinLitresPerDay = 0, milkYieldMaxLitresPerDay = 0)
        val content = ReportContentBuilder.build(record(), info, "now", "then")
        val breedSection = content.sections.first { it.title == "Breed information" }

        breedSection.fields.map { it.label } shouldNotContain "Typical milk yield"
    }

    test("omits alternatives when there are none") {
        val content = ReportContentBuilder.build(record(alternatives = emptyList()), null, "now", "then")
        content.sections.first().fields.map { it.label } shouldNotContain "Other candidates"
    }

    test("processing section records the on-device guarantee and model version") {
        val content = ReportContentBuilder.build(record(), breedInfo(), "now", "then")
        val processing = content.sections.last()

        processing.fields.map { it.label } shouldContain "Model"
        processing.fields.first { it.label == "Analysis" }.value shouldContain "on device"
        processing.fields.first { it.label == "Processing time" }.value shouldBe "420 ms"
    }
})
