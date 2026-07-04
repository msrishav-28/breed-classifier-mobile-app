package com.livestock.recognition.data.db

import com.livestock.recognition.core.model.AnimalType
import com.livestock.recognition.core.model.ClassificationRecord
import com.livestock.recognition.core.model.Prediction
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test

class ClassificationEntityTest {

    private fun record(alternatives: List<Prediction>) = ClassificationRecord(
        breedLabel = "Red_Sindhi",
        confidence = 0.87f,
        animalType = AnimalType.DAIRY,
        alternatives = alternatives,
        capturedAtEpochMillis = 1_700_000_000_000,
        processingTimeMillis = 420,
        modelVersion = "test-model",
    )

    @Test
    fun `record round-trips through the entity`() {
        val original = record(listOf(Prediction("Gir", 0.08f), Prediction("Sahiwal", 0.03f)))
        val roundTripped = original.toEntity("/data/images/a.jpg").toRecord()

        assertEquals(original, roundTripped)
    }

    @Test
    fun `empty alternatives round-trip`() {
        val original = record(emptyList())
        assertEquals(original, original.toEntity("/x.jpg").toRecord())
    }

    @Test
    fun `unknown animal type value degrades to null`() {
        val entity = record(emptyList()).toEntity("/x.jpg").copy(animalType = "MARINE")
        assertNull(entity.toRecord().animalType)
    }

    @Test
    fun `corrupt alternatives entries are dropped`() {
        assertEquals(emptyList<Prediction>(), decodeAlternatives("garbage-without-separator"))
        assertEquals(
            listOf(Prediction("Gir", 0.5f)),
            decodeAlternatives("Gir=0.5;=0.2;Broken=x"),
        )
    }

    @Test
    fun `out-of-range stored confidence is clamped`() {
        val entity = record(emptyList()).toEntity("/x.jpg").copy(confidence = 42f)
        assertEquals(1f, entity.toRecord().confidence)
    }
}
