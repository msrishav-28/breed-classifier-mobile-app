package com.livestock.recognition.core.catalog

import com.livestock.recognition.core.model.AnimalType
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.nulls.shouldBeNull
import io.kotest.matchers.nulls.shouldNotBeNull
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.kotest.matchers.types.shouldBeInstanceOf
import java.io.StringReader

private const val HEADER =
    "breed_name,scientific_name,origin,animal_type,milk_yield_min,milk_yield_max,characteristics"

private fun parse(vararg rows: String): BreedCatalog.ParseResult =
    BreedCatalog.parse(StringReader((listOf(HEADER) + rows).joinToString("\n")))

class BreedCatalogTest : FunSpec({

    test("parses a valid catalog") {
        val result = parse(
            "Gir,Bos indicus,Gujarat,dairy,8,12,\"Heat tolerant|Docile\"",
            "Red Sindhi,Bos indicus,Sindh,dairy,8,14,\"Hardy\"",
        ).shouldBeInstanceOf<BreedCatalog.ParseResult.Success>()

        result.warnings shouldBe emptyList()
        result.catalog.size shouldBe 2

        val gir = result.catalog.find("Gir").shouldNotBeNull()
        gir.species shouldBe "Bos indicus"
        gir.type shouldBe AnimalType.DAIRY
        gir.characteristics shouldBe listOf("Heat tolerant", "Docile")
    }

    test("model labels with underscores resolve to catalog entries") {
        val result = parse("Red Sindhi,Bos indicus,Sindh,dairy,8,14,\"Hardy\"")
            .shouldBeInstanceOf<BreedCatalog.ParseResult.Success>()

        result.catalog.find("Red_Sindhi").shouldNotBeNull().name shouldBe "Red Sindhi"
        result.catalog.find("red_sindhi").shouldNotBeNull()
        result.catalog.find("Sahiwal").shouldBeNull()
    }

    test("empty input fails") {
        BreedCatalog.parse(StringReader(""))
            .shouldBeInstanceOf<BreedCatalog.ParseResult.Failure>()
    }

    test("wrong header fails") {
        BreedCatalog.parse(StringReader("name,type\nGir,dairy"))
            .shouldBeInstanceOf<BreedCatalog.ParseResult.Failure>()
            .reason shouldContain "header"
    }

    test("header with no valid rows fails") {
        parse().shouldBeInstanceOf<BreedCatalog.ParseResult.Failure>()
    }

    test("malformed rows are skipped with warnings, valid rows survive") {
        val result = parse(
            "Gir,Bos indicus,Gujarat,dairy,8,12,\"Docile\"",
            "Broken,only,three",
            "BadType,Bos indicus,Punjab,marine,1,2,\"x\"",
            "BadYield,Bos indicus,Punjab,dairy,many,2,\"x\"",
            "Inverted,Bos indicus,Punjab,dairy,9,2,\"x\"",
        ).shouldBeInstanceOf<BreedCatalog.ParseResult.Success>()

        result.catalog.size shouldBe 1
        result.warnings shouldHaveSize 4
    }

    test("duplicate breeds keep the first entry and warn") {
        val result = parse(
            "Gir,Bos indicus,Gujarat,dairy,8,12,\"First\"",
            "gir,Bos indicus,Gujarat,dairy,1,2,\"Second\"",
        ).shouldBeInstanceOf<BreedCatalog.ParseResult.Success>()

        result.catalog.size shouldBe 1
        result.catalog.find("Gir").shouldNotBeNull().characteristics shouldBe listOf("First")
        result.warnings shouldHaveSize 1
    }

    test("byType filters correctly") {
        val result = parse(
            "Gir,Bos indicus,Gujarat,dairy,8,12,\"a\"",
            "Khillari,Bos indicus,Maharashtra,draught,2,4,\"b\"",
            "Rathi,Bos indicus,Rajasthan,dual_purpose,5,8,\"c\"",
        ).shouldBeInstanceOf<BreedCatalog.ParseResult.Success>()

        result.catalog.byType(AnimalType.DAIRY).map { it.name } shouldBe listOf("Gir")
        result.catalog.byType(AnimalType.DRAUGHT).map { it.name } shouldBe listOf("Khillari")
        result.catalog.byType(AnimalType.DUAL_PURPOSE).map { it.name } shouldBe listOf("Rathi")
    }
})
