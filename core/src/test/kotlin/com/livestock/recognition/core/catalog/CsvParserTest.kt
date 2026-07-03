package com.livestock.recognition.core.catalog

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.list
import io.kotest.property.arbitrary.stringPattern
import io.kotest.property.checkAll

class CsvParserTest : FunSpec({

    test("splits plain fields") {
        CsvParser.splitLine("a,b,c") shouldBe listOf("a", "b", "c")
    }

    test("keeps empty fields") {
        CsvParser.splitLine("a,,c,") shouldBe listOf("a", "", "c", "")
    }

    test("quoted fields may contain commas and pipes") {
        CsvParser.splitLine("Gir,\"Heat tolerant, docile|Hardy\",x") shouldBe
            listOf("Gir", "Heat tolerant, docile|Hardy", "x")
    }

    test("doubled quotes escape a literal quote") {
        CsvParser.splitLine("\"say \"\"hi\"\"\",b") shouldBe listOf("say \"hi\"", "b")
    }

    test("unterminated quote is rejected") {
        shouldThrow<IllegalArgumentException> {
            CsvParser.splitLine("a,\"unterminated")
        }
    }

    test("round-trips any quote-free fields") {
        checkAll(Arb.list(Arb.stringPattern("[^\",\\n]{0,15}"), 1..8)) { fields ->
            CsvParser.splitLine(fields.joinToString(",")) shouldBe fields
        }
    }
})
