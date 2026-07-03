package com.livestock.recognition.core.catalog

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.string
import io.kotest.property.arbitrary.stringPattern
import io.kotest.property.checkAll

class BreedNamesTest : FunSpec({

    context("normalize") {
        test("model labels and display names collapse to the same key") {
            BreedNames.normalize("Red_Sindhi") shouldBe BreedNames.normalize("Red Sindhi")
            BreedNames.normalize("Nili_Ravi") shouldBe BreedNames.normalize("Nili-Ravi")
            BreedNames.normalize("  Gir  ") shouldBe "gir"
            BreedNames.normalize("Krishna   Valley") shouldBe "krishna valley"
        }

        test("is idempotent for any input") {
            checkAll(Arb.string(0..40)) { s ->
                val once = BreedNames.normalize(s)
                BreedNames.normalize(once) shouldBe once
            }
        }

        test("never contains separators other than single spaces") {
            checkAll(Arb.stringPattern("[A-Za-z_\\- ]{1,30}")) { s ->
                val normalized = BreedNames.normalize(s)
                normalized.contains('_') shouldBe false
                normalized.contains('-') shouldBe false
                normalized.contains("  ") shouldBe false
            }
        }
    }

    context("displayName") {
        test("turns model labels into readable names") {
            BreedNames.displayName("Red_Sindhi") shouldBe "Red Sindhi"
            BreedNames.displayName("murrah") shouldBe "Murrah"
            BreedNames.displayName("nili_ravi") shouldBe "Nili Ravi"
        }

        test("display name normalises back to the same key as the label") {
            checkAll(Arb.stringPattern("[A-Za-z]{1,10}(_[A-Za-z]{1,10}){0,3}")) { label ->
                BreedNames.normalize(BreedNames.displayName(label)) shouldBe
                    BreedNames.normalize(label)
            }
        }
    }
})
