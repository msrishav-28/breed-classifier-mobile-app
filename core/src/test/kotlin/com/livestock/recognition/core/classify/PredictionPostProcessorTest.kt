package com.livestock.recognition.core.classify

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.doubles.plusOrMinus
import io.kotest.matchers.floats.shouldBeGreaterThanOrEqual
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.float
import io.kotest.property.arbitrary.int
import io.kotest.property.arbitrary.list
import io.kotest.property.arbitrary.numericFloat
import io.kotest.property.checkAll

class PredictionPostProcessorTest : FunSpec({

    val labels = listOf("Gir", "Sahiwal", "Murrah", "Surti")

    context("topPredictions") {
        test("ranks by confidence and returns k entries") {
            val scores = floatArrayOf(0.1f, 0.6f, 0.25f, 0.05f)
            val top = PredictionPostProcessor.topPredictions(scores, labels, 3)

            top.map { it.label } shouldBe listOf("Sahiwal", "Murrah", "Gir")
            top[0].confidence shouldBe 0.6f
        }

        test("k larger than label count is capped") {
            val scores = floatArrayOf(0.5f, 0.5f, 0f, 0f)
            PredictionPostProcessor.topPredictions(scores, labels, 10) shouldHaveSize 4
        }

        test("mismatched score and label counts are rejected") {
            shouldThrow<IllegalArgumentException> {
                PredictionPostProcessor.topPredictions(floatArrayOf(1f), labels, 1)
            }
        }

        test("non-positive k is rejected") {
            shouldThrow<IllegalArgumentException> {
                PredictionPostProcessor.topPredictions(floatArrayOf(1f, 0f, 0f, 0f), labels, 0)
            }
        }

        test("results are always sorted and within [0,1] for arbitrary scores") {
            checkAll(Arb.list(Arb.numericFloat(-100f, 100f), 4..4)) { raw ->
                val top = PredictionPostProcessor.topPredictions(raw.toFloatArray(), labels, 4)
                top shouldHaveSize 4
                top.zipWithNext().forEach { (a, b) ->
                    a.confidence shouldBeGreaterThanOrEqual b.confidence
                }
                top.forEach { it.confidence shouldBe it.confidence.coerceIn(0f, 1f) }
            }
        }
    }

    context("normalize") {
        test("probabilities that already sum to one pass through") {
            val out = PredictionPostProcessor.normalize(floatArrayOf(0.2f, 0.3f, 0.5f))
            out.sum().toDouble() shouldBe (1.0 plusOrMinus 1e-4)
            out[2] shouldBe 0.5f
        }

        test("negative and NaN scores are clamped to zero") {
            val out = PredictionPostProcessor.normalize(floatArrayOf(-1f, Float.NaN, 2f))
            out[0] shouldBe 0f
            out[1] shouldBe 0f
            out[2] shouldBe 1f
        }

        test("all-zero input becomes a uniform distribution") {
            val out = PredictionPostProcessor.normalize(FloatArray(4))
            out.forEach { it shouldBe 0.25f }
        }

        test("output always sums to approximately one") {
            checkAll(Arb.list(Arb.numericFloat(-10f, 10f), 1..16)) { raw ->
                val sum = PredictionPostProcessor.normalize(raw.toFloatArray()).sum().toDouble()
                sum shouldBe (1.0 plusOrMinus 1e-3)
            }
        }
    }

    context("softmax") {
        test("is numerically stable for large logits") {
            val out = PredictionPostProcessor.softmax(floatArrayOf(1000f, 1000f))
            out[0].toDouble() shouldBe (0.5 plusOrMinus 1e-6)
        }

        test("orders monotonically with logits and sums to one") {
            checkAll(Arb.list(Arb.numericFloat(-50f, 50f), 2..12)) { raw ->
                val logits = raw.toFloatArray()
                val out = PredictionPostProcessor.softmax(logits)
                out.sum().toDouble() shouldBe (1.0 plusOrMinus 1e-3)
                val maxLogit = logits.indices.maxBy { logits[it] }
                out[maxLogit] shouldBe out.max()
            }
        }

        test("empty input yields empty output") {
            PredictionPostProcessor.softmax(FloatArray(0)) shouldHaveSize 0
        }
    }

    context("confidence policy") {
        test("boundary values map to the documented levels") {
            ConfidencePolicy.levelFor(0.75f) shouldBe ConfidencePolicy.Level.HIGH
            ConfidencePolicy.levelFor(0.7499f) shouldBe ConfidencePolicy.Level.MEDIUM
            ConfidencePolicy.levelFor(0.5f) shouldBe ConfidencePolicy.Level.MEDIUM
            ConfidencePolicy.levelFor(0.4999f) shouldBe ConfidencePolicy.Level.LOW
            ConfidencePolicy.levelFor(0f) shouldBe ConfidencePolicy.Level.LOW
        }

        test("warning is required exactly below the high threshold") {
            checkAll(Arb.float(0f, 1f)) { c ->
                ConfidencePolicy.requiresWarning(c) shouldBe
                    (c < ConfidencePolicy.HIGH_THRESHOLD)
            }
        }

        test("levels are monotonic in confidence") {
            checkAll(Arb.int(0..100), Arb.int(0..100)) { a, b ->
                val lower = minOf(a, b) / 100f
                val higher = maxOf(a, b) / 100f
                ConfidencePolicy.levelFor(lower).ordinal shouldBeGreaterThanOrEqualInt
                    ConfidencePolicy.levelFor(higher).ordinal
            }
        }
    }
})

private infix fun Int.shouldBeGreaterThanOrEqualInt(other: Int) {
    if (this < other) throw AssertionError("$this should be >= $other")
}
