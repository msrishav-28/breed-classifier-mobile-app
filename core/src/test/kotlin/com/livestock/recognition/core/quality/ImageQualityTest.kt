package com.livestock.recognition.core.quality

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.collections.shouldContain
import io.kotest.matchers.collections.shouldNotContain
import io.kotest.matchers.doubles.plusOrMinus
import io.kotest.matchers.doubles.shouldBeGreaterThan
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.int
import io.kotest.property.arbitrary.list
import io.kotest.property.checkAll

private fun uniformImage(width: Int, height: Int, value: Int) =
    IntArray(width * height) { value }

private fun checkerboard(width: Int, height: Int, low: Int = 0, high: Int = 255) =
    IntArray(width * height) { i ->
        val x = i % width
        val y = i / width
        if ((x + y) % 2 == 0) high else low
    }

class ImageQualityTest : FunSpec({

    context("LuminanceStatistics") {
        test("uniform image has zero deviation and zero sharpness") {
            val m = LuminanceStatistics.compute(uniformImage(8, 8, 128), 8, 8)
            m.meanLuminance shouldBe (128.0 plusOrMinus 1e-9)
            m.luminanceStdDev shouldBe (0.0 plusOrMinus 1e-9)
            m.sharpness shouldBe (0.0 plusOrMinus 1e-9)
        }

        test("checkerboard is sharp and high contrast") {
            val m = LuminanceStatistics.compute(checkerboard(16, 16), 16, 16)
            m.luminanceStdDev shouldBeGreaterThan 100.0
            m.sharpness shouldBeGreaterThan ImageQualityPolicy.MIN_SHARPNESS
        }

        test("reports original resolution when analysing a downscaled buffer") {
            val m = LuminanceStatistics.compute(
                uniformImage(4, 4, 10), 4, 4,
                originalWidth = 4000, originalHeight = 3000,
            )
            m.width shouldBe 4000
            m.height shouldBe 3000
        }

        test("rejects a buffer that does not match its dimensions") {
            shouldThrow<IllegalArgumentException> {
                LuminanceStatistics.compute(IntArray(10), 4, 4)
            }
        }

        test("mean stays within the luminance range for arbitrary images") {
            checkAll(Arb.list(Arb.int(0..255), 9..64)) { pixels ->
                val width = 3
                val height = pixels.size / width
                val trimmed = pixels.take(width * height).toIntArray()
                val m = LuminanceStatistics.compute(trimmed, width, height)
                (m.meanLuminance in 0.0..255.0) shouldBe true
                (m.sharpness >= 0.0) shouldBe true
                (m.luminanceStdDev >= 0.0) shouldBe true
            }
        }
    }

    context("ImageQualityPolicy") {
        fun goodMetrics() = QualityMetrics(
            width = 1000, height = 1000,
            meanLuminance = 120.0, luminanceStdDev = 50.0, sharpness = 200.0,
        )

        test("good metrics are acceptable") {
            ImageQualityPolicy.assess(goodMetrics()).isAcceptable shouldBe true
        }

        test("small images are flagged") {
            val a = ImageQualityPolicy.assess(goodMetrics().copy(width = 100))
            a.issues shouldContain QualityIssue.LOW_RESOLUTION
        }

        test("dark and bright images are mutually exclusive flags") {
            val dark = ImageQualityPolicy.assess(goodMetrics().copy(meanLuminance = 10.0))
            dark.issues shouldContain QualityIssue.TOO_DARK
            dark.issues shouldNotContain QualityIssue.TOO_BRIGHT

            val bright = ImageQualityPolicy.assess(goodMetrics().copy(meanLuminance = 250.0))
            bright.issues shouldContain QualityIssue.TOO_BRIGHT
            bright.issues shouldNotContain QualityIssue.TOO_DARK
        }

        test("flat images are flagged as low contrast") {
            ImageQualityPolicy.assess(goodMetrics().copy(luminanceStdDev = 2.0))
                .issues shouldContain QualityIssue.LOW_CONTRAST
        }

        test("soft images are flagged as blurry") {
            ImageQualityPolicy.assess(goodMetrics().copy(sharpness = 5.0))
                .issues shouldContain QualityIssue.BLURRY
        }
    }
})
