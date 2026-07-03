package com.livestock.recognition.image

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class BitmapLoaderTest {

    @Test
    fun `small images are not subsampled`() {
        assertEquals(1, BitmapLoader.sampleSize(width = 800, height = 600, maxDimension = 1280))
    }

    @Test
    fun `large images are subsampled by powers of two`() {
        assertEquals(2, BitmapLoader.sampleSize(width = 4000, height = 3000, maxDimension = 1280))
        assertEquals(4, BitmapLoader.sampleSize(width = 8000, height = 6000, maxDimension = 1280))
    }

    @Test
    fun `the longest edge drives the sample size`() {
        assertEquals(
            BitmapLoader.sampleSize(width = 6000, height = 100, maxDimension = 1280),
            BitmapLoader.sampleSize(width = 100, height = 6000, maxDimension = 1280),
        )
    }

    @Test
    fun `result keeps the longest edge at or above the target`() {
        val sample = BitmapLoader.sampleSize(width = 4000, height = 3000, maxDimension = 1280)
        val resultingLongestEdge = 4000 / sample
        assert(resultingLongestEdge >= 1280) {
            "Subsampling must never undershoot the requested dimension"
        }
    }
}
