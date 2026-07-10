package com.livestock.recognition.report

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import com.livestock.recognition.core.report.ReportContent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Renders a [ReportContent] to a single- or multi-page A4 PDF using the
 * platform [PdfDocument] API with no third-party PDF dependency.
 */
class PdfReportGenerator(private val context: Context) {

    suspend fun generate(content: ReportContent, photo: Bitmap?): File =
        withContext(Dispatchers.IO) {
            val document = PdfDocument()
            try {
                val writer = PageWriter(document)
                writer.drawTitle(content.title, content.generatedAt)
                photo?.let { writer.drawPhoto(it) }
                content.sections.forEach { section ->
                    writer.drawSectionTitle(section.title)
                    section.fields.forEach { field ->
                        writer.drawField(field.label, field.value)
                    }
                }
                writer.finishPage()

                val file = newReportFile()
                file.outputStream().use { document.writeTo(it) }
                file
            } finally {
                document.close()
            }
        }

    private fun newReportFile(): File {
        val dir = File(context.filesDir, REPORTS_DIR).apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File(dir, "breed_report_$stamp.pdf")
    }

    /** Cursor-based page layout with automatic page breaks. */
    private class PageWriter(private val document: PdfDocument) {

        private val titlePaint = paint(18f, bold = true)
        private val captionPaint = paint(9f, color = 0xFF666666.toInt())
        private val sectionPaint = paint(13f, bold = true)
        private val labelPaint = paint(10f, bold = true)
        private val valuePaint = paint(10f)

        private var pageNumber = 0
        private var page = newPage()
        private var y = MARGIN

        fun drawTitle(title: String, generatedAt: String) {
            page.canvas.drawText(title, MARGIN, y + titlePaint.textSize, titlePaint)
            y += titlePaint.textSize + 6f
            page.canvas.drawText("Generated $generatedAt", MARGIN, y + captionPaint.textSize, captionPaint)
            y += captionPaint.textSize + 18f
        }

        fun drawPhoto(photo: Bitmap) {
            val maxWidth = PAGE_WIDTH - 2 * MARGIN
            val maxHeight = 260f
            val scale = minOf(maxWidth / photo.width, maxHeight / photo.height, 1f)
            val width = (photo.width * scale).toInt().coerceAtLeast(1)
            val height = (photo.height * scale).toInt().coerceAtLeast(1)
            ensureSpace(height + 18f)

            val scaled = Bitmap.createScaledBitmap(photo, width, height, true)
            try {
                page.canvas.drawBitmap(scaled, MARGIN, y, null)
            } finally {
                if (scaled !== photo) scaled.recycle()
            }
            y += height + 18f
        }

        fun drawSectionTitle(title: String) {
            ensureSpace(sectionPaint.textSize + 20f)
            y += 8f
            page.canvas.drawText(title, MARGIN, y + sectionPaint.textSize, sectionPaint)
            y += sectionPaint.textSize + 8f
        }

        fun drawField(label: String, value: String) {
            val labelWidth = 130f
            val valueX = MARGIN + labelWidth
            val maxValueWidth = PAGE_WIDTH - MARGIN - valueX
            val lines = wrap(value, valuePaint, maxValueWidth)
            ensureSpace(lines.size * LINE_HEIGHT + 2f)

            page.canvas.drawText(label, MARGIN, y + valuePaint.textSize, labelPaint)
            lines.forEach { line ->
                page.canvas.drawText(line, valueX, y + valuePaint.textSize, valuePaint)
                y += LINE_HEIGHT
            }
            y += 2f
        }

        fun finishPage() {
            document.finishPage(page)
        }

        private fun ensureSpace(needed: Float) {
            if (y + needed <= PAGE_HEIGHT - MARGIN) return
            document.finishPage(page)
            page = newPage()
            y = MARGIN
        }

        private fun newPage(): PdfDocument.Page {
            pageNumber++
            val info = PdfDocument.PageInfo
                .Builder(PAGE_WIDTH.toInt(), PAGE_HEIGHT.toInt(), pageNumber)
                .create()
            return document.startPage(info)
        }

        private fun wrap(text: String, paint: Paint, maxWidth: Float): List<String> {
            if (paint.measureText(text) <= maxWidth) return listOf(text)
            val words = text.split(" ")
            val lines = mutableListOf<String>()
            var current = StringBuilder()
            for (word in words) {
                val candidate = if (current.isEmpty()) word else "$current $word"
                if (paint.measureText(candidate) <= maxWidth || current.isEmpty()) {
                    current = StringBuilder(candidate)
                } else {
                    lines.add(current.toString())
                    current = StringBuilder(word)
                }
            }
            if (current.isNotEmpty()) lines.add(current.toString())
            return lines
        }

        private companion object {
            // A4 at 72 dpi.
            const val PAGE_WIDTH = 595f
            const val PAGE_HEIGHT = 842f
            const val MARGIN = 48f
            const val LINE_HEIGHT = 15f

            fun paint(size: Float, bold: Boolean = false, color: Int = Color.BLACK) =
                Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    textSize = size
                    this.color = color
                    typeface = if (bold) Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                    else Typeface.DEFAULT
                }
        }
    }

    private companion object {
        const val REPORTS_DIR = "reports"
    }
}
