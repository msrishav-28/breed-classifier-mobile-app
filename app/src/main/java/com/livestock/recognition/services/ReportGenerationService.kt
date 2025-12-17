package com.livestock.recognition.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.itextpdf.io.image.ImageDataFactory
import com.itextpdf.kernel.pdf.PdfDocument
import com.itextpdf.kernel.pdf.PdfWriter
import com.itextpdf.layout.Document
import com.itextpdf.layout.element.Image
import com.itextpdf.layout.element.Paragraph
import com.itextpdf.layout.element.Table
import com.itextpdf.layout.property.TextAlignment
import com.itextpdf.layout.property.UnitValue
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.data.models.BreedInfo
import com.livestock.recognition.data.models.Report
import com.livestock.recognition.data.models.ValidationResult
import java.io.ByteArrayOutputStream
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Service for generating comprehensive PDF reports from classification results.
 * Includes embedded images, metadata, breed information, and verification details.
 */
class ReportGenerationService(
    private val context: Context,
    private val typeClassificationService: TypeClassificationService
) {

    private val dateFormatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    private val fileNameFormatter = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())

    /**
     * Generates a comprehensive PDF report from classification result.
     * @param result The classification result to generate report for
     * @param imageBitmap The original image bitmap to embed in the report
     * @param outputDir The directory to save the PDF file
     * @return Report object containing file path and metadata or null if generation failed
     */
    fun generateReport(
        result: ClassificationResult,
        imageBitmap: Bitmap?,
        outputDir: File
    ): Report? {
        return try {
            // Validate input
            val validation = result.validate()
            if (!validation.isValid) {
                return null
            }

            // Ensure output directory exists
            if (!outputDir.exists()) {
                outputDir.mkdirs()
            }

            // Generate unique filename
            val timestamp = fileNameFormatter.format(Date(result.timestamp))
            val fileName = "livestock_report_${result.breed.replace(" ", "_")}_$timestamp.pdf"
            val outputFile = File(outputDir, fileName)

            // Create PDF document
            val pdfWriter = PdfWriter(outputFile)
            val pdfDocument = PdfDocument(pdfWriter)
            val document = Document(pdfDocument)

            // Add report content
            addReportHeader(document, result)
            addClassificationSummary(document, result)
            
            if (imageBitmap != null) {
                addImageSection(document, imageBitmap)
            }
            
            addBreedInformation(document, result)
            addTechnicalDetails(document, result)
            addVerificationSection(document, result)
            addReportFooter(document)

            // Close document
            document.close()

            Report(
                filePath = outputFile.absolutePath,
                fileName = fileName,
                fileSize = outputFile.length(),
                generationTime = System.currentTimeMillis(),
                classificationResult = result
            )
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Validates that a report can be generated from the given classification result.
     * @param result The classification result to validate
     * @return ValidationResult indicating if report generation is possible
     */
    fun validateReportGeneration(result: ClassificationResult): ValidationResult {
        val errors = mutableListOf<String>()

        // Validate classification result
        val resultValidation = result.validate()
        if (!resultValidation.isValid) {
            errors.addAll(resultValidation.errors)
        }

        // Check if breed information is available
        val breedInfo = typeClassificationService.getBreedCharacteristics(result.breed)
        if (breedInfo == null) {
            errors.add("Breed information not available for: ${result.breed}")
        }

        return ValidationResult(errors.isEmpty(), errors)
    }

    /**
     * Gets the default output directory for reports.
     * @return File object representing the reports directory
     */
    fun getDefaultOutputDirectory(): File {
        val reportsDir = File(context.getExternalFilesDir(null), "reports")
        if (!reportsDir.exists()) {
            reportsDir.mkdirs()
        }
        return reportsDir
    }

    private fun addReportHeader(document: Document, result: ClassificationResult) {
        // Title
        val title = Paragraph("Livestock Breed Recognition Report")
            .setFontSize(20f)
            .setBold()
            .setTextAlignment(TextAlignment.CENTER)
        document.add(title)

        // Subtitle with timestamp
        val subtitle = Paragraph("Generated on ${dateFormatter.format(Date(result.timestamp))}")
            .setFontSize(12f)
            .setTextAlignment(TextAlignment.CENTER)
            .setMarginBottom(20f)
        document.add(subtitle)
    }

    private fun addClassificationSummary(document: Document, result: ClassificationResult) {
        val summaryTitle = Paragraph("Classification Summary")
            .setFontSize(16f)
            .setBold()
            .setMarginBottom(10f)
        document.add(summaryTitle)

        val table = Table(UnitValue.createPercentArray(floatArrayOf(30f, 70f)))
            .setWidth(UnitValue.createPercentValue(100f))

        table.addCell("Identified Breed:")
        table.addCell(result.breed)

        table.addCell("Breed Confidence:")
        table.addCell("${(result.breedConfidence * 100).toInt()}%")

        table.addCell("Animal Type:")
        table.addCell(result.animalType.displayName)

        table.addCell("Type Confidence:")
        table.addCell("${(result.typeConfidence * 100).toInt()}%")

        table.addCell("Processing Time:")
        table.addCell("${result.processingTime}ms")

        document.add(table)
        document.add(Paragraph("\n"))
    }

    private fun addImageSection(document: Document, imageBitmap: Bitmap) {
        val imageTitle = Paragraph("Analyzed Image")
            .setFontSize(16f)
            .setBold()
            .setMarginBottom(10f)
        document.add(imageTitle)

        try {
            // Convert bitmap to byte array
            val stream = ByteArrayOutputStream()
            imageBitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
            val imageData = ImageDataFactory.create(stream.toByteArray())
            
            // Create image element with appropriate sizing
            val image = Image(imageData)
            image.setWidth(UnitValue.createPercentValue(60f))
            image.setTextAlignment(TextAlignment.CENTER)
            
            document.add(image)
            document.add(Paragraph("\n"))
        } catch (e: Exception) {
            val errorParagraph = Paragraph("Image could not be embedded in the report.")
                .setItalic()
            document.add(errorParagraph)
            document.add(Paragraph("\n"))
        }
    }

    private fun addBreedInformation(document: Document, result: ClassificationResult) {
        val breedTitle = Paragraph("Breed Information")
            .setFontSize(16f)
            .setBold()
            .setMarginBottom(10f)
        document.add(breedTitle)

        val breedInfo = typeClassificationService.getBreedCharacteristics(result.breed)
        if (breedInfo != null) {
            val infoTable = Table(UnitValue.createPercentArray(floatArrayOf(30f, 70f)))
                .setWidth(UnitValue.createPercentValue(100f))

            infoTable.addCell("Scientific Name:")
            infoTable.addCell(breedInfo.scientificName)

            infoTable.addCell("Origin:")
            infoTable.addCell(breedInfo.origin)

            infoTable.addCell("Primary Use:")
            infoTable.addCell(breedInfo.primaryUse.description)

            infoTable.addCell("Milk Yield Range:")
            infoTable.addCell("${breedInfo.averageMilkYieldMin} - ${breedInfo.averageMilkYieldMax} liters/day")

            document.add(infoTable)

            // Add characteristics
            if (breedInfo.characteristics.isNotEmpty()) {
                val characteristicsTitle = Paragraph("Key Characteristics:")
                    .setFontSize(14f)
                    .setBold()
                    .setMarginTop(10f)
                document.add(characteristicsTitle)

                breedInfo.characteristics.forEach { characteristic ->
                    val bullet = Paragraph("• $characteristic")
                        .setMarginLeft(20f)
                    document.add(bullet)
                }
            }
        } else {
            val noInfoParagraph = Paragraph("Detailed breed information not available.")
                .setItalic()
            document.add(noInfoParagraph)
        }

        document.add(Paragraph("\n"))
    }

    private fun addTechnicalDetails(document: Document, result: ClassificationResult) {
        val technicalTitle = Paragraph("Technical Details")
            .setFontSize(16f)
            .setBold()
            .setMarginBottom(10f)
        document.add(technicalTitle)

        val techTable = Table(UnitValue.createPercentArray(floatArrayOf(30f, 70f)))
            .setWidth(UnitValue.createPercentValue(100f))

        techTable.addCell("Image Resolution:")
        techTable.addCell("${result.imageMetadata.width} x ${result.imageMetadata.height} pixels")

        techTable.addCell("File Size:")
        techTable.addCell("${result.imageMetadata.fileSize / 1024} KB")

        techTable.addCell("Quality Score:")
        techTable.addCell("${(result.imageMetadata.qualityScore * 100).toInt()}%")

        techTable.addCell("Device Info:")
        techTable.addCell(result.imageMetadata.deviceInfo)

        techTable.addCell("Capture Time:")
        techTable.addCell(dateFormatter.format(Date(result.imageMetadata.captureTime)))

        document.add(techTable)
        document.add(Paragraph("\n"))
    }

    private fun addVerificationSection(document: Document, result: ClassificationResult) {
        val verificationTitle = Paragraph("Verification Details")
            .setFontSize(16f)
            .setBold()
            .setMarginBottom(10f)
        document.add(verificationTitle)

        // Confidence assessment
        val confidenceAssessment = when {
            result.breedConfidence >= 0.9f -> "High confidence - Reliable identification"
            result.breedConfidence >= 0.75f -> "Medium confidence - Good identification"
            else -> "Low confidence - Manual verification recommended"
        }

        val assessmentParagraph = Paragraph("Confidence Assessment: $confidenceAssessment")
            .setBold()
        document.add(assessmentParagraph)

        // Recommendations
        val recommendationsTitle = Paragraph("Recommendations:")
            .setFontSize(14f)
            .setBold()
            .setMarginTop(10f)
        document.add(recommendationsTitle)

        val recommendations = mutableListOf<String>()
        
        if (result.breedConfidence < 0.75f) {
            recommendations.add("Consider retaking the image with better lighting and focus")
            recommendations.add("Consult with a veterinary expert for manual verification")
        }
        
        if (result.imageMetadata.qualityScore < 0.7f) {
            recommendations.add("Image quality could be improved for better accuracy")
        }
        
        recommendations.add("Use this report for documentation and record-keeping purposes")
        recommendations.add("Store the original image alongside this report for future reference")

        recommendations.forEach { recommendation ->
            val bullet = Paragraph("• $recommendation")
                .setMarginLeft(20f)
            document.add(bullet)
        }

        document.add(Paragraph("\n"))
    }

    private fun addReportFooter(document: Document) {
        val footer = Paragraph("This report was generated by the Comprehensive Cattle and Buffalo Recognition System.")
            .setFontSize(10f)
            .setItalic()
            .setTextAlignment(TextAlignment.CENTER)
            .setMarginTop(20f)
        document.add(footer)

        val disclaimer = Paragraph("Note: This automated identification should be used as a guide. For critical decisions, please consult with qualified veterinary professionals.")
            .setFontSize(9f)
            .setItalic()
            .setTextAlignment(TextAlignment.CENTER)
        document.add(disclaimer)
    }
}

