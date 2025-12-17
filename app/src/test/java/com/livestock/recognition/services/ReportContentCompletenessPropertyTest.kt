package com.livestock.recognition.services

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import com.livestock.recognition.data.models.*
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.runner.RunWith
import org.mockito.kotlin.*
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Property-based test for report content completeness
 * **Feature: cattle-buffalo-recognition, Property 10: Report content completeness**
 * **Validates: Requirements 5.2**
 * 
 * Tests that generated reports include breed name, type classification, confidence scores, and timestamp
 * across all valid classification results
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ReportContentCompletenessPropertyTest : StringSpec({
    
    "Property 10: Report content completeness - generated reports should include all required information" {
        
        // Arrange - Create test dependencies
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockTypeClassificationService = mock<TypeClassificationService>()
        
        // Configure mock to return breed information
        val mockBreedInfo = BreedInfo(
            name = "Test Breed",
            scientificName = "Bos taurus testicus",
            origin = "Test Region",
            primaryUse = AnimalType.DAIRY,
            averageMilkYieldMin = 10,
            averageMilkYieldMax = 20,
            characteristics = listOf("High milk yield", "Disease resistant", "Heat tolerant"),
            imageUrl = null
        )
        
        whenever(mockTypeClassificationService.getBreedCharacteristics(any())).thenReturn(mockBreedInfo)
        
        val reportGenerationService = ReportGenerationService(context, mockTypeClassificationService)
        
        // Property test generators
        val breedGen = Arb.element(listOf(
            "Gir", "Sahiwal", "Red_Sindhi", "Tharparkar", "Rathi",
            "Hariana", "Ongole", "Krishna_Valley", "Deoni", "Khillari",
            "Murrah", "Mehsana", "Surti", "Jaffarabadi", "Nili_Ravi"
        ))
        
        val confidenceGen = Arb.float(0.3f, 1.0f)
        val timestampGen = Arb.long(1000000000L, 2000000000L)
        val processingTimeGen = Arb.long(100L, 5000L)
        
        val imageMetadataGen = Arb.bind(
            Arb.int(640, 4096), // width
            Arb.int(480, 3072), // height
            Arb.long(100000L, 10000000L), // file size
            timestampGen, // capture time
            Arb.element(listOf("Samsung Galaxy", "OnePlus", "Xiaomi", "Google Pixel")), // device info
            Arb.float(0.5f, 1.0f) // quality score
        ) { width, height, fileSize, captureTime, deviceInfo, qualityScore ->
            ImageMetadata(
                width = width,
                height = height,
                fileSize = fileSize,
                captureTime = captureTime,
                deviceInfo = deviceInfo,
                qualityScore = qualityScore
            )
        }
        
        val classificationResultGen = Arb.bind(
            breedGen,
            confidenceGen,
            Arb.enum<AnimalType>(),
            confidenceGen,
            timestampGen,
            imageMetadataGen,
            processingTimeGen
        ) { breed, breedConfidence, animalType, typeConfidence, timestamp, imageMetadata, processingTime ->
            ClassificationResult(
                breed = breed,
                breedConfidence = breedConfidence,
                animalType = animalType,
                typeConfidence = typeConfidence,
                timestamp = timestamp,
                imageMetadata = imageMetadata,
                processingTime = processingTime
            )
        }
        
        // Create test bitmap
        val testBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.RGB_565)
        val outputDir = File(context.cacheDir, "test_reports")
        
        checkAll(100, classificationResultGen) { classificationResult ->
            
            // Update mock to return breed info for the specific breed
            val breedSpecificInfo = mockBreedInfo.copy(name = classificationResult.breed)
            whenever(mockTypeClassificationService.getBreedCharacteristics(classificationResult.breed))
                .thenReturn(breedSpecificInfo)
            
            // Act - Generate report
            val report = reportGenerationService.generateReport(
                classificationResult,
                testBitmap,
                outputDir
            )
            
            // Assert - Verify report was generated
            report shouldNotBe null
            report!!.validate().isValid shouldBe true
            
            // Verify report file exists
            val reportFile = File(report.filePath)
            reportFile.exists() shouldBe true
            reportFile.length() shouldNotBe 0L
            
            // Verify report contains required information
            report.classificationResult shouldBe classificationResult
            report.fileName.shouldContain(classificationResult.breed.replace(" ", "_"))
            report.fileSize shouldNotBe 0L
            report.generationTime shouldNotBe 0L
            
            // Verify file extension is PDF
            report.getFileExtension() shouldBe "pdf"
            
            // Verify timestamp formatting
            val dateFormatter = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            val expectedTimestamp = dateFormatter.format(Date(classificationResult.timestamp))
            report.fileName.shouldContain(expectedTimestamp)
            
            // Verify breed name is included in filename
            val sanitizedBreedName = classificationResult.breed.replace(" ", "_")
            report.fileName.shouldContain(sanitizedBreedName)
            
            // Verify report validation passes
            val validation = report.validate()
            validation.isValid shouldBe true
            validation.errors.size shouldBe 0
            
            // Clean up test file
            reportFile.delete()
        }
    }
    
    "Property 10a: Report metadata consistency - report metadata should accurately reflect classification data" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockTypeClassificationService = mock<TypeClassificationService>()
        
        val mockBreedInfo = BreedInfo(
            name = "Test Breed",
            scientificName = "Bos taurus testicus",
            origin = "Test Region",
            primaryUse = AnimalType.DAIRY,
            averageMilkYieldMin = 15,
            averageMilkYieldMax = 25,
            characteristics = listOf("High yield", "Hardy", "Adaptable"),
            imageUrl = null
        )
        
        whenever(mockTypeClassificationService.getBreedCharacteristics(any())).thenReturn(mockBreedInfo)
        
        val reportGenerationService = ReportGenerationService(context, mockTypeClassificationService)
        
        // Property test for metadata consistency
        val classificationResultGen = Arb.bind(
            Arb.element(listOf("Gir", "Sahiwal", "Murrah", "Mehsana")),
            Arb.float(0.5f, 1.0f),
            Arb.enum<AnimalType>(),
            Arb.float(0.5f, 1.0f),
            Arb.long(1500000000L, 1700000000L),
            Arb.long(500L, 3000L)
        ) { breed, breedConf, animalType, typeConf, timestamp, processingTime ->
            ClassificationResult(
                breed = breed,
                breedConfidence = breedConf,
                animalType = animalType,
                typeConfidence = typeConf,
                timestamp = timestamp,
                imageMetadata = ImageMetadata(
                    width = 1920,
                    height = 1080,
                    fileSize = 2048000L,
                    captureTime = timestamp - 1000L,
                    deviceInfo = "Test Device",
                    qualityScore = 0.8f
                ),
                processingTime = processingTime
            )
        }
        
        val outputDir = File(context.cacheDir, "test_reports_metadata")
        val testBitmap = Bitmap.createBitmap(50, 50, Bitmap.Config.RGB_565)
        
        checkAll(50, classificationResultGen) { classificationResult ->
            
            // Update mock for specific breed
            val breedInfo = mockBreedInfo.copy(name = classificationResult.breed)
            whenever(mockTypeClassificationService.getBreedCharacteristics(classificationResult.breed))
                .thenReturn(breedInfo)
            
            val startTime = System.currentTimeMillis()
            
            // Act - Generate report
            val report = reportGenerationService.generateReport(
                classificationResult,
                testBitmap,
                outputDir
            )
            
            val endTime = System.currentTimeMillis()
            
            // Assert - Verify metadata consistency
            report shouldNotBe null
            
            // Verify classification result is preserved exactly
            report!!.classificationResult.breed shouldBe classificationResult.breed
            report.classificationResult.breedConfidence shouldBe classificationResult.breedConfidence
            report.classificationResult.animalType shouldBe classificationResult.animalType
            report.classificationResult.typeConfidence shouldBe classificationResult.typeConfidence
            report.classificationResult.timestamp shouldBe classificationResult.timestamp
            report.classificationResult.processingTime shouldBe classificationResult.processingTime
            
            // Verify generation time is reasonable
            report.generationTime shouldNotBe 0L
            (report.generationTime >= startTime) shouldBe true
            (report.generationTime <= endTime) shouldBe true
            
            // Verify file size is positive
            report.fileSize shouldNotBe 0L
            
            // Verify file path and name consistency
            report.filePath.shouldContain(report.fileName)
            report.fileName.shouldContain("livestock_report")
            report.fileName.shouldContain(classificationResult.breed.replace(" ", "_"))
            
            // Clean up
            File(report.filePath).delete()
        }
    }
    
    "Property 10b: Report validation completeness - all report components should pass validation" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockTypeClassificationService = mock<TypeClassificationService>()
        
        // Configure comprehensive breed information
        val comprehensiveBreedInfo = BreedInfo(
            name = "Comprehensive Test Breed",
            scientificName = "Bos taurus comprehensivus",
            origin = "Test Valley, Test State",
            primaryUse = AnimalType.DUAL_PURPOSE,
            averageMilkYieldMin = 8,
            averageMilkYieldMax = 18,
            characteristics = listOf(
                "Excellent milk production",
                "Strong draught capability", 
                "Disease resistance",
                "Heat tolerance",
                "Good feed conversion"
            ),
            imageUrl = "https://example.com/breed.jpg"
        )
        
        whenever(mockTypeClassificationService.getBreedCharacteristics(any())).thenReturn(comprehensiveBreedInfo)
        
        val reportGenerationService = ReportGenerationService(context, mockTypeClassificationService)
        
        // Property test for comprehensive validation
        val validClassificationResultGen = Arb.bind(
            Arb.element(listOf("Holstein", "Jersey", "Angus", "Brahman", "Simmental")),
            Arb.float(0.75f, 1.0f), // High confidence only
            Arb.enum<AnimalType>(),
            Arb.float(0.75f, 1.0f), // High confidence only
            Arb.long(1600000000L, 1800000000L), // Valid timestamp range
            Arb.long(800L, 2500L) // Reasonable processing time
        ) { breed, breedConf, animalType, typeConf, timestamp, processingTime ->
            ClassificationResult(
                breed = breed,
                breedConfidence = breedConf,
                animalType = animalType,
                typeConfidence = typeConf,
                timestamp = timestamp,
                imageMetadata = ImageMetadata(
                    width = 2048,
                    height = 1536,
                    fileSize = 3145728L, // 3MB
                    captureTime = timestamp - 500L,
                    deviceInfo = "Premium Test Device Model X",
                    qualityScore = 0.9f
                ),
                processingTime = processingTime
            )
        }
        
        val outputDir = File(context.cacheDir, "test_reports_validation")
        val highQualityBitmap = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888)
        
        checkAll(30, validClassificationResultGen) { classificationResult ->
            
            // Update mock for specific breed with comprehensive info
            val breedSpecificInfo = comprehensiveBreedInfo.copy(name = classificationResult.breed)
            whenever(mockTypeClassificationService.getBreedCharacteristics(classificationResult.breed))
                .thenReturn(breedSpecificInfo)
            
            // Act - Generate report with high-quality inputs
            val report = reportGenerationService.generateReport(
                classificationResult,
                highQualityBitmap,
                outputDir
            )
            
            // Assert - Comprehensive validation
            report shouldNotBe null
            
            // Validate report structure
            val reportValidation = report!!.validate()
            reportValidation.isValid shouldBe true
            reportValidation.errors.isEmpty() shouldBe true
            
            // Validate classification result within report
            val classificationValidation = report.classificationResult.validate()
            classificationValidation.isValid shouldBe true
            classificationValidation.errors.isEmpty() shouldBe true
            
            // Validate image metadata within classification result
            val imageMetadataValidation = report.classificationResult.imageMetadata!!.validate()
            imageMetadataValidation.isValid shouldBe true
            imageMetadataValidation.errors.isEmpty() shouldBe true
            
            // Validate breed information is available
            val breedInfo = mockTypeClassificationService.getBreedCharacteristics(classificationResult.breed)
            breedInfo shouldNotBe null
            val breedValidation = breedInfo!!.validate()
            breedValidation.isValid shouldBe true
            breedValidation.errors.isEmpty() shouldBe true
            
            // Validate file system properties
            val reportFile = File(report.filePath)
            reportFile.exists() shouldBe true
            reportFile.canRead() shouldBe true
            reportFile.length() shouldBe report.fileSize
            
            // Validate report generation service validation
            val serviceValidation = reportGenerationService.validateReportGeneration(classificationResult)
            serviceValidation.isValid shouldBe true
            serviceValidation.errors.isEmpty() shouldBe true
            
            // Clean up
            reportFile.delete()
        }
    }
    
    "Property 10c: Report content structure - reports should maintain consistent structure across all inputs" {
        
        // Arrange
        val context = ApplicationProvider.getApplicationContext<Context>()
        val mockTypeClassificationService = mock<TypeClassificationService>()
        
        val reportGenerationService = ReportGenerationService(context, mockTypeClassificationService)
        
        // Property test for content structure consistency
        val diverseClassificationResultGen = Arb.bind(
            Arb.element(listOf("Rare_Breed_1", "Common_Breed_2", "Exotic_Breed_3")),
            Arb.float(0.1f, 1.0f), // Full confidence range
            Arb.enum<AnimalType>(),
            Arb.float(0.1f, 1.0f), // Full confidence range
            Arb.long(946684800000L, 2147483647000L), // Wide timestamp range
            Arb.long(50L, 10000L) // Wide processing time range
        ) { breed, breedConf, animalType, typeConf, timestamp, processingTime ->
            ClassificationResult(
                breed = breed,
                breedConfidence = breedConf,
                animalType = animalType,
                typeConfidence = typeConf,
                timestamp = timestamp,
                imageMetadata = ImageMetadata(
                    width = Arb.int(320, 8192).next(),
                    height = Arb.int(240, 6144).next(),
                    fileSize = Arb.long(50000L, 50000000L).next(),
                    captureTime = timestamp - Arb.long(1L, 10000L).next(),
                    deviceInfo = Arb.element(listOf(
                        "Budget Device", "Mid-range Device", "Premium Device", "Unknown Device"
                    )).next(),
                    qualityScore = Arb.float(0.1f, 1.0f).next()
                ),
                processingTime = processingTime
            )
        }
        
        val outputDir = File(context.cacheDir, "test_reports_structure")
        val structureBitmap = Bitmap.createBitmap(150, 150, Bitmap.Config.RGB_565)
        
        checkAll(25, diverseClassificationResultGen) { classificationResult ->
            
            // Configure mock with varying breed information availability
            val hasBreedInfo = Math.random() > 0.3 // 70% chance of having breed info
            
            if (hasBreedInfo) {
                val randomBreedInfo = BreedInfo(
                    name = classificationResult.breed,
                    scientificName = "Bos taurus ${classificationResult.breed.lowercase()}",
                    origin = Arb.element(listOf("India", "Africa", "Europe", "Americas")).next(),
                    primaryUse = classificationResult.animalType,
                    averageMilkYieldMin = Arb.int(5, 15).next(),
                    averageMilkYieldMax = Arb.int(16, 30).next(),
                    characteristics = Arb.list(
                        Arb.element(listOf("Hardy", "High yield", "Disease resistant", "Heat tolerant")),
                        1..5
                    ).next(),
                    imageUrl = if (Math.random() > 0.5) "https://example.com/image.jpg" else null
                )
                whenever(mockTypeClassificationService.getBreedCharacteristics(classificationResult.breed))
                    .thenReturn(randomBreedInfo)
            } else {
                whenever(mockTypeClassificationService.getBreedCharacteristics(classificationResult.breed))
                    .thenReturn(null)
            }
            
            // Act - Generate report with diverse inputs
            val report = reportGenerationService.generateReport(
                classificationResult,
                structureBitmap,
                outputDir
            )
            
            // Assert - Verify consistent structure regardless of input variation
            report shouldNotBe null
            
            // Verify basic structure is maintained
            report!!.filePath.isNotEmpty() shouldBe true
            report.fileName.isNotEmpty() shouldBe true
            report.fileSize shouldNotBe 0L
            report.generationTime shouldNotBe 0L
            
            // Verify filename structure consistency
            report.fileName.shouldContain("livestock_report")
            report.fileName.shouldContain(classificationResult.breed.replace(" ", "_"))
            report.fileName.endsWith(".pdf") shouldBe true
            
            // Verify file path structure
            report.filePath.shouldContain(outputDir.absolutePath)
            report.filePath.shouldContain(report.fileName)
            
            // Verify classification result preservation
            report.classificationResult shouldBe classificationResult
            
            // Verify file exists and has content
            val reportFile = File(report.filePath)
            reportFile.exists() shouldBe true
            reportFile.length() shouldBe report.fileSize
            reportFile.length() shouldNotBe 0L
            
            // Clean up
            reportFile.delete()
            
            // Reset mock for next iteration
            reset(mockTypeClassificationService)
        }
    }
})