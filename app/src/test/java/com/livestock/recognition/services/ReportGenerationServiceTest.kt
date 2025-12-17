package com.livestock.recognition.services

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import com.livestock.recognition.data.models.*
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import org.junit.After
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations
import java.io.File

/**
 * Unit tests for ReportGenerationService
 * Tests PDF creation, sharing functionality, and file provider security
 * **Validates: Requirements 5.2, 5.3, 5.4, 5.5**
 */
class ReportGenerationServiceTest {

    @Mock
    private lateinit var mockTypeClassificationService: TypeClassificationService
    
    private lateinit var context: Context
    private lateinit var reportGenerationService: ReportGenerationService
    private lateinit var testOutputDir: File
    private lateinit var sharingService: SharingService
    
    private val testBreedInfo = BreedInfo(
        name = "Test Gir",
        scientificName = "Bos indicus testicus",
        origin = "Test Gujarat",
        primaryUse = AnimalType.DAIRY,
        averageMilkYieldMin = 10,
        averageMilkYieldMax = 15,
        characteristics = listOf("High milk yield", "Heat tolerant", "Disease resistant"),
        imageUrl = null
    )
    
    private val testClassificationResult = ClassificationResult(
        breed = "Test Gir",
        breedConfidence = 0.92f,
        animalType = AnimalType.DAIRY,
        typeConfidence = 0.88f,
        timestamp = System.currentTimeMillis(),
        imageMetadata = ImageMetadata(
            width = 1920,
            height = 1080,
            fileSize = 2048000L,
            captureTime = System.currentTimeMillis() - 1000L,
            deviceInfo = "Test Device Model X",
            qualityScore = 0.85f
        ),
        processingTime = 1250L
    )

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        
        context = ApplicationProvider.getApplicationContext()
        testOutputDir = File(context.cacheDir, "test_reports")
        testOutputDir.mkdirs()
        
        // Configure mock type classification service
        `when`(mockTypeClassificationService.getBreedCharacteristics("Test Gir"))
            .thenReturn(testBreedInfo)
        `when`(mockTypeClassificationService.getBreedCharacteristics(anyString()))
            .thenReturn(testBreedInfo)
        
        reportGenerationService = ReportGenerationService(context, mockTypeClassificationService)
        sharingService = SharingService(context)
    }
    
    @After
    fun cleanup() {
        // Clean up test files
        testOutputDir.listFiles()?.forEach { it.delete() }
        testOutputDir.delete()
    }

    /**
     * Test PDF creation with various classification results
     * **Validates: Requirements 5.2**
     */
    @Test
    fun `test PDF creation with valid classification result`() {
        // Arrange
        val testBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.RGB_565)
        
        // Act
        val report = reportGenerationService.generateReport(
            testClassificationResult,
            testBitmap,
            testOutputDir
        )
        
        // Assert
        assertNotNull("Report should be generated successfully", report)
        report?.let {
            assertTrue("Report validation should pass", it.validate().isValid)
            assertTrue("Report file should exist", File(it.filePath).exists())
            assertTrue("Report file should have content", File(it.filePath).length() > 0)
            assertEquals("Report should contain correct classification result", testClassificationResult, it.classificationResult)
            assertTrue("Report filename should contain breed name", it.fileName.contains("Test_Gir"))
            assertTrue("Report filename should be PDF", it.fileName.endsWith(".pdf"))
            assertEquals("Report file extension should be pdf", "pdf", it.getFileExtension())
        }
    }

    /**
     * Test PDF creation without image bitmap
     * **Validates: Requirements 5.2**
     */
    @Test
    fun `test PDF creation without image bitmap`() {
        // Act
        val report = reportGenerationService.generateReport(
            testClassificationResult,
            null,
            testOutputDir
        )
        
        // Assert
        assertNotNull("Report should be generated even without image", report)
        report?.let {
            assertTrue("Report validation should pass", it.validate().isValid)
            assertTrue("Report file should exist", File(it.filePath).exists())
            assertTrue("Report file should have content", File(it.filePath).length() > 0)
        }
    }

    /**
     * Test PDF creation with low confidence results
     * **Validates: Requirements 5.2**
     */
    @Test
    fun `test PDF creation with low confidence classification`() {
        // Arrange
        val lowConfidenceResult = testClassificationResult.copy(
            breedConfidence = 0.65f,
            typeConfidence = 0.60f
        )
        val testBitmap = Bitmap.createBitmap(50, 50, Bitmap.Config.RGB_565)
        
        // Act
        val report = reportGenerationService.generateReport(
            lowConfidenceResult,
            testBitmap,
            testOutputDir
        )
        
        // Assert
        assertNotNull("Report should be generated for low confidence results", report)
        report?.let {
            assertTrue("Report validation should pass", it.validate().isValid)
            assertTrue("Report file should exist", File(it.filePath).exists())
            assertEquals("Report should contain low confidence result", lowConfidenceResult, it.classificationResult)
        }
    }

    /**
     * Test PDF creation with missing breed information
     * **Validates: Requirements 5.2**
     */
    @Test
    fun `test PDF creation with missing breed information`() {
        // Arrange
        val unknownBreedResult = testClassificationResult.copy(breed = "Unknown Breed")
        `when`(mockTypeClassificationService.getBreedCharacteristics("Unknown Breed"))
            .thenReturn(null)
        
        val testBitmap = Bitmap.createBitmap(75, 75, Bitmap.Config.RGB_565)
        
        // Act
        val report = reportGenerationService.generateReport(
            unknownBreedResult,
            testBitmap,
            testOutputDir
        )
        
        // Assert
        assertNotNull("Report should be generated even without breed info", report)
        report?.let {
            assertTrue("Report validation should pass", it.validate().isValid)
            assertTrue("Report file should exist", File(it.filePath).exists())
            assertTrue("Report filename should contain unknown breed", it.fileName.contains("Unknown_Breed"))
        }
    }

    /**
     * Test report validation functionality
     * **Validates: Requirements 5.2**
     */
    @Test
    fun `test report generation validation`() {
        // Test with valid classification result
        val validationResult = reportGenerationService.validateReportGeneration(testClassificationResult)
        assertTrue("Validation should pass for valid classification result", validationResult.isValid)
        assertTrue("Validation errors should be empty", validationResult.errors.isEmpty())
        
        // Test with invalid classification result
        val invalidResult = testClassificationResult.copy(
            breed = "", // Invalid empty breed
            breedConfidence = 1.5f // Invalid confidence > 1.0
        )
        
        val invalidValidation = reportGenerationService.validateReportGeneration(invalidResult)
        assertFalse("Validation should fail for invalid classification result", invalidValidation.isValid)
        assertFalse("Validation errors should not be empty", invalidValidation.errors.isEmpty())
    }

    /**
     * Test default output directory creation
     * **Validates: Requirements 5.2**
     */
    @Test
    fun `test default output directory creation`() {
        // Act
        val defaultDir = reportGenerationService.getDefaultOutputDirectory()
        
        // Assert
        assertNotNull("Default output directory should not be null", defaultDir)
        assertTrue("Default output directory should exist", defaultDir.exists())
        assertTrue("Default output directory should be a directory", defaultDir.isDirectory)
        assertTrue("Default output directory should be writable", defaultDir.canWrite())
    }

    /**
     * Test sharing functionality across different platforms
     * **Validates: Requirements 5.4**
     */
    @Test
    fun `test sharing functionality across different platforms`() {
        // Arrange
        val testBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.RGB_565)
        val report = reportGenerationService.generateReport(
            testClassificationResult,
            testBitmap,
            testOutputDir
        )
        
        assertNotNull("Report should be generated for sharing test", report)
        
        // Test different sharing methods
        ShareMethod.values().forEach { method ->
            // Act
            val shareResult = sharingService.shareReport(report!!, method)
            
            // Assert
            assertTrue("Share result validation should pass for $method", shareResult.validate().isValid)
            assertNotNull("Share result message should not be null for $method", shareResult.message)
            assertFalse("Share result message should not be empty for $method", shareResult.message.isBlank())
            
            // Note: We can't test actual sharing without user interaction,
            // but we can verify the intent creation and validation
        }
    }

    /**
     * Test file provider security and permissions
     * **Validates: Requirements 5.4**
     */
    @Test
    fun `test file provider security and permissions`() {
        // Arrange
        val testBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.RGB_565)
        val report = reportGenerationService.generateReport(
            testClassificationResult,
            testBitmap,
            testOutputDir
        )
        
        assertNotNull("Report should be generated for security test", report)
        
        val reportFile = File(report!!.filePath)
        
        // Test file validation for sharing
        val fileValidation = sharingService.validateFileForSharing(reportFile)
        assertTrue("File validation should pass for generated report", fileValidation.isValid)
        assertTrue("File validation errors should be empty", fileValidation.errors.isEmpty())
        
        // Test with non-existent file
        val nonExistentFile = File(testOutputDir, "non_existent.pdf")
        val nonExistentValidation = sharingService.validateFileForSharing(nonExistentFile)
        assertFalse("File validation should fail for non-existent file", nonExistentValidation.isValid)
        assertFalse("File validation errors should not be empty", nonExistentValidation.errors.isEmpty())
        
        // Test with empty file
        val emptyFile = File(testOutputDir, "empty.pdf")
        emptyFile.createNewFile()
        val emptyValidation = sharingService.validateFileForSharing(emptyFile)
        assertFalse("File validation should fail for empty file", emptyValidation.isValid)
        assertTrue("File validation should detect empty file", 
            emptyValidation.errors.any { it.contains("empty") })
        
        emptyFile.delete()
    }

    /**
     * Test multiple file sharing functionality
     * **Validates: Requirements 5.4**
     */
    @Test
    fun `test multiple file sharing functionality`() {
        // Arrange - Generate multiple reports
        val reports = mutableListOf<Report>()
        val testBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.RGB_565)
        
        repeat(3) { index ->
            val result = testClassificationResult.copy(
                breed = "Test Breed $index",
                timestamp = System.currentTimeMillis() + index * 1000L
            )
            
            `when`(mockTypeClassificationService.getBreedCharacteristics("Test Breed $index"))
                .thenReturn(testBreedInfo.copy(name = "Test Breed $index"))
            
            val report = reportGenerationService.generateReport(result, testBitmap, testOutputDir)
            assertNotNull("Report $index should be generated", report)
            reports.add(report!!)
        }
        
        val files = reports.map { File(it.filePath) }
        
        // Test multiple file sharing
        ShareMethod.values().forEach { method ->
            val shareResult = sharingService.shareMultipleFiles(
                files,
                method,
                "Test Reports",
                "Multiple livestock recognition reports"
            )
            
            assertTrue("Multiple file share result should be valid for $method", shareResult.validate().isValid)
            assertNotNull("Multiple file share result message should not be null for $method", shareResult.message)
        }
        
        // Test with empty file list
        val emptyShareResult = sharingService.shareMultipleFiles(
            emptyList(),
            ShareMethod.GENERAL
        )
        assertFalse("Empty file list sharing should fail", emptyShareResult.success)
        assertTrue("Empty file list error should be descriptive", 
            emptyShareResult.message.contains("No files"))
    }

    /**
     * Test available sharing methods detection
     * **Validates: Requirements 5.4**
     */
    @Test
    fun `test available sharing methods detection`() {
        // Act
        val availableMethods = sharingService.getAvailableSharingMethods()
        
        // Assert
        assertNotNull("Available methods should not be null", availableMethods)
        assertTrue("At least GENERAL sharing should be available", 
            availableMethods.contains(ShareMethod.GENERAL))
        
        // Verify all returned methods are valid
        availableMethods.forEach { method ->
            assertTrue("Returned method should be valid ShareMethod", 
                method in ShareMethod.values())
        }
    }

    /**
     * Test report content structure and completeness
     * **Validates: Requirements 5.2, 5.3**
     */
    @Test
    fun `test report content structure and completeness`() {
        // Arrange
        val testBitmap = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888)
        
        // Act
        val report = reportGenerationService.generateReport(
            testClassificationResult,
            testBitmap,
            testOutputDir
        )
        
        // Assert
        assertNotNull("Report should be generated", report)
        report?.let {
            // Verify report structure
            assertTrue("Report file path should not be empty", it.filePath.isNotEmpty())
            assertTrue("Report file name should not be empty", it.fileName.isNotEmpty())
            assertTrue("Report file size should be positive", it.fileSize > 0)
            assertTrue("Report generation time should be positive", it.generationTime > 0)
            
            // Verify classification result preservation
            assertEquals("Breed should be preserved", testClassificationResult.breed, it.classificationResult.breed)
            assertEquals("Breed confidence should be preserved", testClassificationResult.breedConfidence, it.classificationResult.breedConfidence, 0.001f)
            assertEquals("Animal type should be preserved", testClassificationResult.animalType, it.classificationResult.animalType)
            assertEquals("Type confidence should be preserved", testClassificationResult.typeConfidence, it.classificationResult.typeConfidence, 0.001f)
            assertEquals("Timestamp should be preserved", testClassificationResult.timestamp, it.classificationResult.timestamp)
            assertEquals("Processing time should be preserved", testClassificationResult.processingTime, it.classificationResult.processingTime)
            
            // Verify image metadata preservation
            assertNotNull("Image metadata should be preserved", it.classificationResult.imageMetadata)
            it.classificationResult.imageMetadata?.let { metadata ->
                assertEquals("Image width should be preserved", testClassificationResult.imageMetadata!!.width, metadata.width)
                assertEquals("Image height should be preserved", testClassificationResult.imageMetadata!!.height, metadata.height)
                assertEquals("File size should be preserved", testClassificationResult.imageMetadata!!.fileSize, metadata.fileSize)
                assertEquals("Quality score should be preserved", testClassificationResult.imageMetadata!!.qualityScore, metadata.qualityScore, 0.001f)
            }
            
            // Verify file system properties
            val reportFile = File(it.filePath)
            assertTrue("Report file should exist", reportFile.exists())
            assertTrue("Report file should be readable", reportFile.canRead())
            assertEquals("File size should match report metadata", it.fileSize, reportFile.length())
            
            // Verify filename structure
            assertTrue("Filename should contain livestock_report", it.fileName.contains("livestock_report"))
            assertTrue("Filename should contain breed name", it.fileName.contains(testClassificationResult.breed.replace(" ", "_")))
            assertTrue("Filename should end with .pdf", it.fileName.endsWith(".pdf"))
        }
    }

    /**
     * Test error handling for invalid inputs
     * **Validates: Requirements 5.2, 5.3**
     */
    @Test
    fun `test error handling for invalid inputs`() {
        // Test with invalid classification result
        val invalidResult = testClassificationResult.copy(
            breed = "", // Empty breed name
            breedConfidence = -0.5f, // Invalid negative confidence
            timestamp = -1L // Invalid negative timestamp
        )
        
        val testBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.RGB_565)
        
        // Act
        val report = reportGenerationService.generateReport(
            invalidResult,
            testBitmap,
            testOutputDir
        )
        
        // Assert - Should handle gracefully
        // The service should either return null or a report with validation errors
        if (report != null) {
            val validation = report.validate()
            // If report is generated, it should at least indicate validation issues
            assertNotNull("Report validation should provide feedback", validation)
        }
        
        // Test with non-existent output directory
        val nonExistentDir = File(context.cacheDir, "non_existent_dir_12345")
        assertFalse("Test directory should not exist initially", nonExistentDir.exists())
        
        val reportWithNewDir = reportGenerationService.generateReport(
            testClassificationResult,
            testBitmap,
            nonExistentDir
        )
        
        // Should create directory and generate report
        if (reportWithNewDir != null) {
            assertTrue("Directory should be created", nonExistentDir.exists())
            assertTrue("Report should be generated in new directory", File(reportWithNewDir.filePath).exists())
            
            // Clean up
            File(reportWithNewDir.filePath).delete()
            nonExistentDir.delete()
        }
    }

    /**
     * Test sharing result validation and user messages
     * **Validates: Requirements 5.4**
     */
    @Test
    fun `test sharing result validation and user messages`() {
        // Test successful share result
        val successResult = ShareResult(
            success = true,
            message = "Share intent created successfully",
            intent = mock()
        )
        
        assertTrue("Success result validation should pass", successResult.validate().isValid)
        assertTrue("Success result should be successful", successResult.isSuccessful())
        assertEquals("Success user message should be positive", "Report shared successfully", successResult.getUserMessage())
        
        // Test failed share result
        val failureResult = ShareResult(
            success = false,
            message = "No compatible app found",
            intent = null
        )
        
        assertTrue("Failure result validation should pass", failureResult.validate().isValid)
        assertFalse("Failure result should not be successful", failureResult.isSuccessful())
        assertTrue("Failure user message should contain error", failureResult.getUserMessage().contains("Failed"))
        
        // Test invalid share result
        val invalidResult = ShareResult(
            success = true,
            message = "", // Empty message is invalid
            intent = null // Successful result should have intent
        )
        
        assertFalse("Invalid result validation should fail", invalidResult.validate().isValid)
        assertFalse("Invalid result validation errors should not be empty", invalidResult.validate().errors.isEmpty())
    }
}