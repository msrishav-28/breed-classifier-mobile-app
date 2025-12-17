package com.livestock.recognition.ui.results

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.livestock.recognition.R
import com.livestock.recognition.data.models.ClassificationResult
import com.livestock.recognition.databinding.ActivityResultsBinding
import com.livestock.recognition.services.ImageAnnotationService

/**
 * ResultsActivity displays comprehensive breed identification results with detailed information.
 * Shows breed name, type, confidence scores, characteristics, and warning messages for low confidence.
 * 
 * Requirements: 2.5, 3.3, 7.3
 */
class ResultsActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityResultsBinding
    private lateinit var viewModel: ResultsViewModel
    private lateinit var annotationService: ImageAnnotationService
    
    companion object {
        const val EXTRA_CLASSIFICATION_RESULT = "extra_classification_result"
        const val EXTRA_IMAGE_PATH = "extra_image_path"
        
        fun createIntent(context: Context, result: ClassificationResult, imagePath: String): Intent {
            return Intent(context, ResultsActivity::class.java).apply {
                putExtra(EXTRA_CLASSIFICATION_RESULT, result)
                putExtra(EXTRA_IMAGE_PATH, imagePath)
            }
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        viewModel = ViewModelProvider(this)[ResultsViewModel::class.java]
        annotationService = ImageAnnotationService()
        
        // Get data from intent
        val classificationResult = intent.getParcelableExtra<ClassificationResult>(EXTRA_CLASSIFICATION_RESULT)
        val imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)
        
        if (classificationResult == null || imagePath == null) {
            finish()
            return
        }
        
        // Initialize UI
        setupUI()
        
        // Load and display results
        viewModel.loadResults(classificationResult, imagePath)
        
        // Observe view model
        observeViewModel()
    }
    
    private fun setupUI() {
        // Set up back button
        binding.backButton.setOnClickListener {
            finish()
        }
        
        // Set up share button
        binding.shareButton.setOnClickListener {
            viewModel.shareResults()
        }
        
        // Set up retake button
        binding.retakeButton.setOnClickListener {
            finish() // Return to previous screen for retaking
        }
    }
    
    private fun observeViewModel() {
        viewModel.resultsState.observe(this) { state ->
            when (state) {
                is ResultsState.Loading -> {
                    binding.progressBar.visibility = View.VISIBLE
                    binding.contentScrollView.visibility = View.GONE
                }
                is ResultsState.Success -> {
                    binding.progressBar.visibility = View.GONE
                    binding.contentScrollView.visibility = View.VISIBLE
                    displayResults(state.data)
                }
                is ResultsState.Error -> {
                    binding.progressBar.visibility = View.GONE
                    binding.errorText.visibility = View.VISIBLE
                    binding.errorText.text = state.message
                }
            }
        }
        
        viewModel.warningMessage.observe(this) { warning ->
            if (warning != null) {
                binding.warningCard.visibility = View.VISIBLE
                binding.warningText.text = warning
                binding.retakeButton.visibility = View.VISIBLE
            } else {
                binding.warningCard.visibility = View.GONE
                binding.retakeButton.visibility = View.GONE
            }
        }
    }
    
    private fun displayResults(data: ResultsDisplayData) {
        // Display captured image with annotations
        val originalBitmap = BitmapFactory.decodeFile(data.imagePath)
        val annotatedBitmap = annotationService.createAnnotatedImage(
            originalBitmap,
            data.classificationResult
        )
        binding.capturedImage.setImageBitmap(annotatedBitmap)
        
        // Display breed information
        binding.breedNameText.text = data.classificationResult.breed
        binding.confidenceText.text = getString(
            R.string.confidence_format, 
            (data.classificationResult.breedConfidence * 100).toInt()
        )
        
        // Display animal type
        binding.animalTypeText.text = data.classificationResult.animalType.displayName
        binding.typeDescriptionText.text = data.classificationResult.animalType.description
        binding.typeConfidenceText.text = getString(
            R.string.type_confidence_format,
            (data.classificationResult.typeConfidence * 100).toInt()
        )
        
        // Display breed characteristics if available
        data.breedInfo?.let { breedInfo ->
            binding.breedInfoCard.visibility = View.VISIBLE
            binding.scientificNameText.text = breedInfo.scientificName
            binding.originText.text = breedInfo.origin
            
            // Display milk yield range
            if (breedInfo.averageMilkYieldMin > 0 || breedInfo.averageMilkYieldMax > 0) {
                binding.milkYieldText.text = getString(
                    R.string.milk_yield_format,
                    breedInfo.averageMilkYieldMin,
                    breedInfo.averageMilkYieldMax
                )
                binding.milkYieldText.visibility = View.VISIBLE
            } else {
                binding.milkYieldText.visibility = View.GONE
            }
            
            // Display characteristics
            if (breedInfo.characteristics.isNotEmpty()) {
                binding.characteristicsText.text = breedInfo.characteristics.joinToString("\n• ", "• ")
                binding.characteristicsText.visibility = View.VISIBLE
            } else {
                binding.characteristicsText.visibility = View.GONE
            }
        } ?: run {
            binding.breedInfoCard.visibility = View.GONE
        }
        
        // Display processing information
        binding.processingTimeText.text = getString(
            R.string.processing_time_format,
            data.classificationResult.processingTime
        )
        
        // Display image metadata
        val metadata = data.classificationResult.imageMetadata
        binding.imageInfoText.text = getString(
            R.string.image_info_format,
            metadata.width,
            metadata.height,
            metadata.qualityScore
        )
    }
}