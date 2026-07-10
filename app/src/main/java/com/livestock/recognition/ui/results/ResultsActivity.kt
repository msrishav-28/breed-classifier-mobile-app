package com.livestock.recognition.ui.results

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.livestock.recognition.R
import com.livestock.recognition.core.catalog.BreedNames
import com.livestock.recognition.databinding.ActivityResultsBinding
import com.livestock.recognition.image.BitmapLoader
import com.livestock.recognition.report.ReportSharer
import com.livestock.recognition.ui.common.confidencePercent
import com.livestock.recognition.ui.common.displayNameRes
import com.livestock.recognition.ui.common.formatDateTime
import com.livestock.recognition.ui.common.messageRes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Shows the outcome of a classification: photo, best match with confidence,
 * alternatives, catalog information and warnings. Supports sharing a PDF
 * report.
 */
class ResultsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultsBinding
    private lateinit var viewModel: ResultsViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel = ViewModelProvider(this)[ResultsViewModel::class.java]

        binding.toolbar.setNavigationOnClickListener { finish() }
        binding.shareButton.setOnClickListener { viewModel.shareReport() }
        binding.doneButton.setOnClickListener { finish() }

        observeState()
        observeShareEvents()
        start()
    }

    private fun start() {
        val recordId = intent.getLongExtra(EXTRA_RECORD_ID, NO_RECORD)
        val imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)
        when {
            recordId != NO_RECORD -> viewModel.loadSavedClassification(recordId)
            imagePath != null -> viewModel.startNewClassification(imagePath)
            else -> finish()
        }
    }

    private fun observeState() {
        viewModel.state.observe(this) { state ->
            binding.loadingGroup.visibility =
                if (state is ResultsViewModel.UiState.Loading) View.VISIBLE else View.GONE
            binding.contentScroll.visibility =
                if (state is ResultsViewModel.UiState.Ready) View.VISIBLE else View.GONE
            binding.buttonBar.visibility =
                if (state is ResultsViewModel.UiState.Ready) View.VISIBLE else View.GONE
            binding.errorGroup.visibility =
                if (state is ResultsViewModel.UiState.Error ||
                    state is ResultsViewModel.UiState.ModelUnavailable
                ) View.VISIBLE else View.GONE

            when (state) {
                is ResultsViewModel.UiState.Ready -> renderResult(state)
                is ResultsViewModel.UiState.Error ->
                    binding.errorText.text = getString(state.messageRes)
                is ResultsViewModel.UiState.ModelUnavailable -> {
                    binding.errorText.text = getString(R.string.model_unavailable_message)
                    binding.errorDetailText.visibility = View.VISIBLE
                    binding.errorDetailText.text = state.detail
                }
                is ResultsViewModel.UiState.Loading -> Unit
            }
        }
    }

    private fun observeShareEvents() {
        viewModel.shareEvent.observe(this) { event ->
            binding.shareButton.isEnabled = event != ResultsViewModel.ShareEvent.InProgress
            when (event) {
                is ResultsViewModel.ShareEvent.Ready -> {
                    startActivity(ReportSharer.shareIntent(this, event.report))
                    viewModel.consumeShareEvent()
                }
                is ResultsViewModel.ShareEvent.Failed -> {
                    Toast.makeText(this, R.string.error_report_generation, Toast.LENGTH_LONG).show()
                    viewModel.consumeShareEvent()
                }
                else -> Unit
            }
        }
    }

    private fun renderResult(state: ResultsViewModel.UiState.Ready) {
        loadPhoto(state.imagePath)

        val record = state.record
        binding.breedNameText.text = BreedNames.displayName(record.breedLabel)
        binding.confidenceText.text =
            getString(R.string.confidence_format, confidencePercent(record.confidence))
        binding.confidenceBar.progress = confidencePercent(record.confidence)

        val type = record.animalType
        binding.animalTypeText.visibility = if (type != null) View.VISIBLE else View.GONE
        type?.let { binding.animalTypeText.text = getString(it.displayNameRes()) }

        if (record.alternatives.isEmpty()) {
            binding.alternativesText.visibility = View.GONE
        } else {
            binding.alternativesText.visibility = View.VISIBLE
            binding.alternativesText.text = getString(
                R.string.alternatives_format,
                record.alternatives.joinToString(", ") {
                    "${BreedNames.displayName(it.label)} (${confidencePercent(it.confidence)}%)"
                }
            )
        }

        renderWarnings(state)
        renderBreedInfo(state)

        binding.processingInfoText.text = getString(
            R.string.processing_info_format,
            formatDateTime(record.capturedAtEpochMillis),
            record.processingTimeMillis,
        )
    }

    private fun renderWarnings(state: ResultsViewModel.UiState.Ready) {
        val warnings = mutableListOf<String>()
        if (state.showConfidenceWarning) {
            warnings.add(getString(R.string.low_confidence_warning))
        }
        state.qualityIssues.forEach { warnings.add(getString(it.messageRes())) }

        binding.warningCard.visibility = if (warnings.isEmpty()) View.GONE else View.VISIBLE
        binding.warningText.text = warnings.joinToString("\n")
    }

    private fun renderBreedInfo(state: ResultsViewModel.UiState.Ready) {
        val info = state.breedInfo
        if (info == null) {
            binding.breedInfoCard.visibility = View.GONE
            return
        }
        binding.breedInfoCard.visibility = View.VISIBLE
        binding.speciesText.text = getString(R.string.species_format, info.species)
        binding.originText.text = getString(R.string.origin_format, info.origin)

        binding.milkYieldText.visibility = if (info.hasMilkYield) View.VISIBLE else View.GONE
        if (info.hasMilkYield) {
            binding.milkYieldText.text = getString(
                R.string.milk_yield_format,
                info.milkYieldMinLitresPerDay,
                info.milkYieldMaxLitresPerDay,
            )
        }

        binding.characteristicsText.visibility =
            if (info.characteristics.isEmpty()) View.GONE else View.VISIBLE
        binding.characteristicsText.text =
            info.characteristics.joinToString(separator = "\n") { "- $it" }
    }

    private fun loadPhoto(imagePath: String) {
        lifecycleScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                BitmapLoader.decode(imagePath, PHOTO_MAX_DIMENSION)
            }
            if (bitmap != null) {
                binding.photoView.setImageBitmap(bitmap)
            } else {
                binding.photoView.setImageResource(R.drawable.ic_gallery)
            }
        }
    }

    companion object {
        private const val EXTRA_IMAGE_PATH = "extra_image_path"
        private const val EXTRA_RECORD_ID = "extra_record_id"
        private const val NO_RECORD = -1L
        private const val PHOTO_MAX_DIMENSION = 1280

        fun newClassificationIntent(context: Context, imagePath: String): Intent =
            Intent(context, ResultsActivity::class.java)
                .putExtra(EXTRA_IMAGE_PATH, imagePath)

        fun savedRecordIntent(context: Context, recordId: Long): Intent =
            Intent(context, ResultsActivity::class.java)
                .putExtra(EXTRA_RECORD_ID, recordId)
    }
}
