package com.livestock.recognition.ui.main

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.livestock.recognition.LivestockApp
import com.livestock.recognition.R
import com.livestock.recognition.databinding.ActivityMainBinding
import com.livestock.recognition.image.ImageFiles
import com.livestock.recognition.ui.camera.CameraActivity
import com.livestock.recognition.ui.history.HistoryActivity
import com.livestock.recognition.ui.results.ResultsActivity
import kotlinx.coroutines.launch

/**
 * Home screen: entry points for capturing a photo, picking one from the
 * gallery, and browsing past classifications. Surfaces a warning when the
 * build ships without a bundled model.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val container get() = (application as LivestockApp).container

    private val captureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val path = result.data?.getStringExtra(CameraActivity.EXTRA_IMAGE_PATH)
        if (result.resultCode == RESULT_OK && path != null) {
            openResults(path)
        }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri == null) return@registerForActivityResult
        binding.pickProgress.visibility = View.VISIBLE
        lifecycleScope.launch {
            val copy = ImageFiles.copyToAppStorage(this@MainActivity, uri)
            binding.pickProgress.visibility = View.GONE
            if (copy != null) {
                openResults(copy.absolutePath)
            } else {
                Toast.makeText(this@MainActivity, R.string.error_image_load, Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.takePhotoButton.setOnClickListener {
            captureLauncher.launch(Intent(this, CameraActivity::class.java))
        }
        binding.pickImageButton.setOnClickListener {
            pickImageLauncher.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }
        binding.historyButton.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }
    }

    override fun onResume() {
        super.onResume()
        val modelBundled = container.classifierProvider.isModelBundled()
        binding.modelWarningCard.visibility = if (modelBundled) View.GONE else View.VISIBLE
    }

    private fun openResults(imagePath: String) {
        startActivity(ResultsActivity.newClassificationIntent(this, imagePath))
    }
}
