package com.livestock.recognition.ui.camera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.livestock.recognition.R
import com.livestock.recognition.databinding.ActivityCameraBinding
import com.livestock.recognition.image.ImageFiles

/**
 * Full-screen CameraX capture. Returns the saved photo's absolute path to
 * the caller via [EXTRA_IMAGE_PATH].
 */
class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding

    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.captureButton.isEnabled = false
        binding.captureButton.setOnClickListener { takePhoto() }
        binding.backButton.setOnClickListener { finish() }

        if (hasCameraPermission()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun hasCameraPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }
                val capture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()
                imageCapture = capture

                provider.unbindAll()
                camera = provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture
                )
                binding.captureButton.isEnabled = true
            } catch (e: Exception) {
                showError(getString(R.string.camera_start_error))
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        binding.captureButton.isEnabled = false
        binding.progressBar.visibility = View.VISIBLE

        val photoFile = ImageFiles.newImageFile(this)
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exception: ImageCaptureException) {
                    binding.captureButton.isEnabled = true
                    binding.progressBar.visibility = View.GONE
                    showError(getString(R.string.camera_capture_error))
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val result = Intent().putExtra(EXTRA_IMAGE_PATH, photoFile.absolutePath)
                    setResult(RESULT_OK, result)
                    finish()
                }
            }
        )
    }

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
    }

    companion object {
        const val EXTRA_IMAGE_PATH = "extra_image_path"
    }
}
