package com.bananaripeness

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.bananaripeness.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var classifier: RipenessClassifier
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else showPermissionDenied()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classifier = RipenessClassifier(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }

        binding.captureButton.setOnClickListener { captureAndClassify() }
        binding.retakeButton.setOnClickListener { showViewfinder() }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewfinder.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                preview, imageCapture)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndClassify() {
        binding.captureButton.isEnabled = false
        binding.loadingIndicator.visibility = View.VISIBLE

        imageCapture?.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(imageProxy: ImageProxy) {
                val buffer = imageProxy.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                imageProxy.close()

                lifecycleScope.launch {
                    val prediction = withContext(Dispatchers.Default) {
                        classifier.predict(bitmap)
                    }
                    showResult(prediction)
                }
            }

            override fun onError(exception: ImageCaptureException) {
                lifecycleScope.launch { resetCapture() }
            }
        })
    }

    private fun showResult(prediction: Prediction) {
        binding.loadingIndicator.visibility = View.GONE
        binding.viewfinderContainer.visibility = View.GONE
        binding.resultContainer.visibility = View.VISIBLE

        val stage = prediction.stage
        binding.resultCard.setCardBackgroundColor(ContextCompat.getColor(this, stage.colorRes))
        binding.stageLabel.text = stage.description
        binding.confidenceLabel.text = "Confidence: ${(prediction.confidence * 100).toInt()}%"

        if (prediction.lowConfidence) {
            binding.lowConfidenceWarning.visibility = View.VISIBLE
            binding.retakeButton.visibility = View.VISIBLE
        } else {
            binding.lowConfidenceWarning.visibility = View.GONE
            binding.retakeButton.visibility = View.VISIBLE
        }
    }

    private fun showViewfinder() {
        binding.resultContainer.visibility = View.GONE
        binding.viewfinderContainer.visibility = View.VISIBLE
        binding.captureButton.isEnabled = true
    }

    private fun resetCapture() {
        binding.loadingIndicator.visibility = View.GONE
        binding.captureButton.isEnabled = true
    }

    private fun showPermissionDenied() {
        binding.captureButton.isEnabled = false
        binding.stageLabel.text = getString(R.string.camera_permission_denied)
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
        cameraExecutor.shutdown()
    }
}
