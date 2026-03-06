package com.micklab.imagene

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat

/**
 * Main Activity for SDXL Image Generator.
 * Performs model existence check on startup and navigates to error screen if missing.
 */
class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 1002
    }
    
    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar
    
    private val modelMissingLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            // User retried and model is now available
            proceedToMainScreen()
        } else {
            // User exited from error screen
            finish()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppLogStore.initialize(this)
        AppLogStore.i(TAG, "MainActivity onCreate")
        setupLoadingUI()
        checkPermissionsAndModel()
    }
    
    private fun setupLoadingUI() {
        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#1a1a2e"))
            setPadding(48, 48, 48, 48)
        }
        
        // Title
        val title = TextView(this).apply {
            text = "SDXL Image Generator"
            textSize = 24f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 48)
        }
        mainLayout.addView(title)
        
        // Progress bar
        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleLarge).apply {
            setPadding(0, 0, 0, 32)
        }
        mainLayout.addView(progressBar)
        
        // Status text
        statusText = TextView(this).apply {
            text = "初期化中..."
            textSize = 16f
            setTextColor(Color.LTGRAY)
            gravity = Gravity.CENTER
        }
        mainLayout.addView(statusText)
        
        setContentView(mainLayout)
    }
    
    private fun checkPermissionsAndModel() {
        AppLogStore.i(TAG, "Checking storage permission and model availability")
        // Check if we need storage permission (Android 12 and below)
        if (SdxlModelLoader.needsStoragePermission() && !SdxlModelLoader.hasStoragePermission(this)) {
            AppLogStore.w(TAG, "Storage permission is missing; requesting permission")
            requestStoragePermission()
        } else {
            checkModelAndProceed()
        }
    }
    
    private fun requestStoragePermission() {
        statusText.text = "ストレージ権限を確認中..."
        AppLogStore.i(TAG, "Requesting READ_EXTERNAL_STORAGE permission")
        
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                PERMISSION_REQUEST_CODE
            )
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                AppLogStore.i(TAG, "Storage permission granted")
                checkModelAndProceed()
            } else {
                // Permission denied, still try to check model (might work on some devices)
                AppLogStore.w(TAG, "Storage permission denied; continuing model check")
                checkModelAndProceed()
            }
        }
    }
    
    private fun checkModelAndProceed() {
        statusText.text = "モデルを確認中..."
        
        // Check if model exists
        if (SdxlModelLoader.isModelAvailable()) {
            AppLogStore.i(TAG, "Model check passed")
            proceedToMainScreen()
        } else {
            AppLogStore.w(TAG, "Model check failed: ${SdxlModelLoader.getMissingComponents()}")
            navigateToErrorScreen()
        }
    }
    
    private fun navigateToErrorScreen() {
        AppLogStore.i(TAG, "Navigating to ModelMissingActivity")
        val intent = Intent(this, ModelMissingActivity::class.java)
        modelMissingLauncher.launch(intent)
    }
    
    private fun proceedToMainScreen() {
        statusText.text = "生成画面へ移動中..."
        progressBar.visibility = android.view.View.GONE
        AppLogStore.i(TAG, "Proceeding to main generation UI")
        startActivity(Intent(this, GenerationActivity::class.java))
        finish()
    }
    
    private fun showMainGenerationUI() {
        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#1a1a2e"))
            setPadding(48, 48, 48, 48)
        }
        
        val successText = TextView(this).apply {
            text = "✅ モデルが見つかりました"
            textSize = 20f
            setTextColor(Color.parseColor("#4caf50"))
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 24)
        }
        mainLayout.addView(successText)
        
        val pathText = TextView(this).apply {
            text = "パス: ${SdxlModelLoader.getModelBasePath()}"
            textSize = 14f
            setTextColor(Color.LTGRAY)
            gravity = Gravity.CENTER
        }
        mainLayout.addView(pathText)
        
        // Show available components
        val componentsTitle = TextView(this).apply {
            text = "\n利用中のランタイムコンポーネント:"
            textSize = 16f
            setTextColor(Color.WHITE)
            setPadding(0, 32, 0, 16)
        }
        mainLayout.addView(componentsTitle)
        
        val components = SdxlModelLoader.getRequiredRuntimeComponents()
        for (component in components) {
            val componentText = TextView(this).apply {
                text = "✓ $component"
                textSize = 14f
                setTextColor(Color.parseColor("#81c784"))
                setPadding(0, 4, 0, 4)
            }
            mainLayout.addView(componentText)
        }
        
        setContentView(mainLayout)
    }

    override fun onDestroy() {
        AppLogStore.i(TAG, "MainActivity onDestroy")
        super.onDestroy()
    }
}
