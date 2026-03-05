package com.micklab.imagene

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * Error screen displayed when SDXL model is not found.
 * Shows instructions for placing the model in the correct location.
 */
class ModelMissingActivity : Activity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupUI()
    }
    
    private fun setupUI() {
        val scrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#1a1a2e"))
        }
        
        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 64, 48, 64)
        }
        
        // Error icon (using text)
        val errorIcon = TextView(this).apply {
            text = "⚠️"
            textSize = 64f
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 32)
        }
        mainLayout.addView(errorIcon)
        
        // Title
        val title = TextView(this).apply {
            text = "モデルがありません"
            textSize = 28f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 24)
        }
        mainLayout.addView(title)
        
        // Main message
        val message = TextView(this).apply {
            text = "SDXLモデルが見つかりませんでした。\n以下の場所にモデルを配置してください:"
            textSize = 16f
            setTextColor(Color.LTGRAY)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 24)
        }
        mainLayout.addView(message)
        
        // Path box
        val pathBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#16213e"))
            setPadding(32, 24, 32, 24)
        }
        
        val pathLabel = TextView(this).apply {
            text = "配置場所:"
            textSize = 14f
            setTextColor(Color.GRAY)
        }
        pathBox.addView(pathLabel)
        
        val pathText = TextView(this).apply {
            text = "Downloads/sdxl/"
            textSize = 18f
            setTextColor(Color.parseColor("#4fc3f7"))
            typeface = android.graphics.Typeface.MONOSPACE
            setPadding(0, 8, 0, 0)
        }
        pathBox.addView(pathText)
        
        val fullPathText = TextView(this).apply {
            text = SdxlModelLoader.getModelBasePath()
            textSize = 12f
            setTextColor(Color.GRAY)
            typeface = android.graphics.Typeface.MONOSPACE
            setPadding(0, 4, 0, 0)
        }
        pathBox.addView(fullPathText)
        
        mainLayout.addView(pathBox)
        
        // Required structure
        val structureTitle = TextView(this).apply {
            text = "\n必要なフォルダ構成:"
            textSize = 16f
            setTextColor(Color.WHITE)
            setPadding(0, 32, 0, 16)
        }
        mainLayout.addView(structureTitle)
        
        val requiredFolders = listOf(
            "📁 unet/",
            "📁 text_encoder/",
            "📁 text_encoder_2/",
            "📁 vae_decoder/",
            "📁 tokenizer/",
            "📁 tokenizer_2/",
            "📁 scheduler/",
            "📄 model_index.json"
        )
        
        val structureBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#16213e"))
            setPadding(32, 24, 32, 24)
        }
        
        for (folder in requiredFolders) {
            val folderText = TextView(this).apply {
                text = folder
                textSize = 14f
                setTextColor(Color.LTGRAY)
                typeface = android.graphics.Typeface.MONOSPACE
                setPadding(0, 4, 0, 4)
            }
            structureBox.addView(folderText)
        }
        mainLayout.addView(structureBox)
        
        // Show missing components if any
        val missingComponents = SdxlModelLoader.getMissingComponents()
        if (missingComponents.isNotEmpty()) {
            val missingTitle = TextView(this).apply {
                text = "\n❌ 見つからないコンポーネント:"
                textSize = 16f
                setTextColor(Color.parseColor("#e94560"))
                setPadding(0, 24, 0, 8)
            }
            mainLayout.addView(missingTitle)
            
            val missingBox = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setBackgroundColor(Color.parseColor("#2d1f2f"))
                setPadding(32, 24, 32, 24)
            }
            
            for (component in missingComponents) {
                val componentText = TextView(this).apply {
                    text = "• $component"
                    textSize = 14f
                    setTextColor(Color.parseColor("#ff8a80"))
                    setPadding(0, 4, 0, 4)
                }
                missingBox.addView(componentText)
            }
            mainLayout.addView(missingBox)
        }
        
        // Retry button
        val retryButton = Button(this).apply {
            text = "再試行"
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#e94560"))
            setPadding(48, 24, 48, 24)
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            params.setMargins(0, 48, 0, 16)
            layoutParams = params
            setOnClickListener { onRetryClicked() }
        }
        mainLayout.addView(retryButton)
        
        // Exit button
        val exitButton = Button(this).apply {
            text = "終了"
            textSize = 14f
            setTextColor(Color.LTGRAY)
            setBackgroundColor(Color.parseColor("#333355"))
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            layoutParams = params
            setOnClickListener { finish() }
        }
        mainLayout.addView(exitButton)
        
        scrollView.addView(mainLayout)
        setContentView(scrollView)
    }
    
    private fun onRetryClicked() {
        if (SdxlModelLoader.isModelAvailable()) {
            // Model is now available, go back to main activity
            setResult(RESULT_OK)
            finish()
        } else {
            // Still missing, refresh the UI to show current state
            setupUI()
        }
    }
    
    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        // Prevent going back without model
        finish()
    }
}
