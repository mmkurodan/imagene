package com.micklab.imagene

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat
import java.io.File

/**
 * SDXL Model Loader for external storage.
 * Handles model path resolution and existence checking.
 */
object SdxlModelLoader {

    private const val TAG = "SdxlModelLoader"
    
    /** Base path for SDXL models */
    private const val SDXL_BASE_PATH = "/storage/emulated/0/Download/sdxl"
    
    /** Required model directories */
    private val REQUIRED_DIRECTORIES = listOf(
        "unet",
        "text_encoder",
        "text_encoder_2",
        "vae_decoder",
        "tokenizer",
        "tokenizer_2",
        "scheduler"
    )
    
    /** Required model files */
    private const val MODEL_INDEX_FILE = "model_index.json"
    
    /**
     * Get the base path for SDXL models.
     * @return Absolute path to the SDXL model directory
     */
    fun getModelBasePath(): String = SDXL_BASE_PATH
    
    /**
     * Get the path to a specific model component.
     * @param component Component name (e.g., "unet", "text_encoder")
     * @return Absolute path to the component directory
     */
    fun getComponentPath(component: String): String {
        return "$SDXL_BASE_PATH/$component"
    }
    
    /**
     * Get the path to the ONNX model file within a component.
     * @param component Component name (e.g., "unet", "vae_decoder")
     * @return Absolute path to the model.onnx file
     */
    fun getOnnxModelPath(component: String): String {
        return "$SDXL_BASE_PATH/$component/model.onnx"
    }
    
    /**
     * Get the path to model_index.json
     * @return Absolute path to model_index.json
     */
    fun getModelIndexPath(): String {
        return "$SDXL_BASE_PATH/$MODEL_INDEX_FILE"
    }
    
    /**
     * Check if all required model files and directories exist.
     * @return true if all models are present, false otherwise
     */
    fun isModelAvailable(): Boolean {
        val baseDir = File(SDXL_BASE_PATH)
        
        // Check if base directory exists
        if (!baseDir.exists() || !baseDir.isDirectory) {
            AppLogStore.w(TAG, "Base model directory is missing: $SDXL_BASE_PATH")
            return false
        }
        
        // Check for model_index.json
        val modelIndex = File(baseDir, MODEL_INDEX_FILE)
        if (!modelIndex.exists() || !modelIndex.isFile) {
            AppLogStore.w(TAG, "Missing required file: ${modelIndex.absolutePath}")
            return false
        }
        
        // Check all required directories and model.onnx files exist
        for (dirName in REQUIRED_DIRECTORIES) {
            val dir = File(baseDir, dirName)
            if (!dir.exists() || !dir.isDirectory) {
                AppLogStore.w(TAG, "Missing required directory: ${dir.absolutePath}")
                return false
            }
            val modelFile = File(dir, "model.onnx")
            if (!modelFile.exists() || !modelFile.isFile) {
                AppLogStore.w(TAG, "Missing model file: ${modelFile.absolutePath}")
                return false
            }
        }
        
        AppLogStore.i(TAG, "All required SDXL model components are available")
        return true
    }
    
    /**
     * Get detailed information about what's missing.
     * @return List of missing components, empty if all present
     */
    fun getMissingComponents(): List<String> {
        val missing = mutableListOf<String>()
        val baseDir = File(SDXL_BASE_PATH)
        
        if (!baseDir.exists()) {
            missing.add("Base directory: $SDXL_BASE_PATH")
            return missing
        }
        
        val modelIndex = File(baseDir, MODEL_INDEX_FILE)
        if (!modelIndex.exists()) {
            missing.add(MODEL_INDEX_FILE)
        }
        
        for (dirName in REQUIRED_DIRECTORIES) {
            val dir = File(baseDir, dirName)
            if (!dir.exists() || !dir.isDirectory) {
                missing.add("$dirName/")
            } else {
                val modelFile = File(dir, "model.onnx")
                if (!modelFile.exists() || !modelFile.isFile) {
                    missing.add("$dirName/model.onnx")
                }
            }
        }
        
        return missing
    }
    
    /**
     * Check if storage permission is needed and granted.
     * Android 13+ doesn't need READ_EXTERNAL_STORAGE for Download folder.
     * @param context Application context
     * @return true if permission is granted or not needed
     */
    fun hasStoragePermission(context: Context): Boolean {
        val granted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ doesn't need READ_EXTERNAL_STORAGE for media/download files
            true
        } else {
            // Android 12 and below need READ_EXTERNAL_STORAGE
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
        }
        AppLogStore.d(TAG, "Storage permission status: granted=$granted, sdk=${Build.VERSION.SDK_INT}")
        return granted
    }
    
    /**
     * Check if storage permission is needed for this Android version.
     * @return true if permission request is needed
     */
    fun needsStoragePermission(): Boolean {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU
    }
}
