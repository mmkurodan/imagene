package com.micklab.imagene

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.SessionOptions
import ai.onnxruntime.SessionOptions.OptLevel
import ai.onnxruntime.SessionOptions.ExecutionMode
import android.util.Log
import java.nio.FloatBuffer

/**
 * Example class demonstrating how to load and run UNet model with ONNX Runtime.
 * Uses createSession() to load from external storage path.
 */
class UNetSessionExample : AutoCloseable {
    
    companion object {
        private const val TAG = "UNetSessionExample"
    }
    
    private var ortEnv: OrtEnvironment? = null
    private var unetSession: OrtSession? = null
    
    /**
     * Initialize ONNX Runtime environment and load UNet model.
     * @throws Exception if model loading fails
     */
    fun initialize() {
        Log.i(TAG, "Initializing ONNX Runtime environment...")
        
        // Get ONNX Runtime environment
        ortEnv = OrtEnvironment.getEnvironment()
        
        // Configure session options
        val sessionOptions = SessionOptions().apply {
            // Set optimization level for best performance
            setOptimizationLevel(OptLevel.ALL_OPT)
            
            // Try to enable NNAPI for hardware acceleration (Android)
            try {
                addNnapi()
                Log.i(TAG, "NNAPI execution provider enabled")
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI not available, using CPU: ${e.message}")
            }
            
            // Optional: Set number of threads for CPU execution
            setIntraOpNumThreads(4)
        }
        
        // Get UNet model path from external storage
        val unetModelPath = SdxlModelLoader.getOnnxModelPath("unet")
        Log.i(TAG, "Loading UNet model from: $unetModelPath")
        
        // Create session using file path
        // This is the key method: createSession() with file path string
        unetSession = ortEnv!!.createSession(unetModelPath, sessionOptions)
        
        Log.i(TAG, "UNet model loaded successfully")
        logModelInfo()
    }
    
    /**
     * Log model input/output information for debugging.
     */
    private fun logModelInfo() {
        unetSession?.let { session ->
            Log.i(TAG, "=== UNet Model Info ===")
            
            // Log input names and shapes
            Log.i(TAG, "Inputs:")
            for ((name, info) in session.inputInfo) {
                Log.i(TAG, "  $name: ${info.info}")
            }
            
            // Log output names and shapes
            Log.i(TAG, "Outputs:")
            for ((name, info) in session.outputInfo) {
                Log.i(TAG, "  $name: ${info.info}")
            }
        }
    }
    
    /**
     * Run UNet inference.
     * This is a simplified example - actual SDXL UNet has complex inputs.
     * 
     * @param sample Latent sample tensor [batch, channels, height, width]
     * @param timestep Current timestep value
     * @param encoderHiddenStates Text encoder output [batch, seq_len, hidden_dim]
     * @return Output latent tensor
     */
    fun runInference(
        sample: FloatArray,
        timestep: Long,
        encoderHiddenStates: FloatArray
    ): FloatArray {
        val env = ortEnv ?: throw IllegalStateException("OrtEnvironment not initialized")
        val session = unetSession ?: throw IllegalStateException("UNet session not initialized")
        
        // Example dimensions (actual SDXL uses different sizes)
        val batchSize = 1
        val channels = 4
        val height = 128  // Latent height (1024 / 8)
        val width = 128   // Latent width (1024 / 8)
        val seqLen = 77
        val hiddenDim = 2048
        
        // Create input tensors
        val sampleTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(sample),
            longArrayOf(batchSize.toLong(), channels.toLong(), height.toLong(), width.toLong())
        )
        
        val timestepTensor = OnnxTensor.createTensor(
            env,
            longArrayOf(timestep)
        )
        
        val encoderTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(encoderHiddenStates),
            longArrayOf(batchSize.toLong(), seqLen.toLong(), hiddenDim.toLong())
        )
        
        // Prepare inputs map
        val inputs = mapOf(
            "sample" to sampleTensor,
            "timestep" to timestepTensor,
            "encoder_hidden_states" to encoderTensor
        )
        
        // Run inference
        val results = session.run(inputs)
        
        // Get output
        val outputTensor = results[0] as OnnxTensor
        val output = outputTensor.floatBuffer.array()
        
        // Clean up
        sampleTensor.close()
        timestepTensor.close()
        encoderTensor.close()
        results.close()
        
        return output
    }
    
    /**
     * Alternative: Load model with custom session options for memory optimization.
     */
    fun initializeWithMemoryOptimization() {
        ortEnv = OrtEnvironment.getEnvironment()
        
        val sessionOptions = SessionOptions().apply {
            // Enable memory pattern optimization
            setMemoryPatternOptimization(true)
            
            // Set execution mode to sequential for lower memory
            setExecutionMode(ExecutionMode.SEQUENTIAL)
            
            // Optimization level
            setOptimizationLevel(OptLevel.ALL_OPT)
        }
        
        val unetModelPath = SdxlModelLoader.getOnnxModelPath("unet")
        unetSession = ortEnv!!.createSession(unetModelPath, sessionOptions)
    }
    
    /**
     * Alternative: Load all SDXL components.
     * Shows how to load multiple models from external storage.
     */
    fun loadAllComponents(): Map<String, OrtSession> {
        val env = ortEnv ?: OrtEnvironment.getEnvironment().also { ortEnv = it }
        val sessionOptions = SessionOptions().apply {
            setOptimizationLevel(OptLevel.ALL_OPT)
        }
        
        val components = listOf(
            "unet",
            "text_encoder",
            "text_encoder_2", 
            "vae_decoder"
        )
        
        val sessions = mutableMapOf<String, OrtSession>()
        
        for (component in components) {
            val modelPath = SdxlModelLoader.getOnnxModelPath(component)
            Log.i(TAG, "Loading $component from: $modelPath")
            
            val session = env.createSession(modelPath, sessionOptions)
            sessions[component] = session
            
            Log.i(TAG, "$component loaded successfully")
        }
        
        return sessions
    }
    
    override fun close() {
        unetSession?.close()
        unetSession = null
        
        ortEnv?.close()
        ortEnv = null
        
        Log.i(TAG, "UNet session closed")
    }
}
