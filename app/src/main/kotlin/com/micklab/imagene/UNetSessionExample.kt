package com.micklab.imagene

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt

class UNetSessionExample : AutoCloseable {

    companion object {
        private const val TAG = "UNetSessionExample"
        private const val NNAPI_THREADS = 4
        private const val CPU_FALLBACK_THREADS = 2

        private val ortEnvironment: OrtEnvironment by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            OrtEnvironment.getEnvironment()
        }
    }

    private var unetSession: OrtSession? = null
    private var activeSessionOptions: SessionOptions? = null
    private val auxiliarySessions = linkedMapOf<String, OrtSession>()
    private val auxiliarySessionOptions = linkedMapOf<String, SessionOptions>()

    fun initialize() {
        closeSession()
        val unetModelPath = SdxlModelLoader.getOnnxModelPath("unet")
        val modelFile = File(unetModelPath)

        if (!modelFile.exists() || !modelFile.isFile) {
            val message = "UNet model not found at path: $unetModelPath"
            AppLogStore.w(TAG, message)
            throw IllegalStateException(message)
        }
        if (modelFile.length() <= 0L) {
            val message = "UNet model file is empty: $unetModelPath"
            AppLogStore.w(TAG, message)
            throw IllegalStateException(message)
        }

        Log.i(TAG, "Initializing ONNX Runtime environment...")
        AppLogStore.i(TAG, "Loading UNet model from: $unetModelPath (${modelFile.length()} bytes)")

        unetSession = createSessionWithFallback(unetModelPath)
        Log.i(TAG, "UNet model loaded successfully")
        logModelInfo()
    }

    private fun createSessionWithFallback(unetModelPath: String): OrtSession {
        val failures = mutableListOf<String>()

        createSession(unetModelPath, "NNAPI acceleration", failures) {
            setOptimizationLevel(OptLevel.ALL_OPT)
            setIntraOpNumThreads(NNAPI_THREADS)
            addNnapi()
        }?.let { return it }

        createSession(unetModelPath, "CPU fallback", failures) {
            setExecutionMode(ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OptLevel.BASIC_OPT)
            setIntraOpNumThreads(CPU_FALLBACK_THREADS)
            setMemoryPatternOptimization(true)
        }?.let { return it }

        throw IllegalStateException(
            "UNet session initialization failed: ${failures.joinToString(" | ")}"
        )
    }

    private fun createSession(
        unetModelPath: String,
        label: String,
        failures: MutableList<String>,
        configure: SessionOptions.() -> Unit
    ): OrtSession? {
        val sessionOptions = SessionOptions()
        return try {
            sessionOptions.configure()
            ortEnvironment.createSession(unetModelPath, sessionOptions).also {
                activeSessionOptions = sessionOptions
                AppLogStore.i(TAG, "UNet session created with $label")
            }
        } catch (e: Exception) {
            val detail = "$label failed: ${e.message ?: e::class.java.simpleName}"
            failures.add(detail)
            AppLogStore.e(TAG, "Failed to create UNet session with $label", e)
            sessionOptions.close()
            null
        }
    }

    private fun logModelInfo() {
        unetSession?.let { session ->
            Log.i(TAG, "=== UNet Model Info ===")
            Log.i(TAG, "Inputs:")
            for ((name, info) in session.inputInfo) {
                Log.i(TAG, "  $name: ${info.info}")
            }
            Log.i(TAG, "Outputs:")
            for ((name, info) in session.outputInfo) {
                Log.i(TAG, "  $name: ${info.info}")
            }
        }
    }

    /**
     * SD15 UNet 推論
     * sample: [1,4,H/8,W/8]
     * timestep: float32
     * encoderHiddenStates: [1,77,768]
     */
    fun runInference(
        sample: FloatArray,
        timestep: Float,
        encoderHiddenStates: FloatArray
    ): FloatArray {
        val session = unetSession ?: throw IllegalStateException("UNet session not initialized")

        val batchSize = 1
        val channels = 4

        // ★ sample の長さから latent の高さ・幅を自動推定
        val spatial = sample.size / (batchSize * channels)
        val side = sqrt(spatial.toDouble()).toInt()
        val height = side
        val width = side

        val seqLen = 77
        val hiddenDim = 768

        val sampleTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(sample),
            longArrayOf(batchSize.toLong(), channels.toLong(), height.toLong(), width.toLong())
        )

        val timestepTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(floatArrayOf(timestep)),
            longArrayOf(1)
        )

        val encoderTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(encoderHiddenStates),
            longArrayOf(batchSize.toLong(), seqLen.toLong(), hiddenDim.toLong())
        )

        return try {
            val inputs = mapOf(
                "sample" to sampleTensor,
                "timestep" to timestepTensor,
                "encoder_hidden_states" to encoderTensor
            )
            val results = session.run(inputs)
            try {
                val outputTensor = results[0] as OnnxTensor
                val buf = outputTensor.floatBuffer
                FloatArray(buf.remaining()).also { buf.get(it) }
            } finally {
                results.close()
            }
        } finally {
            sampleTensor.close()
            timestepTensor.close()
            encoderTensor.close()
        }
    }

    fun initializeWithMemoryOptimization() {
        closeSession()
        val sessionOptions = SessionOptions().apply {
            setMemoryPatternOptimization(true)
            setExecutionMode(ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OptLevel.ALL_OPT)
        }

        val unetModelPath = SdxlModelLoader.getOnnxModelPath("unet")
        try {
            unetSession = ortEnvironment.createSession(unetModelPath, sessionOptions)
            activeSessionOptions = sessionOptions
        } catch (e: Exception) {
            sessionOptions.close()
            throw e
        }
    }

    fun loadAllComponents(): Map<String, OrtSession> {
        val env = ortEnvironment
        val components = listOf(
            "unet",
            "text_encoder",
            "vae_decoder"
        )

        val sessions = mutableMapOf<String, OrtSession>()
        closeAuxiliarySessions()

        try {
            for (component in components) {
                val modelPath = SdxlModelLoader.getOnnxModelPath(component)
                Log.i(TAG, "Loading $component from: $modelPath")

                val componentOptions = SessionOptions().apply {
                    setOptimizationLevel(OptLevel.ALL_OPT)
                }
                try {
                    val session = env.createSession(modelPath, componentOptions)
                    auxiliarySessions[component] = session
                    auxiliarySessionOptions[component] = componentOptions
                    sessions[component] = session
                    Log.i(TAG, "$component loaded successfully")
                } catch (e: Exception) {
                    componentOptions.close()
                    throw e
                }
            }
        } catch (e: Exception) {
            closeAuxiliarySessions()
            throw e
        }

        return sessions
    }

    private fun closeAuxiliarySessions() {
        auxiliarySessions.values.forEach { it.close() }
        auxiliarySessions.clear()
        auxiliarySessionOptions.values.forEach { it.close() }
        auxiliarySessionOptions.clear()
    }

    private fun closeSession() {
        unetSession?.close()
        unetSession = null
        activeSessionOptions?.close()
        activeSessionOptions = null
        closeAuxiliarySessions()
    }

    override fun close() {
        closeSession()
        Log.i(TAG, "UNet session closed")
    }
}
