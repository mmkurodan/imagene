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
        private const val CPU_THREADS = 2

        private val ortEnvironment: OrtEnvironment by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            OrtEnvironment.getEnvironment()
        }
    }

    private var unetSession: OrtSession? = null
    private var activeSessionOptions: SessionOptions? = null

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

        unetSession = createCpuSession(unetModelPath)
        Log.i(TAG, "UNet model loaded successfully")
        logModelInfo()
    }

    private fun createCpuSession(unetModelPath: String): OrtSession {
        val sessionOptions = SessionOptions().apply {
            setExecutionMode(ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OptLevel.BASIC_OPT)
            setIntraOpNumThreads(CPU_THREADS)
            setMemoryPatternOptimization(true)
        }

        return try {
            ortEnvironment.createSession(unetModelPath, sessionOptions).also {
                activeSessionOptions = sessionOptions
                AppLogStore.i(TAG, "UNet session created with CPU only")
            }
        } catch (e: Exception) {
            sessionOptions.close()
            throw e
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

        val sampleBuffer = java.nio.ByteBuffer.allocateDirect(sample.size * 4)
            .order(java.nio.ByteOrder.nativeOrder())
        sampleBuffer.asFloatBuffer().put(sample)
        sampleBuffer.rewind()

        val sampleTensor = OnnxTensor.createTensor(
            ortEnvironment,
            sampleBuffer,
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

    private fun closeSession() {
        unetSession?.close()
        unetSession = null
        activeSessionOptions?.close()
        activeSessionOptions = null
    }

    override fun close() {
        closeSession()
        Log.i(TAG, "UNet session closed")
    }
}
