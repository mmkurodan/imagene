package com.micklab.imagene

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

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
        encoderHiddenStates: FloatArray,
        latentWidth: Int,
        latentHeight: Int
    ): FloatArray {
        val session = unetSession ?: throw IllegalStateException("UNet session not initialized")

        val batchSize = 1
        val channels = 4
        val seqLen = 77
        val hiddenDim = 768

        val expectedSampleSize = batchSize * channels * latentWidth * latentHeight
        require(sample.size == expectedSampleSize) {
            "sample size (${sample.size}) does not match shape [1, $channels, $latentHeight, $latentWidth] ($expectedSampleSize)"
        }

        val expectedEncoderSize = batchSize * seqLen * hiddenDim
        require(encoderHiddenStates.size == expectedEncoderSize) {
            "encoder_hidden_states size (${encoderHiddenStates.size}) does not match shape [1, $seqLen, $hiddenDim] ($expectedEncoderSize)"
        }

        AppLogStore.i(
            TAG,
            "Preparing UNet tensors (sample=${sample.size}, encoder=${encoderHiddenStates.size}, shape=[1,$channels,$latentHeight,$latentWidth])"
        )
        val sampleData = ByteBuffer.allocateDirect(sample.size * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(sample)
                rewind()
            }
        val timestepData = ByteBuffer.allocateDirect(java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(timestep)
                rewind()
            }
        val encoderData = ByteBuffer.allocateDirect(encoderHiddenStates.size * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(encoderHiddenStates)
                rewind()
            }

        val sampleTensor = OnnxTensor.createTensor(
            ortEnvironment,
            sampleData,
            longArrayOf(batchSize.toLong(), channels.toLong(), latentHeight.toLong(), latentWidth.toLong())
        )

        val timestepTensor = OnnxTensor.createTensor(
            ortEnvironment,
            timestepData,
            longArrayOf(1)
        )

        val encoderTensor = OnnxTensor.createTensor(
            ortEnvironment,
            encoderData,
            longArrayOf(batchSize.toLong(), seqLen.toLong(), hiddenDim.toLong())
        )

        return try {
            AppLogStore.i(TAG, "Running UNet session")
            val inputs = mapOf(
                "sample" to sampleTensor,
                "timestep" to timestepTensor,
                "encoder_hidden_states" to encoderTensor
            )
            val results = session.run(inputs)
            try {
                val outputTensor = results[0] as OnnxTensor
                val buf = outputTensor.floatBuffer
                FloatArray(buf.remaining()).also {
                    buf.get(it)
                    AppLogStore.i(TAG, "UNet session completed (output=${it.size})")
                }
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
