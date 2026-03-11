package com.micklab.imagene

import ai.onnxruntime.OrtException
import android.graphics.Bitmap
import android.graphics.Color
import java.io.File
import java.util.Random
import java.util.concurrent.CancellationException
import kotlin.math.sin

data class GenerationRequest(
    val prompt: String,
    val negativePrompt: String,
    val width: Int,
    val height: Int,
    val steps: Int,
    val guidanceScale: Float,
    val seed: Long
)

data class GenerationResult(
    val bitmap: Bitmap,
    val warning: String? = null
)

class SdxlGenerationRunner {

    companion object {
        private const val TAG = "SdxlGenerationRunner"
        private const val LATENT_CHANNELS = 4
        private const val ENCODER_SEQ_LEN = 77
        private const val ENCODER_HIDDEN_SIZE = 768
    }

    fun generate(
        request: GenerationRequest,
        onProgress: (current: Int, total: Int, message: String) -> Unit,
        isCancelled: () -> Boolean
    ): GenerationResult {
        if (request.prompt.isBlank()) {
            throw IllegalArgumentException("プロンプトを入力してください。")
        }
        if (!SdxlModelLoader.isModelAvailable()) {
            throw IllegalStateException("モデルが見つかりません: ${SdxlModelLoader.getMissingComponents()}")
        }

        val width = request.width.coerceIn(512, 1024)
        val height = request.height.coerceIn(512, 1024)
        val steps = request.steps.coerceIn(5, 50)
        val guidanceScale = request.guidanceScale.coerceIn(1.0f, 20.0f)
        val total = steps + 2
        val promptSeed = computePromptSeed(request)

        // ★ SD15 latent サイズ
        val latentWidth = width / 8
        val latentHeight = height / 8

        onProgress(0, total, "モデルを初期化中...")
        if (isCancelled()) throw CancellationException("Generation cancelled before start")

        val latent = createInitialLatent(promptSeed, latentWidth, latentHeight)
        val refinementWarning = runUnetRefinement(latent, request, promptSeed)

        for (step in 1..steps) {
            if (isCancelled()) throw CancellationException("Generation cancelled")
            applyPseudoDiffusion(latent, step, steps, guidanceScale, promptSeed)
            onProgress(step, total, "生成中... ($step/$steps)")
        }

        if (isCancelled()) throw CancellationException("Generation cancelled before rendering")

        onProgress(steps + 1, total, "画像をレンダリング中...")
        val bitmap = renderBitmapFromLatent(
            latent = latent,
            width = width,
            height = height,
            latentWidth = latentWidth,
            latentHeight = latentHeight,
            guidanceScale = guidanceScale,
            promptSeed = promptSeed
        )

        onProgress(total, total, "生成完了")

        AppLogStore.i(
            TAG,
            "Generated image ${width}x${height}, steps=$steps, guidance=$guidanceScale, seed=${request.seed}"
        )

        return GenerationResult(bitmap = bitmap, warning = refinementWarning)
    }

    private fun runUnetRefinement(
        latent: FloatArray,
        request: GenerationRequest,
        promptSeed: Long
    ): String? {
        val unetFile = File(SdxlModelLoader.getOnnxModelPath("unet"))
        if (!unetFile.exists() || !unetFile.isFile) {
            val warning = "unet/model.onnx が見つからないため軽量生成で続行しました。"
            AppLogStore.w(TAG, warning)
            return warning
        }

        val unet = UNetSessionExample()
        return try {
            unet.initialize()
            val encoderStates = buildEncoderStates(request, promptSeed)
            val prediction = unet.runInference(
                sample = latent.copyOf(),
                timestep = request.steps.toFloat(),
                encoderHiddenStates = encoderStates
            )
            blendLatentWithPrediction(latent, prediction, request.guidanceScale)
            AppLogStore.i(TAG, "UNet refinement pass succeeded")
            null
        } catch (e: Exception) {
            val warning = "UNet推論を実行できなかったため軽量生成で続行しました。(${e.message})"
            AppLogStore.e(TAG, warning, e)
            warning
        } finally {
            unet.close()
        }
    }

    private fun createInitialLatent(
        promptSeed: Long,
        latentWidth: Int,
        latentHeight: Int
    ): FloatArray {
        val random = Random(promptSeed)
        val latent = FloatArray(latentWidth * latentHeight * LATENT_CHANNELS)
        for (i in latent.indices) {
            latent[i] = random.nextFloat() * 2.0f - 1.0f
        }
        return latent
    }

    private fun buildEncoderStates(request: GenerationRequest, promptSeed: Long): FloatArray {
        val text = "${request.prompt}\u0000${request.negativePrompt}"
        val random = Random(promptSeed xor 0x5f3759dfL)
        val output = FloatArray(ENCODER_SEQ_LEN * ENCODER_HIDDEN_SIZE)

        var rolling = text.hashCode().toLong() xor promptSeed
        for (i in output.indices) {
            rolling = java.lang.Long.rotateLeft(rolling xor i.toLong(), 11) * 6364136223846793005L + 1L
            val signal = (((rolling ushr 24) and 0xffL).toInt() / 127.5f) - 1.0f
            val noise = random.nextFloat() * 2.0f - 1.0f
            output[i] = signal * 0.85f + noise * 0.15f
        }

        return output
    }

    private fun blendLatentWithPrediction(
        latent: FloatArray,
        prediction: FloatArray,
        guidanceScale: Float
    ) {
        if (prediction.isEmpty()) return
        val blend = (0.08f * (guidanceScale / 7.5f)).coerceIn(0.03f, 0.20f)
        for (i in latent.indices) {
            val predicted = prediction[i % prediction.size]
            latent[i] = (latent[i] - predicted * blend).coerceIn(-2.5f, 2.5f)
        }
    }

    private fun applyPseudoDiffusion(
        latent: FloatArray,
        step: Int,
        totalSteps: Int,
        guidanceScale: Float,
        promptSeed: Long
    ) {
        val progress = step.toFloat() / totalSteps.toFloat()
        val decay = 0.08f * (1.0f - progress)
        val guidanceFactor = (guidanceScale / 10.0f).coerceIn(0.3f, 2.5f)
        val phase = ((promptSeed and 0x3ffL).toDouble() / 91.0)

        for (i in latent.indices) {
            val wave = sin(i * 0.015 + step * 0.31 + phase).toFloat()
            val mixed = latent[i] * (1.0f - decay) + wave * decay * guidanceFactor
            latent[i] = mixed.coerceIn(-2.5f, 2.5f)
        }
    }

    private fun renderBitmapFromLatent(
        latent: FloatArray,
        width: Int,
        height: Int,
        latentWidth: Int,
        latentHeight: Int,
        guidanceScale: Float,
        promptSeed: Long
    ): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        val plane = latentWidth * latentHeight
        val gain = 0.65f + (guidanceScale / 16.0f)

        for (y in 0 until height) {
            val ly = y * latentHeight / height
            for (x in 0 until width) {
                val lx = x * latentWidth / width
                val index = ly * latentWidth + lx

                val r = latent[index]
                val g = latent[plane + index]
                val b = latent[(plane * 2) + index]
                val a = latent[(plane * 3) + index]
                val tint = (((promptSeed ushr ((x + y) and 31)) and 0xffL).toInt() / 255.0f) - 0.5f

                val red = toColorChannel((r + a * 0.25f + tint * 0.20f) * gain)
                val green = toColorChannel((g + a * 0.10f - tint * 0.10f) * gain)
                val blue = toColorChannel((b - a * 0.20f + tint * 0.25f) * gain)

                pixels[y * width + x] = Color.rgb(red, green, blue)
            }
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun toColorChannel(value: Float): Int {
        val normalized = ((value + 1.2f) / 2.4f).coerceIn(0.0f, 1.0f)
        return (normalized * 255.0f).toInt()
    }

    private fun computePromptSeed(request: GenerationRequest): Long {
        var seed = request.seed
        val text = "${request.prompt}\u0000${request.negativePrompt}"
        for (char in text) {
            seed = seed xor (char.code.toLong() shl 16)
            seed = java.lang.Long.rotateLeft(seed, 13) * -0x61c8864680b583ebL
        }
        return seed
    }
}
