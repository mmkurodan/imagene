package com.micklab.imagene

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import java.io.File
import java.util.Locale
import java.util.Random
import java.util.concurrent.CancellationException
import kotlin.math.cos
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

    private enum class SemanticSubject {
        NONE, APPLE, FLOWER, MOUNTAIN, SUN, TREE, CAT, HOUSE
    }

    private enum class SemanticBackground {
        NONE, SKY, SUNSET, OCEAN, FOREST, NIGHT
    }

    private data class PromptSemanticScene(
        val subject: SemanticSubject,
        val background: SemanticBackground,
        val subjectColor: Int
    )

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

        // ★ ここで latent サイズをログ出力
        AppLogStore.i(
            TAG,
            "latent size = ${latent.size}, latentWidth=$latentWidth, latentHeight=$latentHeight"
        )

        val refinementWarning = runUnetRefinement(latent, request, promptSeed, latentWidth, latentHeight)

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
            promptSeed = promptSeed,
            prompt = request.prompt,
            negativePrompt = request.negativePrompt
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
        promptSeed: Long,
        latentWidth: Int,
        latentHeight: Int
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
            AppLogStore.i(TAG, "Starting UNet refinement inference")
            val prediction = unet.runInference(
                sample = latent.copyOf(),
                timestep = request.steps.toFloat(),
                encoderHiddenStates = encoderStates,
                latentWidth = latentWidth,
                latentHeight = latentHeight
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
        promptSeed: Long,
        prompt: String,
        negativePrompt: String
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
        applyPromptSemanticOverlay(bitmap, prompt, negativePrompt, guidanceScale, promptSeed)
        return bitmap
    }

    private fun applyPromptSemanticOverlay(
        bitmap: Bitmap,
        prompt: String,
        negativePrompt: String,
        guidanceScale: Float,
        promptSeed: Long
    ) {
        val scene = parsePromptSemanticScene(prompt, negativePrompt)
        if (scene.subject == SemanticSubject.NONE && scene.background == SemanticBackground.NONE) {
            return
        }

        val canvas = Canvas(bitmap)
        val width = bitmap.width.toFloat()
        val height = bitmap.height.toFloat()
        val backgroundAlpha = (70 + guidanceScale * 6f).toInt().coerceIn(60, 190)
        val subjectAlpha = (120 + guidanceScale * 6f).toInt().coerceIn(120, 230)

        drawBackgroundOverlay(canvas, scene.background, width, height, backgroundAlpha, promptSeed)
        if (scene.subject != SemanticSubject.NONE) {
            drawSubjectOverlay(canvas, scene, width, height, subjectAlpha, promptSeed)
        }
    }

    private fun parsePromptSemanticScene(prompt: String, negativePrompt: String): PromptSemanticScene {
        val positiveText = prompt.lowercase(Locale.ROOT)
        val negativeText = negativePrompt.lowercase(Locale.ROOT)
        val subject = resolveSubject(positiveText, negativeText)
        val background = resolveBackground(positiveText, subject)
        val subjectColor = resolveSubjectColor(positiveText, subject)
        AppLogStore.i(TAG, "Semantic scene: subject=$subject, background=$background")
        return PromptSemanticScene(subject = subject, background = background, subjectColor = subjectColor)
    }

    private fun resolveSubject(positiveText: String, negativeText: String): SemanticSubject {
        fun present(keywords: List<String>): Boolean =
            containsAny(positiveText, keywords) && !containsAny(negativeText, keywords)

        return when {
            present(listOf("apple", "りんご", "林檎")) -> SemanticSubject.APPLE
            present(listOf("flower", "rose", "sunflower", "花", "薔薇", "ひまわり")) -> SemanticSubject.FLOWER
            present(listOf("mountain", "alps", "山", "富士")) -> SemanticSubject.MOUNTAIN
            present(listOf("sun", "sunset", "sunrise", "太陽", "夕日", "朝日")) -> SemanticSubject.SUN
            present(listOf("tree", "forest", "木", "森")) -> SemanticSubject.TREE
            present(listOf("cat", "kitten", "猫", "ねこ")) -> SemanticSubject.CAT
            present(listOf("house", "home", "building", "家", "建物")) -> SemanticSubject.HOUSE
            else -> SemanticSubject.NONE
        }
    }

    private fun resolveBackground(positiveText: String, subject: SemanticSubject): SemanticBackground {
        return when {
            containsAny(positiveText, listOf("sunset", "dusk", "evening", "夕焼け", "夕日")) -> SemanticBackground.SUNSET
            containsAny(positiveText, listOf("ocean", "sea", "beach", "海", "浜")) -> SemanticBackground.OCEAN
            containsAny(positiveText, listOf("night", "space", "star", "moon", "夜", "宇宙", "月")) -> SemanticBackground.NIGHT
            containsAny(positiveText, listOf("forest", "woods", "nature", "森", "木")) -> SemanticBackground.FOREST
            subject == SemanticSubject.MOUNTAIN || subject == SemanticSubject.TREE -> SemanticBackground.FOREST
            subject != SemanticSubject.NONE -> SemanticBackground.SKY
            else -> SemanticBackground.NONE
        }
    }

    private fun resolveSubjectColor(positiveText: String, subject: SemanticSubject): Int {
        val explicitColor = when {
            containsAny(positiveText, listOf("red", "crimson", "scarlet", "赤")) -> Color.rgb(225, 59, 59)
            containsAny(positiveText, listOf("blue", "azure", "青")) -> Color.rgb(64, 140, 255)
            containsAny(positiveText, listOf("green", "emerald", "緑")) -> Color.rgb(65, 173, 93)
            containsAny(positiveText, listOf("yellow", "gold", "黄")) -> Color.rgb(246, 204, 70)
            containsAny(positiveText, listOf("orange", "橙")) -> Color.rgb(243, 146, 69)
            containsAny(positiveText, listOf("purple", "violet", "紫")) -> Color.rgb(171, 112, 243)
            containsAny(positiveText, listOf("pink", "桃")) -> Color.rgb(242, 131, 182)
            containsAny(positiveText, listOf("white", "白")) -> Color.rgb(240, 240, 240)
            containsAny(positiveText, listOf("black", "黒")) -> Color.rgb(35, 35, 35)
            containsAny(positiveText, listOf("brown", "茶")) -> Color.rgb(153, 98, 61)
            else -> null
        }
        if (explicitColor != null) return explicitColor

        return when (subject) {
            SemanticSubject.APPLE -> Color.rgb(220, 65, 58)
            SemanticSubject.FLOWER -> Color.rgb(235, 128, 182)
            SemanticSubject.MOUNTAIN -> Color.rgb(120, 136, 156)
            SemanticSubject.SUN -> Color.rgb(248, 196, 62)
            SemanticSubject.TREE -> Color.rgb(76, 170, 81)
            SemanticSubject.CAT -> Color.rgb(188, 150, 122)
            SemanticSubject.HOUSE -> Color.rgb(206, 175, 142)
            SemanticSubject.NONE -> Color.rgb(220, 220, 220)
        }
    }

    private fun drawBackgroundOverlay(
        canvas: Canvas,
        background: SemanticBackground,
        width: Float,
        height: Float,
        alpha: Int,
        promptSeed: Long
    ) {
        if (background == SemanticBackground.NONE) return
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
        val horizon = height * 0.62f

        when (background) {
            SemanticBackground.SKY -> {
                paint.color = withAlpha(Color.rgb(113, 184, 255), alpha)
                canvas.drawRect(0f, 0f, width, height, paint)
            }
            SemanticBackground.SUNSET -> {
                paint.color = withAlpha(Color.rgb(247, 154, 91), alpha)
                canvas.drawRect(0f, 0f, width, horizon, paint)
                paint.color = withAlpha(Color.rgb(120, 89, 170), alpha)
                canvas.drawRect(0f, horizon, width, height, paint)
            }
            SemanticBackground.OCEAN -> {
                paint.color = withAlpha(Color.rgb(110, 189, 247), alpha)
                canvas.drawRect(0f, 0f, width, horizon, paint)
                paint.color = withAlpha(Color.rgb(45, 113, 187), alpha)
                canvas.drawRect(0f, horizon, width, height, paint)
            }
            SemanticBackground.FOREST -> {
                paint.color = withAlpha(Color.rgb(127, 186, 255), alpha)
                canvas.drawRect(0f, 0f, width, horizon, paint)
                paint.color = withAlpha(Color.rgb(66, 136, 77), alpha)
                canvas.drawRect(0f, horizon, width, height, paint)
            }
            SemanticBackground.NIGHT -> {
                paint.color = withAlpha(Color.rgb(26, 40, 92), alpha)
                canvas.drawRect(0f, 0f, width, height, paint)
                paint.color = withAlpha(Color.rgb(242, 242, 204), (alpha + 20).coerceIn(0, 255))
                val starCount = 18
                for (i in 0 until starCount) {
                    val sx = (((promptSeed ushr ((i % 8) * 8)) and 0xffL).toInt() / 255f) * width
                    val sy = (((promptSeed ushr (((i + 3) % 8) * 8)) and 0xffL).toInt() / 255f) * (height * 0.5f)
                    canvas.drawCircle(sx, sy, height * 0.005f, paint)
                }
            }
            SemanticBackground.NONE -> Unit
        }
    }

    private fun drawSubjectOverlay(
        canvas: Canvas,
        scene: PromptSemanticScene,
        width: Float,
        height: Float,
        alpha: Int,
        promptSeed: Long
    ) {
        val centerX = width * (0.50f + ((((promptSeed ushr 8) and 0x3fL).toInt() - 32) / 380f))
        val centerY = height * (0.62f + ((((promptSeed ushr 16) and 0x1fL).toInt() - 16) / 620f))

        when (scene.subject) {
            SemanticSubject.APPLE -> drawApple(canvas, centerX, centerY, width, height, scene.subjectColor, alpha)
            SemanticSubject.FLOWER -> drawFlower(canvas, centerX, centerY, width, height, scene.subjectColor, alpha, promptSeed)
            SemanticSubject.MOUNTAIN -> drawMountain(canvas, width, height, scene.subjectColor, alpha)
            SemanticSubject.SUN -> drawSun(canvas, centerX, centerY, width, height, scene.subjectColor, alpha)
            SemanticSubject.TREE -> drawTree(canvas, centerX, centerY, width, height, scene.subjectColor, alpha)
            SemanticSubject.CAT -> drawCat(canvas, centerX, centerY, width, height, scene.subjectColor, alpha)
            SemanticSubject.HOUSE -> drawHouse(canvas, centerX, centerY, width, height, scene.subjectColor, alpha)
            SemanticSubject.NONE -> Unit
        }
    }

    private fun drawApple(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        color: Int,
        alpha: Int
    ) {
        val bodyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(color, alpha)
        }
        val radius = minOf(width, height) * 0.17f
        canvas.drawOval(RectF(centerX - radius, centerY - radius, centerX + radius * 0.20f, centerY + radius), bodyPaint)
        canvas.drawOval(RectF(centerX - radius * 0.20f, centerY - radius, centerX + radius, centerY + radius), bodyPaint)

        val stemPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(Color.rgb(110, 68, 40), (alpha + 20).coerceIn(0, 255))
        }
        canvas.drawRoundRect(
            RectF(centerX - radius * 0.06f, centerY - radius * 1.18f, centerX + radius * 0.06f, centerY - radius * 0.60f),
            radius * 0.06f,
            radius * 0.06f,
            stemPaint
        )

        val leafPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(Color.rgb(88, 166, 73), (alpha + 10).coerceIn(0, 255))
        }
        canvas.drawOval(
            RectF(centerX, centerY - radius * 1.18f, centerX + radius * 0.60f, centerY - radius * 0.75f),
            leafPaint
        )
    }

    private fun drawFlower(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        color: Int,
        alpha: Int,
        promptSeed: Long
    ) {
        val petalPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(color, alpha)
        }
        val centerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(Color.rgb(244, 210, 88), (alpha + 20).coerceIn(0, 255))
        }
        val stemPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = minOf(width, height) * 0.03f
            strokeCap = Paint.Cap.ROUND
            this.color = withAlpha(Color.rgb(67, 152, 69), (alpha + 10).coerceIn(0, 255))
        }

        val petalOrbit = minOf(width, height) * 0.11f
        val petalRadius = minOf(width, height) * 0.07f
        val phase = ((promptSeed and 0xffL).toInt() / 255f) * (Math.PI.toFloat())

        for (i in 0 until 8) {
            val angle = (Math.PI * 2.0 * i / 8.0).toFloat() + phase
            val px = centerX + cos(angle) * petalOrbit
            val py = centerY + sin(angle) * petalOrbit
            canvas.drawCircle(px, py, petalRadius, petalPaint)
        }
        canvas.drawCircle(centerX, centerY, petalRadius * 0.9f, centerPaint)
        canvas.drawLine(centerX, centerY + petalRadius, centerX, height * 0.92f, stemPaint)
    }

    private fun drawMountain(
        canvas: Canvas,
        width: Float,
        height: Float,
        color: Int,
        alpha: Int
    ) {
        val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(color, alpha)
        }
        val mountain = Path().apply {
            moveTo(0f, height * 0.90f)
            lineTo(width * 0.22f, height * 0.50f)
            lineTo(width * 0.42f, height * 0.90f)
            lineTo(width * 0.58f, height * 0.45f)
            lineTo(width * 0.82f, height * 0.90f)
            lineTo(width, height * 0.90f)
            lineTo(width, height)
            lineTo(0f, height)
            close()
        }
        canvas.drawPath(mountain, fill)
    }

    private fun drawSun(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        color: Int,
        alpha: Int
    ) {
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(color, alpha)
        }
        val rayPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeWidth = minOf(width, height) * 0.02f
            this.color = withAlpha(color, (alpha - 15).coerceIn(0, 255))
        }
        val radius = minOf(width, height) * 0.15f
        canvas.drawCircle(centerX, centerY, radius, paint)
        for (i in 0 until 8) {
            val angle = (Math.PI * 2.0 * i / 8.0).toFloat()
            val sx = centerX + cos(angle) * radius * 1.25f
            val sy = centerY + sin(angle) * radius * 1.25f
            val ex = centerX + cos(angle) * radius * 1.75f
            val ey = centerY + sin(angle) * radius * 1.75f
            canvas.drawLine(sx, sy, ex, ey, rayPaint)
        }
    }

    private fun drawTree(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        color: Int,
        alpha: Int
    ) {
        val trunkPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(Color.rgb(114, 75, 45), (alpha + 20).coerceIn(0, 255))
        }
        val leafPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(color, alpha)
        }
        val trunkWidth = width * 0.09f
        val trunkTop = centerY - height * 0.06f
        val trunkBottom = centerY + height * 0.23f
        canvas.drawRoundRect(
            RectF(centerX - trunkWidth / 2f, trunkTop, centerX + trunkWidth / 2f, trunkBottom),
            trunkWidth * 0.2f,
            trunkWidth * 0.2f,
            trunkPaint
        )
        val canopyRadius = minOf(width, height) * 0.14f
        canvas.drawCircle(centerX, centerY - canopyRadius * 0.6f, canopyRadius, leafPaint)
        canvas.drawCircle(centerX - canopyRadius * 0.65f, centerY, canopyRadius * 0.75f, leafPaint)
        canvas.drawCircle(centerX + canopyRadius * 0.65f, centerY, canopyRadius * 0.75f, leafPaint)
    }

    private fun drawCat(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        color: Int,
        alpha: Int
    ) {
        val facePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(color, alpha)
        }
        val featurePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(Color.rgb(45, 45, 45), (alpha + 20).coerceIn(0, 255))
        }
        val radius = minOf(width, height) * 0.15f
        canvas.drawCircle(centerX, centerY, radius, facePaint)

        val leftEar = Path().apply {
            moveTo(centerX - radius * 0.90f, centerY - radius * 0.20f)
            lineTo(centerX - radius * 0.45f, centerY - radius * 1.20f)
            lineTo(centerX - radius * 0.10f, centerY - radius * 0.35f)
            close()
        }
        val rightEar = Path().apply {
            moveTo(centerX + radius * 0.90f, centerY - radius * 0.20f)
            lineTo(centerX + radius * 0.45f, centerY - radius * 1.20f)
            lineTo(centerX + radius * 0.10f, centerY - radius * 0.35f)
            close()
        }
        canvas.drawPath(leftEar, facePaint)
        canvas.drawPath(rightEar, facePaint)
        canvas.drawCircle(centerX - radius * 0.35f, centerY - radius * 0.1f, radius * 0.12f, featurePaint)
        canvas.drawCircle(centerX + radius * 0.35f, centerY - radius * 0.1f, radius * 0.12f, featurePaint)
    }

    private fun drawHouse(
        canvas: Canvas,
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        color: Int,
        alpha: Int
    ) {
        val bodyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(color, alpha)
        }
        val roofPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(Color.rgb(171, 74, 64), (alpha + 15).coerceIn(0, 255))
        }
        val doorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            this.color = withAlpha(Color.rgb(108, 73, 48), (alpha + 20).coerceIn(0, 255))
        }
        val houseWidth = width * 0.32f
        val houseHeight = height * 0.24f
        val left = centerX - houseWidth / 2f
        val top = centerY - houseHeight / 2f
        canvas.drawRect(left, top, left + houseWidth, top + houseHeight, bodyPaint)

        val roof = Path().apply {
            moveTo(left - houseWidth * 0.06f, top)
            lineTo(centerX, top - houseHeight * 0.60f)
            lineTo(left + houseWidth + houseWidth * 0.06f, top)
            close()
        }
        canvas.drawPath(roof, roofPaint)
        canvas.drawRect(
            centerX - houseWidth * 0.10f,
            top + houseHeight * 0.45f,
            centerX + houseWidth * 0.10f,
            top + houseHeight,
            doorPaint
        )
    }

    private fun containsAny(text: String, keywords: List<String>): Boolean {
        return keywords.any { keyword -> text.contains(keyword) }
    }

    private fun withAlpha(color: Int, alpha: Int): Int {
        return Color.argb(alpha.coerceIn(0, 255), Color.red(color), Color.green(color), Color.blue(color))
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
