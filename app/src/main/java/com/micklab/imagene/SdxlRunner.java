package com.micklab.imagene;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * SDXL inference runner using ONNX Runtime Mobile.
 * Handles text encoding, UNet denoising, and VAE decoding.
 */
public class SdxlRunner implements AutoCloseable {
    private static final String TAG = "SdxlRunner";
    private static final String MODELS_DIR = "models";
    
    private final Context context;
    private final AssetManager assetManager;
    
    private OrtEnvironment ortEnv;
    private OrtSession textEncoder1Session;
    private OrtSession textEncoder2Session;
    private OrtSession unetSession;
    private OrtSession vaeDecoderSession;
    
    private Map<String, Integer> vocab1;
    private Map<String, Integer> vocab2;
    
    private ProgressCallback progressCallback;
    
    // SDXL constants
    private static final int MAX_TOKEN_LENGTH = 77;
    private static final int TEXT_ENCODER_1_DIM = 768;
    private static final int TEXT_ENCODER_2_DIM = 1280;
    private static final int UNET_HIDDEN_DIM = 2048;
    private static final float VAE_SCALING_FACTOR = 0.13025f;
    
    public interface ProgressCallback {
        void onProgress(int step, int totalSteps, String phase);
    }
    
    public SdxlRunner(Context context) {
        this.context = context;
        this.assetManager = context.getAssets();
    }
    
    public void setProgressCallback(ProgressCallback callback) {
        this.progressCallback = callback;
    }
    
    /**
     * Initialize ONNX Runtime and load all models.
     */
    public void initialize() throws Exception {
        Log.i(TAG, "Initializing SDXL Runner...");
        
        ortEnv = OrtEnvironment.getEnvironment();
        
        SessionOptions sessionOptions = new SessionOptions();
        sessionOptions.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
        
        // Try to use NNAPI for hardware acceleration on Android
        try {
            sessionOptions.addNnapi();
            Log.i(TAG, "NNAPI enabled");
        } catch (Exception e) {
            Log.w(TAG, "NNAPI not available, using CPU: " + e.getMessage());
        }
        
        // Load models from assets
        reportProgress(0, 4, "Loading text encoder 1");
        textEncoder1Session = loadModel("text_encoder_1.onnx", sessionOptions);
        
        reportProgress(1, 4, "Loading text encoder 2");
        textEncoder2Session = loadModel("text_encoder_2.onnx", sessionOptions);
        
        reportProgress(2, 4, "Loading UNet");
        // UNet uses FP16, may need different options
        SessionOptions unetOptions = new SessionOptions();
        unetOptions.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
        try {
            unetOptions.addNnapi();
        } catch (Exception e) {
            Log.w(TAG, "NNAPI not available for UNet");
        }
        unetSession = loadModel("unet.onnx", unetOptions);
        
        reportProgress(3, 4, "Loading VAE decoder");
        vaeDecoderSession = loadModel("vae_decoder.onnx", sessionOptions);
        
        // Load tokenizer vocabularies
        vocab1 = loadVocabulary("tokenizer_1/vocab.json");
        vocab2 = loadVocabulary("tokenizer_2/vocab.json");
        
        reportProgress(4, 4, "Initialization complete");
        Log.i(TAG, "SDXL Runner initialized successfully");
    }
    
    /**
     * Load an ONNX model from assets.
     */
    private OrtSession loadModel(String modelName, SessionOptions options) throws Exception {
        String modelPath = MODELS_DIR + "/" + modelName;
        
        // Copy model to cache directory (ONNX Runtime needs file path)
        File cacheDir = context.getCacheDir();
        File modelFile = new File(cacheDir, modelName);
        
        if (!modelFile.exists()) {
            Log.i(TAG, "Copying model to cache: " + modelName);
            try (InputStream is = assetManager.open(modelPath);
                 FileOutputStream fos = new FileOutputStream(modelFile)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, bytesRead);
                }
            }
        }
        
        Log.i(TAG, "Loading model: " + modelName);
        return ortEnv.createSession(modelFile.getAbsolutePath(), options);
    }
    
    /**
     * Load tokenizer vocabulary from assets.
     */
    private Map<String, Integer> loadVocabulary(String path) throws IOException {
        String fullPath = MODELS_DIR + "/" + path;
        try (InputStream is = assetManager.open(fullPath);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            StringBuilder json = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                json.append(line);
            }
            Gson gson = new Gson();
            return gson.fromJson(json.toString(), 
                new TypeToken<Map<String, Integer>>(){}.getType());
        }
    }
    
    /**
     * Generate an image from a text prompt.
     */
    public Bitmap generate(String prompt, String negativePrompt, 
                          int numSteps, float guidanceScale,
                          int width, int height) throws Exception {
        Log.i(TAG, "Starting generation: " + prompt);
        
        int latentHeight = height / 8;
        int latentWidth = width / 8;
        int totalSteps = numSteps + 3; // +3 for encoding and VAE
        
        // Step 1: Encode prompts
        reportProgress(0, totalSteps, "Encoding prompts");
        float[][] promptEmbeddings = encodePrompts(prompt, negativePrompt, width, height);
        float[] promptEmbeds = promptEmbeddings[0];
        float[] pooledPromptEmbeds = promptEmbeddings[1];
        float[] negPromptEmbeds = promptEmbeddings[2];
        float[] negPooledPromptEmbeds = promptEmbeddings[3];
        float[] timeIds = promptEmbeddings[4];
        
        // Step 2: Initialize latents with random noise
        reportProgress(1, totalSteps, "Initializing latents");
        float[] latents = initializeLatents(latentHeight, latentWidth);
        
        // Step 3: Prepare scheduler (Euler Discrete)
        float[] timesteps = getTimesteps(numSteps);
        float[] sigmas = getSigmas(timesteps);
        
        // Step 4: Denoising loop
        for (int i = 0; i < numSteps; i++) {
            reportProgress(i + 2, totalSteps, "Denoising step " + (i + 1));
            
            float sigma = sigmas[i];
            float timestep = timesteps[i];
            
            // Scale latents
            float[] scaledLatents = scaleModelInput(latents, sigma);
            
            // Predict noise for conditional (with prompt)
            float[] noisePredCond = predictNoise(
                scaledLatents, timestep, promptEmbeds, pooledPromptEmbeds, timeIds,
                latentHeight, latentWidth);
            
            // Predict noise for unconditional (with negative prompt)
            float[] noisePredUncond = predictNoise(
                scaledLatents, timestep, negPromptEmbeds, negPooledPromptEmbeds, timeIds,
                latentHeight, latentWidth);
            
            // Classifier-free guidance
            float[] noisePred = classifierFreeGuidance(
                noisePredUncond, noisePredCond, guidanceScale);
            
            // Euler step
            latents = eulerStep(latents, noisePred, sigma, sigmas, i);
        }
        
        // Step 5: Decode latents with VAE
        reportProgress(numSteps + 2, totalSteps, "Decoding image");
        float[] image = decodeLatents(latents, latentHeight, latentWidth);
        
        // Step 6: Convert to Bitmap
        Bitmap bitmap = floatArrayToBitmap(image, width, height);
        
        reportProgress(totalSteps, totalSteps, "Complete");
        return bitmap;
    }
    
    /**
     * Encode prompts using both text encoders.
     */
    private float[][] encodePrompts(String prompt, String negativePrompt, 
                                   int width, int height) throws Exception {
        // Tokenize
        long[] tokens1 = tokenize(prompt, vocab1);
        long[] tokens2 = tokenize(prompt, vocab2);
        long[] negTokens1 = tokenize(negativePrompt, vocab1);
        long[] negTokens2 = tokenize(negativePrompt, vocab2);
        
        // Encode with text encoder 1
        float[] hidden1 = runTextEncoder(textEncoder1Session, tokens1, TEXT_ENCODER_1_DIM);
        float[] negHidden1 = runTextEncoder(textEncoder1Session, negTokens1, TEXT_ENCODER_1_DIM);
        
        // Encode with text encoder 2
        float[][] enc2Result = runTextEncoder2(textEncoder2Session, tokens2);
        float[] hidden2 = enc2Result[0];
        float[] pooled = enc2Result[1];
        
        float[][] negEnc2Result = runTextEncoder2(textEncoder2Session, negTokens2);
        float[] negHidden2 = negEnc2Result[0];
        float[] negPooled = negEnc2Result[1];
        
        // Concatenate hidden states [batch, seq, hidden1 + hidden2]
        float[] promptEmbeds = concatenateEmbeddings(hidden1, hidden2);
        float[] negPromptEmbeds = concatenateEmbeddings(negHidden1, negHidden2);
        
        // Time IDs for SDXL: [original_height, original_width, crop_top, crop_left, target_height, target_width]
        float[] timeIds = new float[]{height, width, 0, 0, height, width};
        
        return new float[][]{promptEmbeds, pooled, negPromptEmbeds, negPooled, timeIds};
    }
    
    /**
     * Simple tokenizer implementation.
     */
    private long[] tokenize(String text, Map<String, Integer> vocab) {
        long[] tokens = new long[MAX_TOKEN_LENGTH];
        Arrays.fill(tokens, vocab.getOrDefault("<|endoftext|>", 49407));
        
        // BOS token
        tokens[0] = vocab.getOrDefault("<|startoftext|>", 49406);
        
        // Simple whitespace tokenization (production would use BPE)
        String[] words = text.toLowerCase().split("\\s+");
        int tokenIdx = 1;
        
        for (String word : words) {
            if (tokenIdx >= MAX_TOKEN_LENGTH - 1) break;
            
            // Try to find exact match or subwords
            Integer tokenId = vocab.get(word + "</w>");
            if (tokenId == null) {
                tokenId = vocab.get(word);
            }
            if (tokenId == null) {
                // Use unknown token
                tokenId = vocab.getOrDefault("<|endoftext|>", 49407);
            }
            tokens[tokenIdx++] = tokenId;
        }
        
        // EOS token
        if (tokenIdx < MAX_TOKEN_LENGTH) {
            tokens[tokenIdx] = vocab.getOrDefault("<|endoftext|>", 49407);
        }
        
        return tokens;
    }
    
    /**
     * Run text encoder 1.
     */
    private float[] runTextEncoder(OrtSession session, long[] tokens, int hiddenDim) throws Exception {
        LongBuffer tokenBuffer = LongBuffer.wrap(tokens);
        OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, tokenBuffer, 
            new long[]{1, MAX_TOKEN_LENGTH});
        
        Map<String, OnnxTensor> inputs = Collections.singletonMap("input_ids", inputTensor);
        
        try (OrtSession.Result result = session.run(inputs)) {
            float[][][] output = (float[][][]) result.get(0).getValue();
            
            // Flatten [1, seq, hidden] -> [seq * hidden]
            float[] flat = new float[MAX_TOKEN_LENGTH * hiddenDim];
            for (int i = 0; i < MAX_TOKEN_LENGTH; i++) {
                System.arraycopy(output[0][i], 0, flat, i * hiddenDim, hiddenDim);
            }
            return flat;
        } finally {
            inputTensor.close();
        }
    }
    
    /**
     * Run text encoder 2 (returns hidden states and pooled output).
     */
    private float[][] runTextEncoder2(OrtSession session, long[] tokens) throws Exception {
        LongBuffer tokenBuffer = LongBuffer.wrap(tokens);
        OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, tokenBuffer, 
            new long[]{1, MAX_TOKEN_LENGTH});
        
        Map<String, OnnxTensor> inputs = Collections.singletonMap("input_ids", inputTensor);
        
        try (OrtSession.Result result = session.run(inputs)) {
            float[][][] hiddenStates = (float[][][]) result.get(0).getValue();
            float[][] pooledOutput = (float[][]) result.get(1).getValue();
            
            // Flatten hidden states
            float[] flat = new float[MAX_TOKEN_LENGTH * TEXT_ENCODER_2_DIM];
            for (int i = 0; i < MAX_TOKEN_LENGTH; i++) {
                System.arraycopy(hiddenStates[0][i], 0, flat, i * TEXT_ENCODER_2_DIM, TEXT_ENCODER_2_DIM);
            }
            
            return new float[][]{flat, pooledOutput[0]};
        } finally {
            inputTensor.close();
        }
    }
    
    /**
     * Concatenate embeddings from both encoders.
     */
    private float[] concatenateEmbeddings(float[] hidden1, float[] hidden2) {
        float[] result = new float[MAX_TOKEN_LENGTH * UNET_HIDDEN_DIM];
        
        for (int i = 0; i < MAX_TOKEN_LENGTH; i++) {
            // Copy from encoder 1
            System.arraycopy(hidden1, i * TEXT_ENCODER_1_DIM, 
                result, i * UNET_HIDDEN_DIM, TEXT_ENCODER_1_DIM);
            // Copy from encoder 2
            System.arraycopy(hidden2, i * TEXT_ENCODER_2_DIM, 
                result, i * UNET_HIDDEN_DIM + TEXT_ENCODER_1_DIM, TEXT_ENCODER_2_DIM);
        }
        
        return result;
    }
    
    /**
     * Initialize random latents.
     */
    private float[] initializeLatents(int height, int width) {
        int size = 4 * height * width;
        float[] latents = new float[size];
        Random random = new Random();
        
        for (int i = 0; i < size; i++) {
            latents[i] = (float) random.nextGaussian();
        }
        
        return latents;
    }
    
    /**
     * Get timesteps for Euler scheduler.
     */
    private float[] getTimesteps(int numSteps) {
        float[] timesteps = new float[numSteps];
        float maxTimestep = 999.0f;
        
        for (int i = 0; i < numSteps; i++) {
            timesteps[i] = maxTimestep - (maxTimestep * i / (numSteps - 1));
        }
        
        return timesteps;
    }
    
    /**
     * Get sigma values for Euler scheduler.
     */
    private float[] getSigmas(float[] timesteps) {
        float[] sigmas = new float[timesteps.length + 1];
        
        for (int i = 0; i < timesteps.length; i++) {
            float t = timesteps[i];
            // Linear schedule beta
            float beta = 0.00085f + (0.012f - 0.00085f) * t / 999.0f;
            float alpha = 1.0f - beta;
            float alphaCumprod = (float) Math.pow(alpha, t);
            sigmas[i] = (float) Math.sqrt((1.0f - alphaCumprod) / alphaCumprod);
        }
        sigmas[timesteps.length] = 0.0f;
        
        return sigmas;
    }
    
    /**
     * Scale model input for Euler scheduler.
     */
    private float[] scaleModelInput(float[] latents, float sigma) {
        float[] scaled = new float[latents.length];
        float scale = 1.0f / (float) Math.sqrt(sigma * sigma + 1.0f);
        
        for (int i = 0; i < latents.length; i++) {
            scaled[i] = latents[i] * scale;
        }
        
        return scaled;
    }
    
    /**
     * Predict noise with UNet.
     */
    private float[] predictNoise(float[] latents, float timestep, 
                                float[] promptEmbeds, float[] pooledEmbeds, float[] timeIds,
                                int latentHeight, int latentWidth) throws Exception {
        // Prepare inputs
        FloatBuffer sampleBuffer = FloatBuffer.wrap(latents);
        OnnxTensor sampleTensor = OnnxTensor.createTensor(ortEnv, sampleBuffer,
            new long[]{1, 4, latentHeight, latentWidth});
        
        FloatBuffer timestepBuffer = FloatBuffer.wrap(new float[]{timestep});
        OnnxTensor timestepTensor = OnnxTensor.createTensor(ortEnv, timestepBuffer, new long[]{1});
        
        FloatBuffer embedsBuffer = FloatBuffer.wrap(promptEmbeds);
        OnnxTensor embedsTensor = OnnxTensor.createTensor(ortEnv, embedsBuffer,
            new long[]{1, MAX_TOKEN_LENGTH, UNET_HIDDEN_DIM});
        
        FloatBuffer pooledBuffer = FloatBuffer.wrap(pooledEmbeds);
        OnnxTensor pooledTensor = OnnxTensor.createTensor(ortEnv, pooledBuffer,
            new long[]{1, TEXT_ENCODER_2_DIM});
        
        FloatBuffer timeIdsBuffer = FloatBuffer.wrap(timeIds);
        OnnxTensor timeIdsTensor = OnnxTensor.createTensor(ortEnv, timeIdsBuffer,
            new long[]{1, 6});
        
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("sample", sampleTensor);
        inputs.put("timestep", timestepTensor);
        inputs.put("encoder_hidden_states", embedsTensor);
        inputs.put("text_embeds", pooledTensor);
        inputs.put("time_ids", timeIdsTensor);
        
        try (OrtSession.Result result = unetSession.run(inputs)) {
            float[][][][] output = (float[][][][]) result.get(0).getValue();
            
            // Flatten output
            float[] flat = new float[4 * latentHeight * latentWidth];
            int idx = 0;
            for (int c = 0; c < 4; c++) {
                for (int h = 0; h < latentHeight; h++) {
                    for (int w = 0; w < latentWidth; w++) {
                        flat[idx++] = output[0][c][h][w];
                    }
                }
            }
            return flat;
        } finally {
            sampleTensor.close();
            timestepTensor.close();
            embedsTensor.close();
            pooledTensor.close();
            timeIdsTensor.close();
        }
    }
    
    /**
     * Apply classifier-free guidance.
     */
    private float[] classifierFreeGuidance(float[] uncond, float[] cond, float scale) {
        float[] result = new float[uncond.length];
        
        for (int i = 0; i < uncond.length; i++) {
            result[i] = uncond[i] + scale * (cond[i] - uncond[i]);
        }
        
        return result;
    }
    
    /**
     * Euler step.
     */
    private float[] eulerStep(float[] latents, float[] noisePred, 
                             float sigma, float[] sigmas, int stepIdx) {
        float[] result = new float[latents.length];
        float dt = sigmas[stepIdx + 1] - sigma;
        
        for (int i = 0; i < latents.length; i++) {
            float derivative = (latents[i] - noisePred[i]) / sigma;
            result[i] = latents[i] + derivative * dt;
        }
        
        return result;
    }
    
    /**
     * Decode latents with VAE.
     */
    private float[] decodeLatents(float[] latents, int latentHeight, int latentWidth) throws Exception {
        // Scale latents
        float[] scaled = new float[latents.length];
        for (int i = 0; i < latents.length; i++) {
            scaled[i] = latents[i] / VAE_SCALING_FACTOR;
        }
        
        FloatBuffer latentBuffer = FloatBuffer.wrap(scaled);
        OnnxTensor latentTensor = OnnxTensor.createTensor(ortEnv, latentBuffer,
            new long[]{1, 4, latentHeight, latentWidth});
        
        Map<String, OnnxTensor> inputs = Collections.singletonMap("latent", latentTensor);
        
        try (OrtSession.Result result = vaeDecoderSession.run(inputs)) {
            float[][][][] output = (float[][][][]) result.get(0).getValue();
            
            int height = output[0][0].length;
            int width = output[0][0][0].length;
            
            // Convert CHW to HWC and normalize to 0-1
            float[] image = new float[height * width * 3];
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = (h * width + w) * 3;
                    for (int c = 0; c < 3; c++) {
                        float val = (output[0][c][h][w] + 1.0f) / 2.0f;
                        image[idx + c] = Math.max(0.0f, Math.min(1.0f, val));
                    }
                }
            }
            return image;
        } finally {
            latentTensor.close();
        }
    }
    
    /**
     * Convert float array to Bitmap.
     */
    private Bitmap floatArrayToBitmap(float[] image, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        
        for (int i = 0; i < width * height; i++) {
            int r = (int) (image[i * 3] * 255);
            int g = (int) (image[i * 3 + 1] * 255);
            int b = (int) (image[i * 3 + 2] * 255);
            pixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }
    
    private void reportProgress(int step, int total, String phase) {
        if (progressCallback != null) {
            progressCallback.onProgress(step, total, phase);
        }
    }
    
    @Override
    public void close() {
        try {
            if (textEncoder1Session != null) textEncoder1Session.close();
            if (textEncoder2Session != null) textEncoder2Session.close();
            if (unetSession != null) unetSession.close();
            if (vaeDecoderSession != null) vaeDecoderSession.close();
            if (ortEnv != null) ortEnv.close();
        } catch (Exception e) {
            Log.e(TAG, "Error closing ONNX sessions", e);
        }
    }
}
