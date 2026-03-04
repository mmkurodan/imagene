package com.micklab.imagene;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.graphics.Color;
import android.view.Gravity;
import android.widget.Toast;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final String TAG = "MainActivity";
    
    private EditText promptInput;
    private EditText negativePromptInput;
    private SeekBar stepsSeekBar;
    private SeekBar guidanceSeekBar;
    private TextView stepsLabel;
    private TextView guidanceLabel;
    private Button generateButton;
    private ImageView resultImage;
    private ProgressBar progressBar;
    private TextView statusText;
    
    private SdxlRunner sdxlRunner;
    private ExecutorService executor;
    private Handler mainHandler;
    
    private int numSteps = 20;
    private float guidanceScale = 7.5f;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Create UI programmatically
        setupUI();
        
        // Initialize
        executor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());
        
        // Initialize SDXL runner in background
        initializeRunner();
    }
    
    private void setupUI() {
        ScrollView scrollView = new ScrollView(this);
        scrollView.setBackgroundColor(Color.parseColor("#1a1a2e"));
        
        LinearLayout mainLayout = new LinearLayout(this);
        mainLayout.setOrientation(LinearLayout.VERTICAL);
        mainLayout.setPadding(32, 32, 32, 32);
        
        // Title
        TextView title = new TextView(this);
        title.setText("SDXL Image Generator");
        title.setTextSize(24);
        title.setTextColor(Color.WHITE);
        title.setGravity(Gravity.CENTER);
        title.setPadding(0, 0, 0, 32);
        mainLayout.addView(title);
        
        // Prompt input
        TextView promptLabel = new TextView(this);
        promptLabel.setText("Prompt:");
        promptLabel.setTextColor(Color.LTGRAY);
        mainLayout.addView(promptLabel);
        
        promptInput = new EditText(this);
        promptInput.setHint("Enter your prompt here...");
        promptInput.setMinLines(3);
        promptInput.setTextColor(Color.WHITE);
        promptInput.setHintTextColor(Color.GRAY);
        promptInput.setBackgroundColor(Color.parseColor("#16213e"));
        promptInput.setPadding(16, 16, 16, 16);
        promptInput.setText("A majestic dragon flying over mountains, fantasy art, detailed, 8k");
        mainLayout.addView(promptInput);
        
        // Negative prompt input
        TextView negPromptLabel = new TextView(this);
        negPromptLabel.setText("Negative Prompt:");
        negPromptLabel.setTextColor(Color.LTGRAY);
        negPromptLabel.setPadding(0, 16, 0, 0);
        mainLayout.addView(negPromptLabel);
        
        negativePromptInput = new EditText(this);
        negativePromptInput.setHint("Things to avoid...");
        negativePromptInput.setMinLines(2);
        negativePromptInput.setTextColor(Color.WHITE);
        negativePromptInput.setHintTextColor(Color.GRAY);
        negativePromptInput.setBackgroundColor(Color.parseColor("#16213e"));
        negativePromptInput.setPadding(16, 16, 16, 16);
        negativePromptInput.setText("blurry, low quality, distorted");
        mainLayout.addView(negativePromptInput);
        
        // Steps slider
        stepsLabel = new TextView(this);
        stepsLabel.setText("Steps: " + numSteps);
        stepsLabel.setTextColor(Color.LTGRAY);
        stepsLabel.setPadding(0, 24, 0, 0);
        mainLayout.addView(stepsLabel);
        
        stepsSeekBar = new SeekBar(this);
        stepsSeekBar.setMax(50);
        stepsSeekBar.setProgress(numSteps);
        stepsSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                numSteps = Math.max(1, progress);
                stepsLabel.setText("Steps: " + numSteps);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        mainLayout.addView(stepsSeekBar);
        
        // Guidance scale slider
        guidanceLabel = new TextView(this);
        guidanceLabel.setText("Guidance Scale: " + guidanceScale);
        guidanceLabel.setTextColor(Color.LTGRAY);
        guidanceLabel.setPadding(0, 16, 0, 0);
        mainLayout.addView(guidanceLabel);
        
        guidanceSeekBar = new SeekBar(this);
        guidanceSeekBar.setMax(200);
        guidanceSeekBar.setProgress((int)(guidanceScale * 10));
        guidanceSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                guidanceScale = progress / 10.0f;
                guidanceLabel.setText("Guidance Scale: " + guidanceScale);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        mainLayout.addView(guidanceSeekBar);
        
        // Generate button
        generateButton = new Button(this);
        generateButton.setText("Generate Image");
        generateButton.setTextColor(Color.WHITE);
        generateButton.setBackgroundColor(Color.parseColor("#e94560"));
        generateButton.setPadding(32, 24, 32, 24);
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        buttonParams.setMargins(0, 32, 0, 16);
        generateButton.setLayoutParams(buttonParams);
        generateButton.setEnabled(false);
        generateButton.setOnClickListener(v -> onGenerateClicked());
        mainLayout.addView(generateButton);
        
        // Progress bar
        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setProgress(0);
        progressBar.setVisibility(View.GONE);
        mainLayout.addView(progressBar);
        
        // Status text
        statusText = new TextView(this);
        statusText.setText("Initializing...");
        statusText.setTextColor(Color.LTGRAY);
        statusText.setGravity(Gravity.CENTER);
        statusText.setPadding(0, 8, 0, 16);
        mainLayout.addView(statusText);
        
        // Result image
        resultImage = new ImageView(this);
        resultImage.setMinimumHeight(512);
        resultImage.setBackgroundColor(Color.parseColor("#0f3460"));
        resultImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        resultImage.setAdjustViewBounds(true);
        mainLayout.addView(resultImage);
        
        scrollView.addView(mainLayout);
        setContentView(scrollView);
    }
    
    private void initializeRunner() {
        updateStatus("Loading models...");
        
        executor.execute(() -> {
            try {
                sdxlRunner = new SdxlRunner(this);
                sdxlRunner.setProgressCallback(this::onProgress);
                sdxlRunner.initialize();
                
                mainHandler.post(() -> {
                    generateButton.setEnabled(true);
                    updateStatus("Ready to generate");
                });
            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize SDXL", e);
                mainHandler.post(() -> {
                    updateStatus("Error: " + e.getMessage());
                    Toast.makeText(this, "Failed to load models: " + e.getMessage(), 
                        Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    private void onGenerateClicked() {
        String prompt = promptInput.getText().toString().trim();
        String negativePrompt = negativePromptInput.getText().toString().trim();
        
        if (prompt.isEmpty()) {
            Toast.makeText(this, "Please enter a prompt", Toast.LENGTH_SHORT).show();
            return;
        }
        
        generateButton.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setProgress(0);
        updateStatus("Generating...");
        
        executor.execute(() -> {
            try {
                long startTime = System.currentTimeMillis();
                
                Bitmap result = sdxlRunner.generate(
                    prompt,
                    negativePrompt,
                    numSteps,
                    guidanceScale,
                    1024,  // width
                    1024   // height
                );
                
                long elapsed = System.currentTimeMillis() - startTime;
                
                mainHandler.post(() -> {
                    resultImage.setImageBitmap(result);
                    progressBar.setVisibility(View.GONE);
                    generateButton.setEnabled(true);
                    updateStatus(String.format("Generated in %.1f seconds", elapsed / 1000.0));
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Generation failed", e);
                mainHandler.post(() -> {
                    progressBar.setVisibility(View.GONE);
                    generateButton.setEnabled(true);
                    updateStatus("Error: " + e.getMessage());
                    Toast.makeText(this, "Generation failed: " + e.getMessage(), 
                        Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    private void onProgress(int step, int totalSteps, String phase) {
        mainHandler.post(() -> {
            int progress = (int)((step * 100.0) / totalSteps);
            progressBar.setProgress(progress);
            updateStatus(phase + " (" + step + "/" + totalSteps + ")");
        });
    }
    
    private void updateStatus(String status) {
        statusText.setText(status);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (sdxlRunner != null) {
            sdxlRunner.close();
        }
        if (executor != null) {
            executor.shutdown();
        }
    }
}
