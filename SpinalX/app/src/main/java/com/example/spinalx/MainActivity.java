package com.example.spinalx;

import android.content.Intent;
import android.graphics.Bitmap;

import androidx.appcompat.app.AlertDialog;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Toast;


import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.splashscreen.SplashScreen;

import java.io.File;

public class MainActivity extends AppCompatActivity {
    private final UserInterface ui = new UserInterface();
    private final ImageAcquisition acquisition = new ImageAcquisition();
    private final ImagePreprocessing preprocessor = new ImagePreprocessing();
    private final ResultInterpretation interpreter = new ResultInterpretation();

    private Bitmap selectedBitmap;
    private Uri imageUri;
    private File photoFile;
    private boolean keepSplash = true;

    private ActivityResultLauncher<Intent> imagePickerLauncher;
    private ActivityResultLauncher<Uri> takePhotoLauncher;
    private ActivityResultLauncher<String> cameraPermissionLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        SplashScreen splash = SplashScreen.installSplashScreen(this);
        splash.setKeepOnScreenCondition(() -> keepSplash);

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ui.initialize(this);
        new Handler().postDelayed(() -> keepSplash = false, 1500);
        initializeSpinner();
        setupLaunchers();
        setupListeners();
    }

    private void initializeSpinner() {
        String[] poses = {"Select posture...", "Frontal", "Side"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, poses);
        ui.getPoseSpinner().setAdapter(adapter);
    }

    private void setupLaunchers() {
        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        imageUri = result.getData().getData();
                        try {
                            selectedBitmap = acquisition.loadFromUri(this, imageUri);
                            selectedBitmap = preprocessor.rotateImageIfRequired(this, selectedBitmap, imageUri);
                            ui.getImageView().setImageBitmap(selectedBitmap);
                        } catch (Exception e) {
                            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                        }
                    }
                });

        takePhotoLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicture(),
                success -> {
                    if (success && photoFile != null) {
                        try {
                            selectedBitmap = acquisition.loadFromUri(this, imageUri);
                            selectedBitmap = preprocessor.rotateImageIfRequired(this, selectedBitmap, imageUri);
                            ui.getImageView().setImageBitmap(selectedBitmap);
                        } catch (Exception e) {
                            Log.e("MainActivity", "Failed to load image", e);
                            Toast.makeText(this, "Failed to load photo", Toast.LENGTH_SHORT).show();
                        }
                    }
                });

        cameraPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                granted -> {
                    if (granted) {
                        acquisition.launchCamera(this, (uri, file) -> {
                            imageUri = uri;
                            photoFile = file;
                        }, takePhotoLauncher);
                    } else {
                        Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
                    }
                });
    }

    private void setupListeners() {
        ui.getSelectBtn().setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            imagePickerLauncher.launch(intent);
        });

        ui.getTakePhotoBtn().setOnClickListener(v ->
                acquisition.requestCameraPermission(this, cameraPermissionLauncher)
        );

        ui.getClassifyBtn().setOnClickListener(v -> {
            if (selectedBitmap == null) {
                Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show();
                return;
            }

            String pose = ui.getPoseSpinner().getSelectedItem().toString();
            if (pose.equals("Select posture...")) {
                Toast.makeText(this, "Please choose a posture orientation", Toast.LENGTH_SHORT).show();
                return;
            }

            runClassification(pose);
        });

        ui.getInfoIcon().setOnClickListener(v -> {
            View dialogView = getLayoutInflater().inflate(R.layout.posture_tip, null);
            AlertDialog dialog = new AlertDialog.Builder(this).setView(dialogView).create();
            dialogView.findViewById(R.id.gotItBtn).setOnClickListener(btn -> dialog.dismiss());
            dialog.show();
        });
    }

    private void runClassification(String posture) {
        String model = posture.equals("Frontal") ? "front_posture_model.tflite" : "side_posture_model.tflite";
        String labels = posture.equals("Frontal") ? "labels_frontal.txt" : "labels_lateral.txt";

        try {
            ImageClassifier classifier = new ImageClassifier(this, model, labels);
            String raw = classifier.classify(selectedBitmap);
            String label = raw.contains("(") ? raw.substring(0, raw.indexOf("(")).trim().toLowerCase() : raw.trim().toLowerCase();
            interpreter.show(this, ui.getResultText(), ui.getTipText(), label);
        } catch (Exception e) {
            Toast.makeText(this, "Classification failed", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (photoFile != null && photoFile.exists()) {
            boolean deleted = photoFile.delete();
            Log.d("MainActivity", "Temporary photo deleted: " + deleted);
        }
    }
}