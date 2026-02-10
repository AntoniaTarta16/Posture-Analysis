package com.example.spinalx;

import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Environment;
import android.widget.Toast;
import android.Manifest;


import androidx.activity.result.ActivityResultLauncher;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.IOException;

public class ImageAcquisition {

    public interface CaptureCallback {
        void onCaptureReady(Uri uri, File file);
    }

    public Bitmap loadFromUri(Context context, Uri uri) throws IOException {
        return BitmapFactory.decodeStream(context.getContentResolver().openInputStream(uri));
    }

    public void requestCameraPermission(Activity activity, ActivityResultLauncher<String> launcher) {
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            launcher.launch(Manifest.permission.CAMERA);
        }
    }
    private File createImageFile(Context context) throws IOException {
        String fileName = "spinal_photo_" + System.currentTimeMillis();
        File storageDir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        return File.createTempFile(fileName, ".jpg", storageDir);
    }

    public void launchCamera(Activity activity, CaptureCallback callback,
                             ActivityResultLauncher<Uri> takePhotoLauncher) {
        try {
            File photoFile = createImageFile(activity);
            Uri photoUri = FileProvider.getUriForFile(activity,
                    "com.example.spinalx.fileprovider", photoFile);
            callback.onCaptureReady(photoUri, photoFile);
            takePhotoLauncher.launch(photoUri);
        } catch (IOException e) {
            Toast.makeText(activity, "Failed to start camera", Toast.LENGTH_SHORT).show();
        }
    }

}
