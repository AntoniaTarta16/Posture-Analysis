package com.example.spinalx;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class ImageClassifier {
    private final Interpreter interpreter;
    private final List<String> labels;
    private final ImagePreprocessing preprocessor = new ImagePreprocessing();

    public ImageClassifier(Context context, String modelName, String labelName) throws IOException {
        interpreter = new Interpreter(loadModelFile(context, modelName));
        labels = loadLabelList(context, labelName);
    }

    private MappedByteBuffer loadModelFile(Context context, String filename) throws IOException {
        try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd(filename);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
             FileChannel fileChannel = inputStream.getChannel()) {

            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();

            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    private List<String> loadLabelList(Context context, String filename) throws IOException {
        List<String> labelList = new ArrayList<>();
        InputStream is = context.getAssets().open(filename);
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    public String classify(Bitmap bitmap) {
        int imageSize = 256;
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);
        ByteBuffer inputBuffer = preprocessor.convertBitmapToByteBuffer(resized, imageSize);

        float[][] output = new float[1][labels.size()];
        interpreter.run(inputBuffer, output);

        int maxIndex = 0;
        float maxProb = 0;
        for (int i = 0; i < labels.size(); i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }
        return labels.get(maxIndex) + " (" + (int)(maxProb * 100) + "%)";
    }
}