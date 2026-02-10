package com.example.spinalx;

import android.app.Activity;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

public class UserInterface
{
    private ImageView imageView;
    private TextView resultText, tipText;
    private Spinner poseSpinner;
    private Button selectBtn, takePhotoBtn, classifyBtn;
    private ImageView infoIcon;

    public void initialize(Activity activity) {
        imageView = activity.findViewById(R.id.imageView);
        resultText = activity.findViewById(R.id.resultText);
        tipText = activity.findViewById(R.id.tipText);
        poseSpinner = activity.findViewById(R.id.poseSpinner);
        selectBtn = activity.findViewById(R.id.selectBtn);
        classifyBtn = activity.findViewById(R.id.classifyBtn);
        takePhotoBtn = activity.findViewById(R.id.takePhotoBtn);
        infoIcon = activity.findViewById(R.id.infoIcon);
    }

    public ImageView getImageView() { return imageView; }
    public TextView getResultText() { return resultText; }
    public TextView getTipText() { return tipText; }
    public Spinner getPoseSpinner() { return poseSpinner; }
    public Button getSelectBtn() { return selectBtn; }
    public Button getTakePhotoBtn() { return takePhotoBtn; }
    public Button getClassifyBtn() { return classifyBtn; }
    public ImageView getInfoIcon() { return infoIcon; }
}
