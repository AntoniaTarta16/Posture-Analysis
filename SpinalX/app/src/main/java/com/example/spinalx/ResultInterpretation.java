package com.example.spinalx;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ImageSpan;
import android.view.View;
import android.widget.TextView;

import androidx.core.content.ContextCompat;

import java.util.Set;

public class ResultInterpretation {
    private static final Set<String> goodLabels = Set.of("neutral_posture", "symmetrical_front");

    public void show(Context context, TextView resultView, TextView tipView, String label) {
        String formatted = formatLabel(label);
        SpannableString spannable = new SpannableString(formatted + "  ");
        Drawable icon = ContextCompat.getDrawable(context,
                goodLabels.contains(label) ? R.drawable.good : R.drawable.wrong);

        if (icon != null) {
            int size = (int)(resultView.getTextSize() * 1.5);
            icon.setBounds(0, 0, size, size);
            spannable.setSpan(new ImageSpan(icon, ImageSpan.ALIGN_BOTTOM),
                    spannable.length() - 1, spannable.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }

        resultView.setAlpha(0f);
        resultView.setText(spannable);
        resultView.animate().alpha(1f).setDuration(500).start();
        showTip(context, tipView, label);
    }

    private String formatLabel(String label) {
        String text = label.replace("_", " ");
        return Character.toUpperCase(text.charAt(0)) + text.substring(1);
    }

    private void showTip(Context context, TextView tipView, String label) {
        String tip;
        switch (label) {
            case "postural_asymmetry":
                tip = context.getString(R.string.tip_asymmetry);
                break;
            case "stooped_posture":
                tip = context.getString(R.string.tip_stooped);
                break;
            case "slouched_posture":
                tip = context.getString(R.string.tip_slouched);
                break;
            default:
                tip = null;
        }

        if (tip!= null) {
            tipView.setText(tip);
            tipView.setVisibility(View.VISIBLE);
        } else {
            tipView.setVisibility(View.GONE);
        }
    }
}
