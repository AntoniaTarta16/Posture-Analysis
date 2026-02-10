# Posture-Analysis

## Overview

Poor sitting posture during office work can lead to long-term musculoskeletal disorders. This project proposes a mobile solution for real-time posture analysis, based on computer vision, designed to run offline and without additional hardware. The system analyzes user posture from camera images and classifies it into ergonomic categories, providing feedback to help users maintain a correct sitting posture during desk activities.
This repository contains the full implementation of the posture analysis system, including dataset preparation, model training, and an Android mobile application.

## System Architecture

The application is organized into modular components that cover the full posture analysis pipeline:

- Image acquisition

- Image preprocessing

- Posture classification

- Result interpretation and feedback


## Posture Classification

The system classifies posture into the following categories:

- Neutral posture

- Stooped posture

- Slouched posture

- Symmetric posture

- Asymmetric posture


## Dataset

A custom dataset was created specifically for this project, containing images of individuals performing office-related activities.

Dataset creation steps:

- Image collection

- Data cleaning to reduce overfitting

- Automatic labeling based on detected body keypoints

- Label assignment using ergonomics rules defined by specialists

- The dataset includes both frontal and lateral perspectives.


## Model Training

Several convolutional neural network architectures were evaluated:

- ResNet50

- EfficientNetB3

- MobileNetV2

- DenseNet121 (best performing)


Training strategy:

- Transfer learning

- Fine-tuning
