!pip install tensorflow matplotlib scikit-learn

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

split_dir = '/kaggle/working'
test_dir = os.path.join(split_dir, "test")
BATCH_SIZE = 4
EPOCHS = 35
EPOCHS_FINE = 20
IMG_SIZE = 256
SEED = 42
IGNORED_CLASSES = {'symmetrical_front', 'postural_asymmetry'}
base_dir = '/kaggle/input/all-data/dataset3'

for d in [split_dir, test_dir]:
    shutil.rmtree(d, ignore_errors=True)
os.makedirs(split_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


classes = [d.name for d in Path(base_dir).iterdir() if d.is_dir() and d.name not in IGNORED_CLASSES]
train_split, val_split, test_split = 0.8, 0.2, 0.1

for cls in classes:
    images = list((Path(base_dir)/cls).glob("*.jpg")) + \
             list((Path(base_dir)/cls).glob("*.jpeg")) + \
             list((Path(base_dir)/cls).glob("*.png"))
    random.shuffle(images)

    test_size = int(len(images) * test_split)
    test_imgs = images[:test_size]
    remaining = images[test_size:]
    train_imgs, val_imgs = train_test_split(
        remaining,
        train_size=train_split / (train_split + val_split),
        random_state=SEED
    )

    for split_name, split_imgs in zip(['train', 'val'], [train_imgs, val_imgs]):
        split_path = Path(split_dir)/split_name/cls
        split_path.mkdir(parents=True, exist_ok=True)
        for img_path in split_imgs:
            shutil.copy(img_path, split_path)

    test_path = Path(test_dir)/cls
    test_path.mkdir(parents=True, exist_ok=True)
    for img_path in test_imgs:
        shutil.copy(img_path, test_path)

from tensorflow.keras.applications.densenet import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=0,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=(0.8, 1.2),
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(split_dir, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(split_dir, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    seed=SEED,
    shuffle=False
)

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

train_generator.reset()

unique, counts = np.unique(train_generator.classes, return_counts=True)
print("Distrib clase:", dict(zip(unique, counts)))

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(unique),
    y=train_generator.classes
)
class_weights_dict = dict(zip(unique, class_weights))
print("Class weights:", class_weights_dict)

steps_per_epoch = train_generator.samples // BATCH_SIZE

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return loss

base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.GaussianNoise(0.1),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation='softmax')
])

loss_init = categorical_focal_loss(gamma=2.0, alpha=0.25)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=steps_per_epoch * EPOCHS,
    alpha=1e-6
)
optimizer_initial = optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer_initial,
    loss=loss_init,
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
checkpoint = callbacks.ModelCheckpoint("best_model.weights.h5", monitor='val_accuracy', save_best_only=True, save_weights_only=True)

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights_dict
)

base_model.trainable = True
for layer in base_model.layers[:313]:
    layer.trainable = False

lr_fine = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=steps_per_epoch * EPOCHS_FINE,
    alpha=1e-7
)
optimizer_fine = optimizers.Adam(learning_rate=1e-5)
loss_fn = categorical_focal_loss(gamma=2.0, alpha=0.25)

model.compile(
    optimizer=optimizer_fine,
    loss=loss_fn,
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE,
    callbacks=[early_stop, lr_reduce, checkpoint],
    class_weight=class_weights_dict
)

top_5_val_acc = sorted(history_fine.history['val_accuracy'], reverse=True)[:5]
mean_top_5_val_acc = sum(top_5_val_acc) / len(top_5_val_acc)
print(f"Media celor mai bune 5 epoci (val_accuracy): {mean_top_5_val_acc:.4f}")
top_5_val_acc = sorted(history_initial.history['val_accuracy'], reverse=True)[:5]
mean_top_5_val_acc = sum(top_5_val_acc) / len(top_5_val_acc)
print(f"Media celor mai bune 5 epoci (val_accuracy): {mean_top_5_val_acc:.4f}")

def plot_confusion_matrix(generator, model, title="Confusion Matrix"):
    y_true = generator.classes
    y_pred_probs = model.predict(generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = list(generator.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("Etichete reale")
    plt.xlabel("Etichete prezise")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

plot_confusion_matrix(val_generator, model, title="Validation Confusion Matrix")

def plot_metrics(history, title="Train & Val Metrics"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_metrics(history_initial, title="Training Fara Fine-Tuning")
plot_metrics(history_fine, title="Fine-Tuning")

base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True
for layer in base_model.layers[:201]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.GaussianNoise(0.1),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.load_weights("best_model.weights.h5")

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

def predict_with_tta(model, generator, tta_steps=10):
    predictions = []

    for i in range(tta_steps):
        print(f"TTA step {i+1}/{tta_steps}")
        generator.reset()
        preds = model.predict(generator, verbose=0)
        predictions.append(preds)

    avg_preds = np.mean(predictions, axis=0)
    return avg_preds

tta_preds = predict_with_tta(model, test_generator, tta_steps=10)
tta_labels = np.argmax(tta_preds, axis=1)
true_labels = test_generator.classes

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

acc = accuracy_score(true_labels, tta_labels)
print(f"Accuracy TTA: {acc * 100:.2f}%")
print("Classification Report:")
print(classification_report(true_labels, tta_labels, target_names=list(test_generator.class_indices.keys())))

plot_confusion_matrix(test_generator, model, title="Test Confusion Matrix")