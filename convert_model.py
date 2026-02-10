from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow as tf

IMG_SIZE = 256
NUM_CLASSES = 2
WEIGHTS_PATH = "model.weights.h5"

base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True
for layer in base_model.layers[:313]:
    layer.trainable = False

reg = tf.keras.regularizers.l2(0.001)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.GaussianNoise(0.1),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.load_weights(WEIGHTS_PATH)

model.save("front_posture_model.h5")

model = tf.keras.models.load_model("front_posture_model.h5", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("front_posture_model.tflite", "wb") as f:
    f.write(tflite_model)