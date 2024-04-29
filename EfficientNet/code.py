import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from pathlib import Path
import os
import seaborn as sns
import matplotlib.cm as cm
import cv2

from helper_functions import plot_loss_curves, create_tensorboard_callback, pred_and_plot

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
DATASET_PATH = "./input/bird-species/train"

image_path = Path(DATASET_PATH)
all_images = list(image_path.rglob('*.JPG')) + list(image_path.rglob('*.jpg')) + list(image_path.rglob('*.png'))

labels = [os.path.split(os.path.split(img)[0])[1] for img in all_images]
df = pd.DataFrame({"Filepath": all_images, "Label": labels})

# Visualizing the Top 20 Most Frequent Bird Species
top_labels = df['Label'].value_counts().head(20)
plt.figure(figsize=(20, 10))
sns.barplot(x=top_labels.values, y=top_labels.index, alpha=0.8)
plt.title("Top 20 Most Frequent Bird Species")
plt.xlabel("Count")
plt.ylabel("Bird Species")
plt.show()

# Splitting Data into Train and Test Sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'])

# Data Generators
train_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1. / 255)

train_data = train_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# MobileNetV2 Model with Transfer Learning
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights=None, classes=len(train_df['Label'].unique()))

base_model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(),
    metrics=['accuracy']
)

# Callbacks and Model Training
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1), ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)]

history = base_model.fit(
    train_data,
    epochs=10,
    validation_data=test_data,
    callbacks=callbacks
)

# Plotting Training Results
plot_loss_curves(history)

# Model Performance Evaluation
test_loss, test_accuracy = base_model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy}")

# Detailed Classification Report
from sklearn.metrics import classification_report, confusion_matrix

test_predictions = base_model.predict(test_data)
predicted_classes = np.argmax(test_predictions, axis=1)

report = classification_report(test_data.classes, predicted_classes, target_names=test_data.class_indices.keys())
print(report)

