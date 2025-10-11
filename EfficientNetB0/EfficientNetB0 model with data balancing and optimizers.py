import os
import random
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# SETTINGS
IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30
LIMIT = 18000
SEED = 42

# GPU Check
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU detected: {gpus[0].name}" if gpus else "No GPU found, running on CPU.")

# Dataset Path
dataset_path = r"E:\Work folder\AIUB\Reasearch Folder\Thesis_G42\PancreasProject\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"


# HELPER FUNCTIONS
def load_dicom_paths_and_labels(root_path):
    class_folders = [os.path.join(root_path, cls) for cls in os.listdir(root_path)
                     if os.path.isdir(os.path.join(root_path, cls))]
    image_paths, labels = [], []
    for idx, folder in enumerate(class_folders):
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.dcm'):
                    image_paths.append(os.path.join(root, file))
                    labels.append(idx)
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    combined = combined[:LIMIT]
    return zip(*combined)

def read_dicom_file(path):
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        return img
    except Exception as e:
        print(f"Skipped file {path}, error: {e}")
        return None

def dicom_generator(paths, labels, batch_size, datagen=None):
    while True:
        x_batch, y_batch = [], []
        zipped = list(zip(paths, labels))
        random.shuffle(zipped)
        for path, label in zipped:
            img = read_dicom_file(path)
            if img is not None:
                if datagen:
                    img = datagen.random_transform(img)
                x_batch.append(img)
                y_batch.append(label)
                if len(x_batch) == batch_size:
                    yield (
                        np.array(x_batch),
                        tf.keras.utils.to_categorical(np.array(y_batch), num_classes=NUM_CLASSES)
                    )
                    x_batch, y_batch = [], []
        if x_batch:
            yield (
                np.array(x_batch),
                tf.keras.utils.to_categorical(np.array(y_batch), num_classes=NUM_CLASSES)
            )


# LOAD DATA
all_paths, all_labels = load_dicom_paths_and_labels(dataset_path)
all_paths, all_labels = list(all_paths), list(all_labels)

NUM_CLASSES = len(set(all_labels))
print(f"Detected {NUM_CLASSES} unique classes.")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, random_state=SEED
)


# DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.1
)


# CLASS WEIGHTS
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(class_weights))


# GENERATORS
train_gen = dicom_generator(train_paths, train_labels, BATCH_SIZE, datagen=datagen)
val_gen = dicom_generator(val_paths, val_labels, BATCH_SIZE)

steps_per_epoch = len(train_paths) // BATCH_SIZE
validation_steps = len(val_paths) // BATCH_SIZE


# PREPARE VALIDATION DATA
val_images, val_labels_clean = [], []
for path, label in zip(val_paths, val_labels):
    img = read_dicom_file(path)
    if img is not None:
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        val_images.append(img)
        val_labels_clean.append(label)

val_images = np.array(val_images, dtype=np.float32)   # Force proper array
val_labels_cat = tf.keras.utils.to_categorical(val_labels_clean, num_classes=NUM_CLASSES)


# MODEL
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output_tensor = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)


# OPTIMIZER (RMSprop)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# TRAIN
model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=(val_images, val_labels_cat),
    validation_steps=validation_steps,
    epochs=EPOCHS,
    verbose=1,
    class_weight=class_weights_dict
)


# EVALUATE
y_pred = model.predict(val_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.array(val_labels_clean)

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='macro')
recall = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')

print(f"\nResults with Adam:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))
