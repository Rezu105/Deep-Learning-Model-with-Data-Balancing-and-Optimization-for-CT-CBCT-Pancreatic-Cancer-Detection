import os
import pydicom
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import gc

# ------------------------
# Reproducibility
# ------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------
# DICOM loader
# ------------------------
from pydicom.pixel_data_handlers.util import apply_voi_lut

def load_and_preprocess_dicom(dcm_path, img_size=(128,128)):
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        if not hasattr(dcm, "pixel_array"):
            return None
        img = apply_voi_lut(dcm.pixel_array, dcm)
        if len(img.shape) == 3:
            img = img[img.shape[0]//2]  # middle slice for 3D
        img = img.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope'):
            img *= float(dcm.RescaleSlope)
        if hasattr(dcm, 'RescaleIntercept'):
            img += float(dcm.RescaleIntercept)
        img_min, img_max = np.min(img), np.max(img)
        if img_max <= img_min:
            return None
        img = (img - img_min) / (img_max - img_min)
        img = cv2.GaussianBlur(img, (3,3), sigmaX=1.0)
        img = cv2.medianBlur(img, 3)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(img, axis=-1)
    except:
        return None

# ------------------------
# Recursive dataset loader
# ------------------------
def load_dataset_recursive(base_path, max_images=18000, img_size=(128,128)):
    images, labels, count = [], [], 0
    print("[INFO] Loading DICOM files from subfolders...")

    for root, dirs, files in os.walk(base_path):
        dicom_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dicom_files:
            continue
        
        class_name = os.path.basename(root)  # deepest folder name as class
        
        for file in dicom_files:
            if count >= max_images:
                break
            img_path = os.path.join(root, file)
            img = load_and_preprocess_dicom(img_path, img_size)
            if img is not None:
                images.append(img)
                labels.append(class_name)
                count += 1
                if count % 500 == 0:
                    print(f"Loaded {count} images...")
                    gc.collect()
    
    print(f"[INFO] Finished loading. Total images: {count}")
    return np.array(images, dtype=np.float32), np.array(labels)

# ------------------------
# ResNet18 block
# ------------------------
def resnet_block(x, filters, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

# ------------------------
# Build ResNet18
# ------------------------
def build_resnet18(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, stride=2)
    x = resnet_block(x, 128)
    x = resnet_block(x, 256, stride=2)
    x = resnet_block(x, 256)
    x = resnet_block(x, 512, stride=2)
    x = resnet_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    # **Do NOT compile** since we are only doing inference
    return model

# ------------------------
# Main
# ------------------------
def main():
    base_path = r"E:\Thesis\Thesis_test\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
    img_size = (128,128)
    max_images = 18000
    test_size = 0.2
    random_state = 42
    
    images, labels = load_dataset_recursive(base_path, max_images, img_size)
    
    counts = Counter(labels)
    valid_classes = [cls for cls, c in counts.items() if c >= 2]
    filtered_idx = [i for i, label in enumerate(labels) if label in valid_classes]
    
    if len(filtered_idx) == 0:
        raise ValueError("No valid images found after filtering!")
    
    images = images[filtered_idx]
    labels = labels[filtered_idx]
    
    unique_labels = np.unique(labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])
    y_cat = tf.keras.utils.to_categorical(y)
    
    # Split dataset (even though we won't train)
    X_train, X_test, y_train, y_test = train_test_split(
        images, y_cat, test_size=test_size, random_state=random_state, stratify=y
    )
    
    del images, labels
    gc.collect()
    
    model = build_resnet18((img_size[0], img_size[1], 1), len(unique_labels))
    model.summary()
    
    # ------------------------
    # Inference only
    # ------------------------
    print("[INFO] Running predictions with random weights (untrained model)...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification Report (random weights):")
    print(classification_report(y_true, y_pred, target_names=unique_labels))

if __name__ == "__main__":
    main()
