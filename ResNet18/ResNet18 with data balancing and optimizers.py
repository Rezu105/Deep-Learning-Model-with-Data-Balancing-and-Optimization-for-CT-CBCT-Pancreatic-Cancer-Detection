import os
import pydicom
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import gc

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# -------------------- Data Loading --------------------
def load_and_preprocess_dicom(dcm_path, img_size=(128,128)):
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        if not hasattr(dcm, 'pixel_array'):
            return None
        img = dcm.pixel_array
        if len(img.shape) == 3:  # 3D, take middle slice
            img = img[img.shape[0] // 2]
        img = img.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope'):
            img = img * float(dcm.RescaleSlope)
        if hasattr(dcm, 'RescaleIntercept'):
            img = img + float(dcm.RescaleIntercept)
        if np.max(img) <= np.min(img):
            return None
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = cv2.GaussianBlur(img, (3,3), sigmaX=1.0)
        img = cv2.medianBlur(img, 3)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(img, axis=-1)
    except:
        return None

def load_dataset(base_path, max_images=18000, img_size=(128,128)):
    images, labels = [], []
    count = 0
    class_counts = Counter()
    
    # Count samples per class
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                class_counts[os.path.basename(root)] += 1
    
    valid_classes = {cls for cls, c in class_counts.items() if c >= 2}
    
    for root, _, files in os.walk(base_path):
        cls = os.path.basename(root)
        if cls not in valid_classes:
            continue
        for file in files:
            if count >= max_images:
                break
            if file.lower().endswith('.dcm'):
                img = load_and_preprocess_dicom(os.path.join(root, file), img_size)
                if img is not None:
                    images.append(img)
                    labels.append(cls)
                    count += 1
                    if count % 500 == 0:
                        print(f"Loaded {count} images")
                        gc.collect()
    print(f"Total images loaded: {count}")
    return np.array(images, dtype=np.float16), np.array(labels)

# -------------------- ResNet-18 --------------------
def resnet_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def build_resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # ResNet-18: 2 blocks per stage
    filters = [64, 128, 256, 512]
    for i, f in enumerate(filters):
        stride = 1 if i == 0 else 2
        x = resnet_block(x, f, stride=stride)
        x = resnet_block(x, f, stride=1)
    
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------- Main --------------------
def main():
    base_path = r"E:\Thesis\Thesis_test\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
    img_size = (128,128)
    max_images = 18000
    batch_size = 64
    test_size = 0.2
    val_size = 0.25
    
    # Load dataset
    images, labels = load_dataset(base_path, max_images, img_size)
    unique_labels = np.unique(labels)
    label_to_idx = {l:i for i,l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels])
    
    # Class balancing
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))
    
    y = to_categorical(y)
    
    # Split dataset (60-20-20)
    X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42, stratify=y_train)
    
    del images, labels
    gc.collect()
    
    # Build model
    model = build_resnet18((img_size[0], img_size[1], 1), len(unique_labels))
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=unique_labels))
    
    # Plot metrics
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8,5))
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [
        precision_score(y_true, y_pred, average='weighted'),
        recall_score(y_true, y_pred, average='weighted'),
        f1_score(y_true, y_pred, average='weighted')
    ]
    plt.bar(metrics, values)
    plt.ylim(0,1)
    plt.title('Test Metrics')
    plt.show()

if __name__ == "__main__":
    main()
