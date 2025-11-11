import os
import pydicom
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import gc
from collections import Counter
import tensorflow as tf

# extra optimizers used in other experiments (not used in this baseline script)
# from tensorflow.keras.optimizers import RMSprop          # tuned RMSprop
# import tensorflow_addons as tfa                          # RAdam, Lion  (pip install tensorflow-addons)
# from adabelief_tf import AdaBelief                       # AdaBelief    (pip install adabelief-tf)

# extra data balancing techniques used in other experiments (not used here)
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import RandomOverSampler     # SMOT (simple minority oversampling)
# from imblearn.over_sampling import SMOTE                 # SMOTE
# from imblearn.combine import SMOTEENN                    # SMOTE + ENN

# set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_dicom(dcm_path, img_size=(128, 128)):
    """Load and preprocess DICOM file with enhanced error handling"""
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        if not hasattr(dcm, 'pixel_array'):
            print(f"Skipped {dcm_path}: No pixel array")
            return None

        img = dcm.pixel_array

        # handle 3D arrays by taking middle slice
        if len(img.shape) == 3:
            img = img[img.shape[0] // 2]  # middle slice

        # convert and normalize using DICOM metadata
        img = img.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope'):
            img = img * float(dcm.RescaleSlope)
        if hasattr(dcm, 'RescaleIntercept'):
            img = img + float(dcm.RescaleIntercept)

        # skip empty/invalid images
        img_min, img_max = np.min(img), np.max(img)
        if img_max <= img_min:
            print(f"Skipped {dcm_path}: Invalid pixel range ({img_min}, {img_max})")
            return None

        # normalize and apply noise reduction
        img = (img - img_min) / (img_max - img_min)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)
        img = cv2.medianBlur(img, 3)  # additional noise reduction

        # resize and add channel dimension
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(img, axis=-1)

    except Exception as e:
        print(f"Error processing {dcm_path}: {str(e)}")
        return None


def load_large_dataset(base_path, max_images=18000, img_size=(128, 128)):
    """Load dataset without any explicit class balancing"""
    images = []
    labels = []
    count = 0

    print(f"[INFO] Loading up to {max_images} images...")

    # simple loading without pre-counting or filtering classes
    for root, _, files in os.walk(base_path):
        class_name = os.path.basename(root)

        for file in files:
            if count >= max_images:
                break

            if file.lower().endswith('.dcm'):
                img = load_and_preprocess_dicom(os.path.join(root, file), img_size)
                if img is not None:
                    images.append(img)
                    labels.append(class_name)
                    count += 1

                    if count % 500 == 0:
                        print(f"Loaded {count} images")
                        gc.collect()

    print(f"\nFinished loading. Total: {count} images")
    return np.array(images, dtype=np.float16), np.array(labels)


def build_model(input_shape, num_classes):
    """Enhanced CNN model with BatchNorm and regularization"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # baseline: no optimizer tuning, use default Adam as generic optimizer
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    # configuration
    base_path = r"D:\NBIA Pancreatic images\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
    img_size = (128, 128)
    max_images = 18000  # matches thesis requirement
    test_size = 0.2
    val_size = 0.25  # 20% test, 20% val (25% of remaining after test)
    random_state = 42
    batch_size = 64

    try:
        # load dataset
        images, labels = load_large_dataset(base_path, max_images, img_size)

        # prepare labels
        unique_labels = np.unique(labels)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_to_idx[label] for label in labels])

        # NOTE: no data balancing here (no class weights, no SMOTE/SMOT/SMOTE+ENN)

        # convert to categorical
        y = to_categorical(y)

        # dataset split (60-20-20) without stratified balancing
        X_train, X_test, y_train, y_test = train_test_split(
            images, y,
            test_size=test_size,
            random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=random_state
        )

        # free memory
        del images, labels
        gc.collect()

        # build and train model
        model = build_model((img_size[0], img_size[1], 1), len(unique_labels))
        model.summary()

        # callbacks (still allowed, they are not "optimizer tuning")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # evaluation
        print("\n=== Final Evaluation ===")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # get actual classes present in the test set
        actual_classes_in_test = np.unique(y_true_classes)
        actual_labels_in_test = [unique_labels[i] for i in actual_classes_in_test]

        print("\nClassification Report:")
        print(classification_report(
            y_true_classes,
            y_pred_classes,
            labels=actual_classes_in_test,
            target_names=actual_labels_in_test
        ))

        # figure 1: accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        # figure 2: loss
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        # figure 3: precision, recall, f1 score
        plt.figure(figsize=(8, 5))
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [
            precision_score(y_true_classes, y_pred_classes, average='weighted'),
            recall_score(y_true_classes, y_pred_classes, average='weighted'),
            f1_score(y_true_classes, y_pred_classes, average='weighted')
        ]
        plt.bar(metrics, values)
        plt.title('Model Metrics on Test Set')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
