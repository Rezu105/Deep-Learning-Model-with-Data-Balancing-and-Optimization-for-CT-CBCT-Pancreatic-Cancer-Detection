import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import BorderlineSMOTE   # ✅ Using Borderline-SMOTE (SMOT equivalent)
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Configuration
# ---------------------------
dataset_path = r"E:\Work folder\AIUB\Reasearch Folder\Thesis_G42\PancreasProject\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
img_size = (128, 128)
max_images = 12000
random_state = 42
batch_size = 16
epochs = 30

np.random.seed(random_state)
tf.random.set_seed(random_state)

# ---------------------------
# Load dataset
# ---------------------------
images, labels = [], []
count = 0
print(f"[INFO] Loading up to {max_images} images...")

for root, dirs, files in os.walk(dataset_path):
    class_name = os.path.basename(root)
    dcm_files = [f for f in files if f.lower().endswith('.dcm')]
    
    if len(dcm_files) < 1:
        continue

    for file in tqdm(dcm_files, desc=f"Loading {class_name}"):
        if count >= max_images:
            break
        try:
            import pydicom
            dcm = pydicom.dcmread(os.path.join(root, file), force=True)
            img = dcm.pixel_array
            if len(img.shape) == 3:
                img = img[img.shape[0] // 2]
            img = img.astype(np.float32)
            if hasattr(dcm, 'RescaleSlope'):
                img = img * float(dcm.RescaleSlope)
            if hasattr(dcm, 'RescaleIntercept'):
                img = img + float(dcm.RescaleIntercept)
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(class_name)
            count += 1
        except:
            continue

images = np.array(images, dtype=np.float32)
images = np.expand_dims(images, axis=-1)
labels = np.array(labels)
print(f"Total loaded images: {len(images)}")

# ---------------------------
# Filter classes with < 2 samples
# ---------------------------
class_counts = Counter(labels)
valid_classes = [cls for cls, c in class_counts.items() if c >= 2]
mask = np.isin(labels, valid_classes)
images = images[mask]
labels = labels[mask]

# ---------------------------
# Encode labels
# ---------------------------
le = LabelEncoder()
y_enc = le.fit_transform(labels)

# ---------------------------
# Apply Borderline-SMOTE (SMOT)
# ---------------------------
print("[INFO] Applying Borderline-SMOTE (SMOT) for data balancing...")
X_2d = images.reshape(len(images), -1)
smot = BorderlineSMOTE(random_state=random_state, kind='borderline-1')
X_res, y_res = smot.fit_resample(X_2d, y_enc)

print(f"Before balancing: {Counter(y_enc)}")
print(f"After SMOT       : {Counter(y_res)}")

# Reshape back
X_res = X_res.reshape(-1, img_size[0], img_size[1], 1)
y_res = to_categorical(y_res)

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=random_state, stratify=y_res
)

# ---------------------------
# Build EfficientNetB0 model
# ---------------------------
input_tensor = Input(shape=(img_size[0], img_size[1], 1))
base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=input_tensor)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(y_res.shape[1], activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# ---------------------------
# Evaluate
# ---------------------------
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
loss, _ = model.evaluate(X_test, y_test, verbose=0)

print("\n=== Final Metrics on Test Set ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Loss     : {loss:.4f}")
