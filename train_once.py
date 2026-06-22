import os, warnings, numpy as np
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout,
                                     Dense, Flatten, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.random.set_seed(42)

CLASS_NAMES = ["Airplane","Automobile","Bird","Cat","Deer",
               "Dog","Frog","Horse","Ship","Truck"]

print("="*50)
print("DATA LOAD HO RAHA HAI...")
print("="*50)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0
y_train_ohe = to_categorical(y_train, 10)
y_test_ohe  = to_categorical(y_test,  10)
print(f"Training: {X_train.shape[0]:,} images")
print(f"Testing:  {X_test.shape[0]:,} images")

# ── Augmentation manually karo ────────────────
# ImageDataGenerator ki jagah numpy se karo
# Windows pe yeh zyada stable hai
def augment_batch(images, labels):
    aug_imgs = []
    aug_lbls = []
    for img, lbl in zip(images, labels):
        aug_imgs.append(img)
        aug_lbls.append(lbl)
        # Horizontal flip
        aug_imgs.append(img[:, ::-1, :])
        aug_lbls.append(lbl)
        # Slight rotation simulation (shift)
        shifted = np.roll(img, 2, axis=0)
        aug_imgs.append(shifted)
        aug_lbls.append(lbl)
    return np.array(aug_imgs), np.array(aug_lbls)

print("Augmented data bana raha hai... (2-3 minute)")
X_aug, y_aug = augment_batch(X_train, y_train_ohe)
print(f"Augmented training size: {len(X_aug):,}")

def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), padding="same",
               activation="relu", input_shape=(32,32,3)),
        BatchNormalization(),
        Conv2D(32, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.30),

        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.40),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.50),
        Dense(128, activation="relu"),
        Dropout(0.30),
        Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1)
]

# ── PLAIN CNN ─────────────────────────────────
print("\n" + "="*50)
print("CNN PLAIN TRAINING SHURU...")
print("="*50)
cnn_plain = build_cnn()
history_plain = cnn_plain.fit(
    X_train, y_train_ohe,
    validation_data=(X_test, y_test_ohe),
    epochs=40, batch_size=64,
    callbacks=callbacks, verbose=1
)
cnn_plain.save("outputs/model_plain.keras")
np.save("outputs/history_plain.npy", history_plain.history)
print("Plain model save ho gaya!")

# ── AUGMENTED CNN ─────────────────────────────
print("\n" + "="*50)
print("CNN AUGMENTED TRAINING SHURU...")
print("="*50)
cnn_aug = build_cnn()
history_aug = cnn_aug.fit(
    X_aug, y_aug,
    validation_data=(X_test, y_test_ohe),
    epochs=40, batch_size=64,
    callbacks=callbacks, verbose=1
)
cnn_aug.save("outputs/model_aug.keras")
cnn_aug.save("outputs/final_model.keras")
np.save("outputs/history_aug.npy", history_aug.history)
print("Augmented model save ho gaya!")

# ── RESULTS ───────────────────────────────────
print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
loss_p, acc_p = cnn_plain.evaluate(
    X_test, y_test_ohe, verbose=0)
loss_a, acc_a = cnn_aug.evaluate(
    X_test, y_test_ohe, verbose=0)
print(f"CNN Plain    : {acc_p*100:.2f}%")
print(f"CNN Aug      : {acc_a*100:.2f}%")
diff = (acc_a - acc_p) * 100
if diff > 0:
    print(f"Improvement  : +{diff:.2f}%")
else:
    print(f"Difference   : {diff:.2f}%")
print("\nAb gui_predictor.py chalao!")