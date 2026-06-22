# ============================================================
# Performance Enhancement in Image Classification Using
# Advanced Data Augmentation Techniques
# ------------------------------------------------------------
# Authors  : Zain Tahir | Huzaifa Khan
# Dataset  : CIFAR-10
# Models   : KNN  |  CNN (with / without augmentation)
# Framework: TensorFlow / Keras  |  scikit-learn
# ============================================================

# ─────────────────────────────────────────────
# SECTION 0 — Library Imports
# ─────────────────────────────────────────────
import os, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout,
                                     Dense, Flatten, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.utils import to_categorical

# ── Reproducibility ──────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ── CIFAR-10 class labels ─────────────────────
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# ─────────────────────────────────────────────
# SECTION 1 — Dataset Loading & Preprocessing
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  LOADING CIFAR-10 DATASET")
print("="*60)

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar10.load_data()

y_train_raw = y_train_raw.flatten()
y_test_raw  = y_test_raw.flatten()

print(f"  Training samples : {X_train_raw.shape[0]:,}")
print(f"  Test samples     : {X_test_raw.shape[0]:,}")
print(f"  Image dimensions : {X_train_raw.shape[1:]} (H × W × C)")
print(f"  Number of classes: {len(CLASS_NAMES)}")
print(f"  Classes          : {CLASS_NAMES}")

# Normalise pixel values to [0, 1]
X_train = X_train_raw.astype("float32") / 255.0
X_test  = X_test_raw.astype("float32")  / 255.0

# One-hot encode labels for CNN
y_train_ohe = to_categorical(y_train_raw, 10)
y_test_ohe  = to_categorical(y_test_raw,  10)

# ─────────────────────────────────────────────
# SECTION 2 — Dataset Visualisation
# ─────────────────────────────────────────────
def plot_sample_images(X, y, title="CIFAR-10 Sample Images", save_path=None):
    """Display one random sample per class in a 2×5 grid."""
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("#0f0f23")

    indices = [np.where(y == c)[0][0] for c in range(10)]
    for ax, idx, name in zip(axes.flat, indices, CLASS_NAMES):
        ax.imshow(X[idx])
        ax.set_title(name, color="white", fontsize=10, fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#4ade80")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
    plt.close()

def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """Bar chart showing sample counts per class."""
    counts = [np.sum(y == c) for c in range(10)]
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(CLASS_NAMES, counts,
                  color=plt.cm.plasma(np.linspace(0.2, 0.9, 10)),
                  edgecolor="white", linewidth=0.8, width=0.65)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                f"{count:,}", ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Class", color="white", fontsize=11)
    ax.set_ylabel("Number of Samples", color="white", fontsize=11)
    ax.tick_params(colors="white", rotation=15)
    ax.spines[:].set_color("#444")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
    plt.close()

plot_sample_images(X_train, y_train_raw, save_path="outputs/fig1_sample_images.png")
plot_class_distribution(y_train_raw, "Training Set — Class Distribution",
                        save_path="outputs/fig2_class_distribution.png")

# ─────────────────────────────────────────────
# SECTION 3 — Data Augmentation Pipeline
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  CONFIGURING DATA AUGMENTATION PIPELINE")
print("="*60)

# Without augmentation — only normalisation
datagen_plain = ImageDataGenerator()

# With advanced augmentation
datagen_aug = ImageDataGenerator(
    rotation_range=20,           # Random rotation ±20°
    width_shift_range=0.15,      # Horizontal translation
    height_shift_range=0.15,     # Vertical translation
    horizontal_flip=True,        # Mirror flip
    zoom_range=0.15,             # Zoom in/out
    brightness_range=[0.8, 1.2], # Brightness variation
    shear_range=0.1,             # Shear transformation
    fill_mode="nearest"          # Fill strategy for empty pixels
)

datagen_aug.fit(X_train)

def add_gaussian_noise(images, mean=0.0, std=0.02):
    """Inject Gaussian noise — augmentation not natively in Keras generator."""
    noisy = images + np.random.normal(mean, std, images.shape)
    return np.clip(noisy, 0.0, 1.0)

def plot_augmentation_comparison(X, y, save_path=None):
    """Show original vs augmented versions of the same image."""
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    fig.suptitle("Data Augmentation — Original vs Augmented Samples",
                 fontsize=14, fontweight="bold", color="white")
    fig.patch.set_facecolor("#0f0f23")

    sample_idx = [np.where(y == c)[0][0] for c in range(6)]

    for col, idx in enumerate(sample_idx):
        orig = X[idx]
        aug  = next(datagen_aug.flow(
            orig[np.newaxis, ...], batch_size=1))[0]
        aug  = np.clip(aug, 0, 1)

        axes[0, col].imshow(orig)
        axes[0, col].set_title(CLASS_NAMES[y[idx]], color="#4ade80",
                               fontsize=9, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(aug)
        axes[1, col].set_title("Augmented", color="#fb923c",
                               fontsize=9)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", color="white", fontsize=10,
                           rotation=90, labelpad=8)
    axes[1, 0].set_ylabel("Augmented", color="white", fontsize=10,
                           rotation=90, labelpad=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
    plt.close()

plot_augmentation_comparison(X_train, y_train_raw,
                             save_path="outputs/fig3_augmentation_comparison.png")

# ─────────────────────────────────────────────
# SECTION 4 — KNN Classifier
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  KNN CLASSIFIER — Training & Evaluation")
print("="*60)

# Use a manageable subset for KNN (full 50k is computationally expensive)
KNN_TRAIN_SIZE = 10_000
KNN_TEST_SIZE  = 2_000

X_knn_train = X_train[:KNN_TRAIN_SIZE].reshape(KNN_TRAIN_SIZE, -1)
X_knn_test  = X_test[:KNN_TEST_SIZE].reshape(KNN_TEST_SIZE, -1)
y_knn_train = y_train_raw[:KNN_TRAIN_SIZE]
y_knn_test  = y_test_raw[:KNN_TEST_SIZE]

# PCA dimensionality reduction (3072 → 150 components)
print("  Applying PCA dimensionality reduction (3072 → 150) ...")
pca = PCA(n_components=150, random_state=SEED)
X_knn_train_pca = pca.fit_transform(X_knn_train)
X_knn_test_pca  = pca.transform(X_knn_test)
print(f"  Explained variance retained: "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

# Standardise
scaler = StandardScaler()
X_knn_train_sc = scaler.fit_transform(X_knn_train_pca)
X_knn_test_sc  = scaler.transform(X_knn_test_pca)

# ── KNN — WITHOUT augmentation ─────────────
print("\n  [KNN] Training WITHOUT augmentation ...")
knn_plain = KNeighborsClassifier(n_neighbors=7, metric="euclidean",
                                  n_jobs=-1, weights="distance")
knn_plain.fit(X_knn_train_sc, y_knn_train)
y_pred_knn_plain = knn_plain.predict(X_knn_test_sc)

knn_acc_plain = accuracy_score(y_knn_test, y_pred_knn_plain)
knn_pre_plain = precision_score(y_knn_test, y_pred_knn_plain, average="macro")
knn_rec_plain = recall_score(y_knn_test, y_pred_knn_plain, average="macro")
knn_f1_plain  = f1_score(y_knn_test, y_pred_knn_plain, average="macro")

print(f"  Accuracy  : {knn_acc_plain*100:.2f}%")
print(f"  Precision : {knn_pre_plain*100:.2f}%")
print(f"  Recall    : {knn_rec_plain*100:.2f}%")
print(f"  F1-Score  : {knn_f1_plain*100:.2f}%")

# ── KNN — WITH augmentation (synthetic) ───
print("\n  [KNN] Generating augmented training samples ...")

# Generate 3 augmented copies per training image
aug_images, aug_labels = [], []
aug_flow = datagen_aug.flow(
    X_train[:KNN_TRAIN_SIZE], y_knn_train,
    batch_size=512, shuffle=False)
for _ in range(3):
    for xb, yb in aug_flow:
        aug_images.append(xb)
        aug_labels.append(yb)
        if len(aug_images) * 512 >= KNN_TRAIN_SIZE * 3:
            break

X_aug_arr = np.concatenate(aug_images, axis=0)[:KNN_TRAIN_SIZE * 3]
y_aug_arr = np.concatenate(aug_labels, axis=0)[:KNN_TRAIN_SIZE * 3]

X_aug_flat = X_aug_arr.reshape(len(X_aug_arr), -1)
X_aug_pca  = pca.transform(X_aug_flat)
X_aug_sc   = scaler.transform(X_aug_pca)

X_knn_train_aug = np.vstack([X_knn_train_sc, X_aug_sc])
y_knn_train_aug = np.concatenate([y_knn_train, y_aug_arr])

print(f"  Total training samples after augmentation: {len(y_knn_train_aug):,}")

knn_aug = KNeighborsClassifier(n_neighbors=7, metric="euclidean",
                                n_jobs=-1, weights="distance")
knn_aug.fit(X_knn_train_aug, y_knn_train_aug)
y_pred_knn_aug = knn_aug.predict(X_knn_test_sc)

knn_acc_aug = accuracy_score(y_knn_test, y_pred_knn_aug)
knn_pre_aug = precision_score(y_knn_test, y_pred_knn_aug, average="macro")
knn_rec_aug = recall_score(y_knn_test, y_pred_knn_aug, average="macro")
knn_f1_aug  = f1_score(y_knn_test, y_pred_knn_aug, average="macro")

print(f"  Accuracy  : {knn_acc_aug*100:.2f}%")
print(f"  Precision : {knn_pre_aug*100:.2f}%")
print(f"  Recall    : {knn_rec_aug*100:.2f}%")
print(f"  F1-Score  : {knn_f1_aug*100:.2f}%")

print("\n  Classification Report (KNN + Augmentation):")
print(classification_report(y_knn_test, y_pred_knn_aug,
                             target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# SECTION 5 — CNN Architecture
# ─────────────────────────────────────────────
def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Custom CNN architecture — 3 convolutional blocks followed by
    fully-connected classification head.
    """
    model = Sequential([
        # ── Block 1 ──────────────────────────
        Conv2D(32, (3, 3), padding="same", activation="relu",
               input_shape=input_shape, name="conv1_1"),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2"),
        BatchNormalization(),
        MaxPooling2D((2, 2), name="pool1"),
        Dropout(0.25, name="drop1"),

        # ── Block 2 ──────────────────────────
        Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2"),
        BatchNormalization(),
        MaxPooling2D((2, 2), name="pool2"),
        Dropout(0.30, name="drop2"),

        # ── Block 3 ──────────────────────────
        Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2"),
        BatchNormalization(),
        MaxPooling2D((2, 2), name="pool3"),
        Dropout(0.40, name="drop3"),

        # ── Classification Head ───────────────
        Flatten(name="flatten"),
        Dense(256, activation="relu", name="fc1"),
        BatchNormalization(),
        Dropout(0.50, name="drop4"),
        Dense(128, activation="relu", name="fc2"),
        Dropout(0.30, name="drop5"),
        Dense(num_classes, activation="softmax", name="output"),
    ], name="CIFAR10_CNN")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

cnn = build_cnn()
cnn.summary()

# ─────────────────────────────────────────────
# SECTION 6 — CNN Training
# ─────────────────────────────────────────────
EPOCHS    = 60
BATCH     = 64

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=12,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint("outputs/best_cnn_aug.keras",
                    monitor="val_accuracy", save_best_only=True, verbose=0)
]

# ── CNN — WITHOUT augmentation ────────────
print("\n" + "="*60)
print("  CNN — Training WITHOUT Augmentation")
print("="*60)
cnn_plain = build_cnn()
hist_plain = cnn_plain.fit(
    X_train, y_train_ohe,
    validation_data=(X_test, y_test_ohe),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=callbacks[:2],   # No checkpoint for plain
    verbose=1
)

# ── CNN — WITH augmentation ───────────────
print("\n" + "="*60)
print("  CNN — Training WITH Augmentation")
print("="*60)
cnn_aug = build_cnn()
hist_aug = cnn_aug.fit(
    datagen_aug.flow(X_train, y_train_ohe, batch_size=BATCH),
    steps_per_epoch=len(X_train) // BATCH,
    validation_data=(X_test, y_test_ohe),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# SECTION 7 — CNN Evaluation
# ─────────────────────────────────────────────
def evaluate_cnn(model, X, y_ohe, y_true, tag):
    loss, acc = model.evaluate(X, y_ohe, verbose=0)
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")
    f1   = f1_score(y_true, y_pred, average="macro")
    print(f"\n  [{tag}]")
    print(f"  Test Loss      : {loss:.4f}")
    print(f"  Test Accuracy  : {acc*100:.2f}%")
    print(f"  Precision      : {prec*100:.2f}%")
    print(f"  Recall         : {rec*100:.2f}%")
    print(f"  F1-Score       : {f1*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return acc, prec, rec, f1, y_pred

print("\n" + "="*60)
print("  CNN EVALUATION RESULTS")
print("="*60)

acc_p, pre_p, rec_p, f1_p, y_pred_plain = evaluate_cnn(
    cnn_plain, X_test, y_test_ohe, y_test_raw, "CNN Without Augmentation")

acc_a, pre_a, rec_a, f1_a, y_pred_aug = evaluate_cnn(
    cnn_aug, X_test, y_test_ohe, y_test_raw, "CNN With Augmentation")

# ─────────────────────────────────────────────
# SECTION 8 — Visualisations
# ─────────────────────────────────────────────

def plot_training_curves(hist_plain, hist_aug, save_path=None):
    """Dual-panel accuracy and loss curves comparing both CNN variants."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f0f23")
    colors = {"plain": "#f87171", "aug": "#4ade80",
              "val_plain": "#fb923c", "val_aug": "#60a5fa"}

    for ax, metric, title in zip(
            axes,
            [("accuracy", "val_accuracy"), ("loss", "val_loss")],
            ["Model Accuracy", "Model Loss"]):
        ax.set_facecolor("#1a1a2e")
        ax.plot(hist_plain.history[metric[0]], color=colors["plain"],
                lw=2, label="Train (No Aug)")
        ax.plot(hist_plain.history[metric[1]], color=colors["val_plain"],
                lw=2, linestyle="--", label="Val (No Aug)")
        ax.plot(hist_aug.history[metric[0]], color=colors["aug"],
                lw=2, label="Train (With Aug)")
        ax.plot(hist_aug.history[metric[1]], color=colors["val_aug"],
                lw=2, linestyle="--", label="Val (With Aug)")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel(metric[0].capitalize(), color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.legend(facecolor="#0f0f23", edgecolor="#444", labelcolor="white")

    plt.suptitle("Training Dynamics — CNN Without vs With Augmentation",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show(); plt.close()


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#0f0f23")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5, linecolor="#333",
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax.set_ylabel("True Label", color="white", fontsize=11)
    ax.tick_params(colors="white", rotation=45)
    ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show(); plt.close()


def plot_model_comparison(save_path=None):
    """Grouped bar chart comparing all models across 4 metrics."""
    models   = ["KNN (Plain)", "KNN (Aug)", "CNN (Plain)", "CNN (Aug)"]
    accuracy  = [knn_acc_plain, knn_acc_aug, acc_p, acc_a]
    precision = [knn_pre_plain, knn_pre_aug, pre_p, pre_a]
    recall_s  = [knn_rec_plain, knn_rec_aug, rec_p, rec_a]
    f1_scores = [knn_f1_plain,  knn_f1_aug,  f1_p,  f1_a]

    x     = np.arange(len(models))
    width = 0.20
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")

    b1 = ax.bar(x - 1.5*width, accuracy,  width, label="Accuracy",
                color="#4ade80", edgecolor="white", lw=0.6)
    b2 = ax.bar(x - 0.5*width, precision, width, label="Precision",
                color="#60a5fa", edgecolor="white", lw=0.6)
    b3 = ax.bar(x + 0.5*width, recall_s,  width, label="Recall",
                color="#f87171", edgecolor="white", lw=0.6)
    b4 = ax.bar(x + 1.5*width, f1_scores, width, label="F1-Score",
                color="#fb923c", edgecolor="white", lw=0.6)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h*100:.1f}", ha="center", va="bottom",
                    color="white", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(models, color="white", fontsize=11)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.05, 0.1)],
                       color="white")
    ax.set_ylim(0, 1.10)
    ax.set_title("Model Performance Comparison — All Metrics",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#0f0f23", edgecolor="#444", labelcolor="white",
              fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show(); plt.close()


# ── Generate all figures ──────────────────
os.makedirs("outputs", exist_ok=True)

plot_training_curves(hist_plain, hist_aug,
                     save_path="outputs/fig4_training_curves.png")

plot_confusion_matrix(y_test_raw, y_pred_plain,
                      "Confusion Matrix — CNN Without Augmentation",
                      save_path="outputs/fig5_cm_plain.png")

plot_confusion_matrix(y_test_raw, y_pred_aug,
                      "Confusion Matrix — CNN With Augmentation",
                      save_path="outputs/fig6_cm_aug.png")

plot_model_comparison(save_path="outputs/fig7_model_comparison.png")

# ─────────────────────────────────────────────
# SECTION 9 — Summary Table
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL PERFORMANCE SUMMARY")
print("="*60)
print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>10} {'F1-Score':>10}")
print("  " + "-"*62)
rows = [
    ("KNN (No Augmentation)", knn_acc_plain, knn_pre_plain,
     knn_rec_plain, knn_f1_plain),
    ("KNN (With Augmentation)", knn_acc_aug,  knn_pre_aug,
     knn_rec_aug,  knn_f1_aug),
    ("CNN (No Augmentation)", acc_p, pre_p, rec_p, f1_p),
    ("CNN (With Augmentation)", acc_a, pre_a, rec_a, f1_a),
]
for name, acc, pre, rec, f1 in rows:
    print(f"  {name:<22} {acc*100:>9.2f}% {pre*100:>9.2f}% "
          f"{rec*100:>9.2f}% {f1*100:>9.2f}%")

# Save best model
cnn_aug.save("outputs/final_model.keras")
print("\n  Best model saved → outputs/final_model.keras")
print("  All figures saved → outputs/")
print("\n  Project execution complete.")
