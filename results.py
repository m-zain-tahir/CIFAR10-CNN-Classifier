import os, warnings, numpy as np
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

CLASS_NAMES = ["Airplane","Automobile","Bird","Cat","Deer",
               "Dog","Frog","Horse","Ship","Truck"]

print("="*50)
print("MODELS CHECK HO RAHE HAIN...")
print("="*50)

if not os.path.exists("outputs/final_model.keras"):
    print("ERROR: final_model.keras nahi mila!")
    print("Pehle train_once.py chalao")
    exit()

if not os.path.exists("outputs/best_cnn_aug.keras"):
    print("ERROR: best_cnn_aug.keras nahi mila!")
    print("Pehle train_once.py chalao")
    exit()

print("Dono models mil gaye!")

print("\nTest data load ho raha hai...")
(_, _), (X_test, y_test) = cifar10.load_data()
y_test = y_test.flatten()
X_test = X_test.astype("float32") / 255.0
y_test_ohe = to_categorical(y_test, 10)
print(f"{len(X_test):,} test images ready!")

print("\nModels load ho rahe hain...")
model_plain = load_model("outputs/final_model.keras")
model_aug   = load_model("outputs/best_cnn_aug.keras")
print("Dono models load ho gaye!")

def evaluate_model(model, name):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    loss, acc = model.evaluate(X_test, y_test_ohe, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    prec = precision_score(y_test, y_pred, average="macro")
    rec  = recall_score(y_test, y_pred, average="macro")
    f1   = f1_score(y_test, y_pred, average="macro")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    return acc, prec, rec, f1, y_pred

acc_p, pre_p, rec_p, f1_p, pred_p = evaluate_model(
    model_plain, "CNN Without Augmentation")

acc_a, pre_a, rec_a, f1_a, pred_a = evaluate_model(
    model_aug, "CNN With Augmentation")

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set