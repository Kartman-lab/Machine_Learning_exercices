import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.metrics import roc_curve, auc

def visualise_confusion_matrix(cm, path_name):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs réelles")
    plt.title("Matrice de confusion")
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    plt.savefig(path_name)
    plt.show()

def visualize_roc_curve(log, X_test, y_test, path_name):
    y_proba = log.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend()
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    plt.savefig(path_name)
    plt.show()
