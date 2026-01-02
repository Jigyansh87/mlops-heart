# ============================================================
# TRAINING SCRIPT: HEART DISEASE CLASSIFICATION
# ============================================================
# Responsibilities:
# - Load cleaned data
# - Feature engineering (scaling)
# - Train Logistic Regression & Random Forest
# - Evaluate with multiple metrics
# - Generate & save plots
# - Track experiments using MLflow
# ============================================================


# =========================
# 1. IMPORTS
# =========================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================
# 2. PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "heart_cleaned.csv")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# =========================
# 3. MLFLOW SETUP
# =========================

mlflow.set_experiment("Heart Disease Classification")


# =========================
# 4. DATA LOADING
# =========================

df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]


# =========================
# 5. TRAIN–TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 6. FEATURE ENGINEERING
# =========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 7. MODEL DEFINITIONS
# =========================

log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)


# =========================
# 8. START MLFLOW RUN
# =========================

with mlflow.start_run():

    # ---------- PARAMETERS ----------
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("log_reg_max_iter", 1000)
    mlflow.log_param("rf_n_estimators", 100)
    mlflow.log_param("cv_folds", 5)

    # ---------- CROSS-VALIDATION ----------
    cv_scores_lr = cross_val_score(
        log_reg, X_train_scaled, y_train, cv=5, scoring="roc_auc"
    )
    cv_scores_rf = cross_val_score(
        rf, X_train, y_train, cv=5, scoring="roc_auc"
    )

    # ---------- TRAINING ----------
    log_reg.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)

    # ---------- SAVE MODEL & PREPROCESSOR (REPRODUCIBILITY) ----------

    MODEL_DIR = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))


    # ---------- PREDICTIONS ----------
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    # ---------- METRICS ----------
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    lr_precision = precision_score(y_test, y_pred_lr, zero_division=0)
    rf_precision = precision_score(y_test, y_pred_rf, zero_division=0)

    lr_recall = recall_score(y_test, y_pred_lr)
    rf_recall = recall_score(y_test, y_pred_rf)

    lr_auc = roc_auc_score(y_test, y_prob_lr)
    rf_auc = roc_auc_score(y_test, y_prob_rf)

    # Log metrics
    mlflow.log_metric("lr_accuracy", lr_accuracy)
    mlflow.log_metric("lr_precision", lr_precision)
    mlflow.log_metric("lr_recall", lr_recall)
    mlflow.log_metric("lr_roc_auc", lr_auc)
    mlflow.log_metric("lr_cv_roc_auc", cv_scores_lr.mean())

    mlflow.log_metric("rf_accuracy", rf_accuracy)
    mlflow.log_metric("rf_precision", rf_precision)
    mlflow.log_metric("rf_recall", rf_recall)
    mlflow.log_metric("rf_roc_auc", rf_auc)
    mlflow.log_metric("rf_cv_roc_auc", cv_scores_rf.mean())

    # ---------- RESULTS TABLE ----------
    results_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [lr_accuracy, rf_accuracy],
        "Precision": [lr_precision, rf_precision],
        "Recall": [lr_recall, rf_recall],
        "ROC-AUC": [lr_auc, rf_auc],
        "CV ROC-AUC Mean": [cv_scores_lr.mean(), cv_scores_rf.mean()]
    })

    print("\nModel comparison summary:")
    print(results_df)

    # ---------- ROC CURVE ----------
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={lr_auc:.3f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)

    roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # ---------- METRIC BAR CHART ----------
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "ROC-AUC"]].plot(
        kind="bar", figsize=(10, 6)
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y")

    bar_path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(bar_path)
    plt.close()

    # ---------- CONFUSION MATRIX ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred_lr),
        display_labels=["No Disease", "Disease"]
    ).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Logistic Regression")

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred_rf),
        display_labels=["No Disease", "Disease"]
    ).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Random Forest")

    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # ---------- LOG ARTIFACTS ----------
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(bar_path)
    mlflow.log_artifact(cm_path)

    # ---------- LOG FINAL MODEL ----------
    mlflow.sklearn.log_model(
        rf,
        artifact_path="model",
        registered_model_name="HeartDiseaseRandomForest"
    )


# =========================
# END OF SCRIPT
# =========================
