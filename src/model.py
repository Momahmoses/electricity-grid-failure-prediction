"""Train and evaluate failure prediction model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURE_COLS, TARGET_COL, RF_PARAMS, OUTPUTS_DIR


def train(gdf):
    X = gdf[FEATURE_COLS].values
    y = gdf[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_ts_s = scaler.transform(X_test)
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr_s, y_train)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(OUTPUTS_DIR, "grid_failure_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "scaler.pkl"))
    return clf, scaler, X_ts_s, y_test, gdf.iloc[len(X_train):]


def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Grid Failure Prediction")
    axes[0].legend()

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    axes[1].barh([FEATURE_COLS[i] for i in idx], importances[idx], color="coral")
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Feature Importances")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "model_evaluation.png"), dpi=150)
    plt.close()
    return y_prob
