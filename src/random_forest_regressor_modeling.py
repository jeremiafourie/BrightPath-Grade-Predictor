# Train a RandomForestRegressor, round predictions to classes, evaluate and save.

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def plot_confusion_heatmap(cm, classes, title):
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt='d',
        cmap='Blues',
        linewidths=0.5,
        linecolor='gray',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    setup_logging()

    # 1) Load engineered data
    data_path = os.path.join('data', 'processed', 'engineered_data.csv')
    if not os.path.exists(data_path):
        logging.error(f"Data not found: {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)

    # 2) Drop GPA if exists
    if 'GPA' in df.columns:
        df.drop(columns=['GPA'], inplace=True)

    # 3) One‑hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        logging.info(f"One‑hot encoding columns: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4) Split features/target
    X = df.drop(columns=['GradeClass'])
    y = df['GradeClass']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5) Train & evaluate regressor
    name = 'RandomForestRegressor'
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    logging.info(f"Training {name}")
    model.fit(X_train, y_train)

    # 6) Round predictions to nearest class
    preds = np.round(model.predict(X_test)).astype(int)

    # 7) Evaluate
    acc = accuracy_score(y_test, preds)
    logging.info(f"{name}→Classifier Accuracy: {acc:.3f}")
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds)
    print(f"=== {name} Confusion Matrix (counts) ===\n{cm}")
    plot_confusion_heatmap(cm, np.unique(y_test), f"{name} Confusion Matrix")

    # 8) Save model
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    save_path = os.path.join(artifacts_dir, f"{name}.joblib")
    joblib.dump(model, save_path)
    logging.info(f"Saved {name} to {save_path}")

if __name__ == '__main__':
    main()
