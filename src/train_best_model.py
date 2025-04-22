# train_best_model.py
# Script to train the final scaled logistic regression model and save it to artifacts/

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def plot_confusion_heatmap(cm, classes, title, save_path=None):
    """
    Plot a normalized confusion matrix heatmap with raw counts annotated.
    If save_path is provided, save the figure there.
    """
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
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    setup_logging()
    model_name = 'ScaledLogisticRegression'
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)

    # Paths
    data_path = os.path.join('data', 'processed', 'engineered_data.csv')
    model_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
    cm_plot_path = os.path.join(artifacts_dir, f"{model_name}_confusion_matrix.png")

    # Load data
    if not os.path.exists(data_path):
        logging.error(f"Engineered data not found at {data_path}")
        sys.exit(1)
    logging.info(f"Loading engineered data from {data_path}")
    df = pd.read_csv(data_path)

    # Drop raw GPA if exists
    if 'GPA' in df.columns:
        df.drop(columns=['GPA'], inplace=True)

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        logging.info(f"One-hot encoding columns: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Split features/target
    X = df.drop(columns=['GradeClass'])
    y = df['GradeClass']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logging.info(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}")

    # Build pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train
    logging.info(f"Training {model_name}")
    pipe.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"{model_name} Accuracy: {acc:.4f}")

    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_heatmap(
        cm,
        classes=pipe.named_steps['lr'].classes_,
        title=f"{model_name} Confusion Matrix",
        save_path=cm_plot_path
    )

    # Save model
    joblib.dump(pipe, model_path)
    logging.info(f"Saved trained model to {model_path}")

if __name__ == '__main__':
    main()
