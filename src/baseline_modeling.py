# Train and evaluate baseline classification models on engineered data.

import os
import sys
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

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

def evaluate_and_save(model, name, X_train, X_test, y_train, y_test, artifacts_dir='artifacts'):
    os.makedirs(artifacts_dir, exist_ok=True)
    logging.info(f"Training & evaluating {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info(f"{name} Accuracy: {acc:.3f}")

    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, preds, zero_division=0))

    cm = confusion_matrix(y_test, preds)
    print(f"=== {name} Confusion Matrix (counts) ===\n{cm}")
    plot_confusion_heatmap(cm, model.classes_, f"{name} Confusion Matrix")

    save_path = os.path.join(artifacts_dir, f"{name}.joblib")
    joblib.dump(model, save_path)
    logging.info(f"Saved {name} to {save_path}\n")

def main():
    setup_logging()

    # 1) Load engineered data
    data_path = os.path.join('data', 'processed', 'engineered_data.csv')
    if not os.path.exists(data_path):
        logging.error(f"Missing engineered data at {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)

    # 2) Drop GPA if present
    if 'GPA' in df.columns:
        df.drop(columns=['GPA'], inplace=True)

    # 3) One‑hot encode any categorical columns
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

    # 5) Define and run baseline models
    models = {
        'LogisticRegression': LogisticRegression(
            C=1.0, solver='liblinear', multi_class='ovr',
            max_iter=500, random_state=42
        ),
        'RandomForestClassifier': RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight='balanced', random_state=42
        ),
        'XGBClassifier': XGBClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', eval_metric='mlogloss',
            use_label_encoder=False, random_state=42
        )
    }

    for name, mdl in models.items():
        evaluate_and_save(mdl, name, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
