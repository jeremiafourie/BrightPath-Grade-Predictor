# Train a simple MLP on engineered data, evaluate and save the model.

import os
import sys
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

    # 4) Prepare features & target
    X = df.drop(columns=['GradeClass']).values
    y = df['GradeClass'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5) Build MLP
    name = 'NeuralNetworkMLP'
    logging.info(f"Building & training {name}")
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = models.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6) Train
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=25,
        batch_size=32,
        verbose=1
    )

    # 7) Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"{name} Test Accuracy: {acc:.3f}")

    # 8) Report & confusion matrix
    probs = model.predict(X_test)
    y_pred = np.argmax(probs, axis=1)
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print(f"=== {name} Confusion Matrix (counts) ===\n{cm}")
    plot_confusion_heatmap(cm, np.unique(y_test), f"{name} Confusion Matrix")

    # 9) Save model
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    save_path = os.path.join(artifacts_dir, f"{name}.keras")
    model.save(save_path)
    logging.info(f"Saved {name} to {save_path}")

if __name__ == '__main__':
    main()
