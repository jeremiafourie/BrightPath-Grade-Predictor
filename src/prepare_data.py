# Script to load and perform initial data preparation and diagnostic checks on the student dataset.

import os
import pandas as pd
import numpy as np
import sys
import logging


def setup_logging():
    """
    Configure the logging format and level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    try:
        logging.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)


def main():
    setup_logging()

    # Define input and output paths relative to project root
    raw_path = os.path.join('data', 'raw', 'student_performance_data.csv')
    processed_dir = os.path.join('data', 'processed')
    output_path = os.path.join(processed_dir, 'cleaned_data.csv')

    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Load the raw data
    data = load_data(raw_path)

    # Drop StudentID if present
    if 'StudentID' in data.columns:
        logging.info("Dropping 'StudentID' column...")
        data = data.drop(columns=['StudentID'])

    # Initial inspection
    logging.info("Performing data.info() and describe()...")
    data.info()
    desc = data.describe().T
    print(desc)

    # Missing value and duplicate checks
    logging.info("Checking for missing values and duplicates...")
    missing = data.isnull().sum()
    dup_count = data.duplicated().sum()
    print("Missing values per column:\n", missing)
    logging.info(f"Duplicate rows found: {dup_count}")

    # Outlier detection via IQR
    logging.info("Detecting outliers via IQR method...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers[col] = int(((data[col] < lower) | (data[col] > upper)).sum())
    print("Outlier counts per column:\n", outliers)

    # Distribution diagnostics
    logging.info("Calculating skewness and kurtosis...")
    skew_kurt = data[numeric_cols].agg(['skew', 'kurtosis']).T.round(2)
    print("Skewness and Kurtosis:\n", skew_kurt)

    # Save cleaned data
    try:
        data.to_csv(output_path, index=False)
        logging.info(f"Cleaned data saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save cleaned data: {e}")
        sys.exit(1)

    logging.info("Data preparation completed successfully.")


if __name__ == '__main__':
    main()
