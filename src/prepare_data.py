# prepare_data.py
# Script to load and perform initial data preparation and diagnostic checks on the student dataset.

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


def initial_inspection(df):
    """
    Perform initial inspection: info and descriptive statistics.
    """
    try:
        logging.info("Performing initial data.info() check...")
        df.info()
        logging.info("data.info() completed.")

        logging.info("Generating descriptive statistics with data.describe()...")
        desc = df.describe().T
        print(desc)
        logging.info("data.describe() completed.")
    except Exception as e:
        logging.error(f"Error during initial inspection: {e}")


def missing_and_duplicates(df):
    """
    Check for missing values and duplicate rows.
    """
    try:
        logging.info("Checking for missing values...")
        missing = df.isnull().sum()
        pct_missing = (missing / len(df) * 100).round(2)
        missing_df = pd.concat([missing, pct_missing.rename('pct_missing')], axis=1)
        print("Missing values per column:")
        print(missing_df)

        logging.info("Checking for duplicate rows...")
        dup_count = df.duplicated().sum()
        logging.info(f"Duplicate rows found: {dup_count}")
    except Exception as e:
        logging.error(f"Error during missing/duplicate checks: {e}")


def outlier_detection(df):
    """
    Detect outliers using the IQR method for numeric features.
    """
    try:
        logging.info("Starting outlier detection via IQR method...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            count = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_summary.append((col, int(count), float(lower), float(upper)))

        outliers_df = pd.DataFrame(
            outlier_summary,
            columns=['feature', 'n_outliers', 'lower_bound', 'upper_bound']
        )
        print("Outlier summary:")
        print(outliers_df)
        logging.info("Outlier detection completed.")
    except Exception as e:
        logging.error(f"Error during outlier detection: {e}")


def distribution_diagnostics(df):
    """
    Compute skewness and kurtosis for numeric features.
    """
    try:
        logging.info("Computing skewness and kurtosis...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = pd.DataFrame({
            'skewness': df[numeric_cols].skew(),
            'kurtosis': df[numeric_cols].kurtosis()
        }).round(2)
        print("Skewness & Kurtosis:")
        print(stats)
        logging.info("Distribution diagnostics completed.")
    except Exception as e:
        logging.error(f"Error during distribution diagnostics: {e}")


def main():
    setup_logging()

    # Replace with your actual file path
    data_filepath = 'data/raw/student_performance_data.csv'

    # Load data
    data = load_data(data_filepath)

    # Drop unnecessary columns (e.g., StudentID)
    if 'StudentID' in data.columns:
        logging.info("Dropping 'StudentID' column...")
        data = data.drop(columns=['StudentID'])
        logging.info("'StudentID' dropped successfully.")

    # Initial inspection
    initial_inspection(data)

    # Missing value and duplicate checks
    missing_and_duplicates(data)

    # Outlier detection
    outlier_detection(data)

    # Distribution diagnostics
    distribution_diagnostics(data)

    logging.info("Data preparation completed successfully.")


if __name__ == '__main__':
    main()
