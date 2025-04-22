# preprocess_data.py
# Script to apply feature engineering transformations to the cleaned student dataset.

import os
import sys
import logging
import pandas as pd
import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def gpa_to_grade_class(gpa_series: pd.Series) -> pd.Series:
    """
    Map continuous GPA into ordinal GradeClass exactly as:
      0: A (GPA >= 3.5)
      1: B (3.0 <= GPA < 3.5)
      2: C (2.5 <= GPA < 3.0)
      3: D (2.0 <= GPA < 2.5)
      4: F (GPA < 2.0)
    """
    return gpa_series.apply(
        lambda g: 0 if g >= 3.5 else
                  1 if g >= 3.0 else
                  2 if g >= 2.5 else
                  3 if g >= 2.0 else
                  4
    ).astype(int)


def main():
    setup_logging()
    raw_path = os.path.join('data', 'processed', 'cleaned_data.csv')
    out_dir = os.path.join('data', 'processed')
    out_path = os.path.join(out_dir, 'engineered_data.csv')

    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # load cleaned data
    try:
        logging.info(f"Loading cleaned data from {raw_path}")
        df = pd.read_csv(raw_path)
    except Exception as e:
        logging.error(f"Failed to load cleaned data: {e}")
        sys.exit(1)

    # Feature engineering
    try:
        # 1) Create ordinal target based on exact GPA thresholds
        logging.info("Mapping GPA to GradeClass")
        df['GradeClass'] = gpa_to_grade_class(df['GPA'])
        logging.info("Dropping original GPA column")
        df.drop(columns=['GPA'], inplace=True)

        # 2) Engagement score
        # logging.info("Creating engagement score")
        # flags = ['Tutoring', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
        # df['Engagement'] = df[flags].sum(axis=1)

        # 3) Absence bands
        # logging.info("Creating absence bands")
        # df['AbsenceBand'] = pd.cut(
        #     df['Absences'],
        #     bins=[-np.inf, 2.0, 2.5, 3.0, 3.5, np.inf],
        #     labels=['F', 'D', 'C', 'B', 'A']
        # )

        # 4) Log transform study time
        # logging.info("Log-transforming StudyTimeWeekly")
        # df['LogStudyTime'] = np.log1p(df['StudyTimeWeekly'])

        # 5) Family support index
        # logging.info("Creating family support index")
        # df['FamilySupport'] = df['ParentalEducation'] * df['ParentalSupport']

        # 6) Interaction term
        # logging.info("Creating interaction: Study_x_Support")
        # df['Study_x_Support'] = df['LogStudyTime'] * df['FamilySupport']

        # 7) One-hot encode all remaining categorical columns
        # cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # if cat_cols:
        #     logging.info(f"One-hot encoding columns: {cat_cols}")
        #     df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # save engineered data
        df.to_csv(out_path, index=False)
        logging.info(f"Engineered data saved to {out_path}")
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()