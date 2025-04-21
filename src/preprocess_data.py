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
    Map continuous GPA into ordinal GradeClass:
    <=1→4, 1-2→3, 2-3→2, 3-4→1, >4→0
    """
    bins = [-np.inf, 1.0, 2.0, 3.0, 4.0, np.inf]
    labels = [4, 3, 2, 1, 0]
    return pd.cut(gpa_series, bins=bins, labels=labels).astype(int)


def main():
    setup_logging()
    raw_path = os.path.join('data', 'processed', 'cleaned_data.csv')
    out_dir = os.path.join('data', 'processed')
    out_path = os.path.join(out_dir, 'engineered_data.csv')

    # ensure directory
    os.makedirs(out_dir, exist_ok=True)

    # load
    try:
        logging.info(f"Loading cleaned data from {raw_path}")
        df = pd.read_csv(raw_path)
    except Exception as e:
        logging.error(f"Failed to load cleaned data: {e}")
        sys.exit(1)

    # create target
    try:
        logging.info("Mapping GPA to GradeClass")
        df['GradeClass'] = gpa_to_grade_class(df['GPA'])
        logging.info("Dropping original GPA column")
        df.drop(columns=['GPA'], inplace=True)

        # extra features
        logging.info("Creating engagement score")
        flags = ['Tutoring','Extracurricular','Sports','Music','Volunteering']
        df['Engagement'] = df[flags].sum(axis=1)

        logging.info("Creating absence bands")
        df['AbsenceBand'] = pd.cut(
            df['Absences'], bins=[-1,5,15,25,df['Absences'].max()],
            labels=['Very Low','Low','Medium','High']
        )

        logging.info("Log-transforming StudyTimeWeekly")
        df['LogStudyTime'] = np.log1p(df['StudyTimeWeekly'])

        logging.info("Creating family support index")
        df['FamilySupport'] = df['ParentalEducation'] * df['ParentalSupport']

        logging.info("Creating interaction: Study_x_Support")
        df['Study_x_Support'] = df['LogStudyTime'] * df['FamilySupport']

        # save
        df.to_csv(out_path, index=False)
        logging.info(f"Engineered data saved to {out_path}")
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()