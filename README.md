# BrightPath-Grade-Predictor

A predictive modeling project to classify student performance (`GradeClass`) at BrightPath Academy using demographic, study, and extracurricular data.

## 📚 Project Overview

This is a team project for Belgium Campus to predict student performance (`GradeClass`) at BrightPath Academy using academic, demographic, and extracurricular data. We aim to identify at-risk students, assess extracurricular impacts, and suggest support strategies through EDA and predictive modeling.

- **Institution:** Belgium Campus
- **Course:** MLG382
- **Group S:**
  - Jeremia Fourie
  - Juan Oosthuizen
  - Busisiwe Radebe
  - Phumlani Ntuli
- **Submission Date:** 22 April 2025, 12:00 AM

### Problem Statement

BrightPath Academy faces challenges in identifying at-risk students early, understanding how extracurricular activities influence grades, and developing targeted support strategies. This project addresses these issues by building a predictive model for `GradeClass` and analyzing key factors affecting student outcomes.

### Hypotheses

We hypothesize that:

- Students with higher `StudyTimeWeekly` are more likely to achieve better grades.
- Higher `Absences` correlate with lower grades.
- Participation in `Extracurricular` activities positively impacts grades.
- `ParentalSupport` levels significantly influence student performance.

These hypotheses are explored in `notebooks/02_eda.ipynb` via visualizations and statistical analyses.

## 📂 Repository Structure

```
BrightPath-Grade-Predictor/
├── artifacts/            # Trained models, plots, and other outputs
├── data/
│   ├── raw/              # Original CSV files
│   └── processed/        # Cleaned & feature‑engineered CSV
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── src/                  # Python modules
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## 🛠️ Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/your‑org/BrightPath-Grade-Predictor.git
   cd BrightPath-Grade-Predictor
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3.8 -m venv venv
   source venv/bin/activate   # on Mac/Linux
   .\venv\Scripts\activate  # on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**
   - Place raw CSV(s) in `data/raw/`
   - Run feature engineering:
     ```bash
     jupyter nbconvert --to notebook --execute notebooks/01_feature_engineering.ipynb
     ```
   - Processed data will be saved to `data/processed/engineered_data.csv`

## 🚀 Usage

1. **Exploratory Data Analysis**  
   Open and run `notebooks/02_eda.ipynb` to review distributions, correlations, and bivariate plots.

2. **Model Training & Evaluation**  
   Execute `notebooks/03_modeling.ipynb` (or run the Python script):

   ```bash
   python src/modeling.py --input data/processed/engineered_data.csv --output artifacts/
   ```

   This will:

   - Perform an 80/20 stratified split
   - Train baseline models (Logistic Regression, Random Forest, XGBoost)
   - Build & train a deep‑learning MLP
   - Conduct hyperparameter tuning and SMOTE experiments
   - Save all models and plots under `artifacts/`

3. **Inference**  
   Load the serialized model (e.g. `artifacts/ScaledLogisticRegression.joblib`) in your own script to predict new student records.

## 📊 Key Findings (from EDA)

- **Study Time:** Most students study between 0–10 hrs/week; a small “super‑studier” tail extends to ~20 hrs.
- **Absences:** Centered around 10–20 days (median 15); a few chronic absentees reach up to 29 days.
- **GPA:** Peaks around 1.5–2.0 with a secondary spike at 4.0, suggesting potential ceiling effects.
- **Gender & Ethnicity:** Gender is nearly balanced (51% vs. 49%), while Ethnicity is dominated by category 0 (50.5%).
- **Extracurricular Flags:** Tutoring (~30%), Extracurricular (~38%), Sports (~30%), Music (~20%), Volunteering (~16%)—all sufficiently represented.
- **Parental Factors:** ParentalEducation clusters at level 2 (~940 students); ParentalSupport at levels 2–3 (~700 each).
- **GradeClass Imbalance:** Class 4 contains ~1,210 students vs. only ~110 in class 0.
- **Bivariate Trends:**
  - Higher StudyTimeWeekly and GPA—and lower Absences—correlate with higher GradeClass.
  - Participation in tutoring and extracurriculars increases with GradeClass; smaller gains seen for Music & Volunteering.
  - Absences vs. GPA has a strong negative correlation (~–0.92); StudyTime vs. GPA is modestly positive (~+0.18); ParentalSupport vs. GPA is weakly positive (~+0.19).
- **Group-Level Summary:** As GradeClass increases from 0 to 4, average StudyTimeWeekly and GPA both rise while Absences fall—underscoring the impact of attendance and study habits on performance.

## 🔧 Requirements

- **Python:** 3.8+
- **Key Libraries:** pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, tensorflow, jupyter

Install all with:

```bash
pip install -r requirements.txt
```

## 📝 Notes

- **Division of work:**
  - _Jeremia Fourie:_ Data loading, cleaning, and feature engineering
  - _Juan Oosthuizen:_ Exploratory Data Analysis & visualizations
  - _Busisiwe Radebe:_ Baseline modeling and hyperparameter tuning
  - _Phumlani Ntuli:_ Deep learning implementation and report writing

Feel free to open issues or submit pull requests for improvements!
