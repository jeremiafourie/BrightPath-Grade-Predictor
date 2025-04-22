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
- **Submission Date:** 22 April 2025, 11:59 PM

### Problem Statement

BrightPath Academy faces challenges in identifying at-risk students early, understanding how extracurricular activities influence grades, and developing targeted support strategies. This project addresses these issues by building a predictive model for `GradeClass` and analyzing key factors affecting student outcomes.

### Hypotheses

We hypothesize that:

- Students with higher `StudyTimeWeekly` are more likely to achieve better grades.
- Higher `Absences` correlate with lower grades.
- Participation in `Extracurricular` activities positively impacts grades.
- `ParentalSupport` levels significantly influence student performance.

These hypotheses are explored in `notebooks/02_eda.ipynb` via visualizations and statistical analyses.

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
  - Absences vs. GPA has a strong negative correlation (–0.92); StudyTime vs. GPA is modestly positive (+0.18); ParentalSupport vs. GPA is weakly positive (~+0.19).
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
  - _Jeremia Fourie:_ Exploratory Data Analysis, visualizations and feature engineering
  - _Juan Oosthuizen:_ Baseline modeling and hyperparameter tuning
  - _Busisiwe Radebe:_ Data loading, cleaning, and report writing
  - _Phumlani Ntuli:_ dash app and deployment

Feel free to open issues or submit pull requests for improvements!
