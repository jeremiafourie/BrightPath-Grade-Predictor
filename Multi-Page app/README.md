# BrightPath-Grade-Predictor

A predictive modeling project to classify student performance (GradeClass) at BrightPath Academy using demographic, study, and extracurricular data.

## 📚 Project Overview

This is a group project for Belgium Campus to predict student performance (GradeClass) at BrightPath Academy using academic, demographic, and extracurricular data. We aim to identify at-risk students, assess extracurricular impacts, and suggest support strategies through EDA and predictive modeling.

- Institution: Belgium Campus
- Course: MLG382
- Group S:
  - Jeremia Fourie
  - Juan Oosthuizen
  - Busisiwe Radebe
  - Phumlani Ntuli
- Submission Date: 22 April 2025, 12:00 AM

### Problem Statement

BrightPath Academy faces challenges in identifying at-risk students early, understanding how extracurricular activities influence grades, and developing targeted support strategies. This project addresses these issues by building a predictive model for `GradeClass` and analyzing key factors affecting student outcomes.

### Hypotheses

We hypothesize that:

- Students with higher `StudyTimeWeekly` are more likely to achieve better grades.
- Higher `Absences` correlate with lower grades.
- Participation in `Extracurricular` activities positively impacts grades.
- `ParentalSupport` levels significantly influence student performance.

These hypotheses will be explored and tested in the `notebooks/02_eda.ipynb` notebook through data visualizations and statistical analysis.

## 📂 Dash App File Structure

ill get to this...

<pre lang="markdown">
```bash
Multi-Page App/
|
|
|--- app.py
```
</pre>

## 📝 Notes

oof... uh...
this is Phumlani and welcome to my thought process of this Dash App...
So I have started... and oh boy. I need to do some designing fr.

uh all in all this has been cooking me for a few days now but we move.

EDA_insights seems alright.

Performance Predictions is NOT alright. jeez. maybe I should just focus on front end things neh?? yeah.. make it look pretty.

### Jupyter Notebooks vs .py files

I wanted to open the app using jupyter notebooks but thats not really good for deployment and multiple pages so its better to just use python file.
have a multiple file structure type thing.

on that note the app runs on this server: http://127.0.0.1:8050

⸻

# Things to keep in mind about the dash app

Here’s a comprehensive guideline for designing your Dash app for the BrightPath Academy student performance project. The design will aim to be clean, interactive, and structured for both educators and analysts to use easily.

⸻

## 🖥 DASH APP DESIGN GUIDELINE

⸻

### 📌 Overall Structure

Use a multi-tab or sectioned layout to separate major functionalities for clarity:

• Tab 1: Overview & Project Summary  
• Tab 2: EDA Insights & Visualizations  
• Tab 3: Student Performance Prediction  
• Tab 4: Model Comparison  
• Tab 5: About the Team / App Instructions

Use dcc.Tabs() or sidebar layout (dash-bootstrap-components) for navigation.

⸻

### 📁 Tab 1: Overview & Project Summary

Purpose: Introduce the app, context, and project goals.

Components:
• Markdown or HTML summary of BrightPath Academy and the problem statement  
• Key project objectives  
• Hypotheses  
• Dataset description (feature explanations)

Dash Elements:

dcc.Markdown(children=markdown_text)

⸻

### 📊 Tab 2: EDA Insights & Visualizations

Purpose: Show important findings from the EDA

Components:
• Graphs showing distributions, correlations, patterns:  
• GPA vs Absences  
• GradeClass distribution  
• Study Time vs GradeClass  
• Parental Support vs GPA  
• Heatmap of correlations  
• Dynamic filters for exploring EDA (optional):  
• Filter by Gender, Ethnicity, etc.

Dash Elements:

dcc.Graph() # For Matplotlib/Plotly figures
dcc.Dropdown() # Optional filtering

⸻

### 🔮 Tab 3: Student Performance Prediction

Purpose: Allow users to input student data and receive GradeClass predictions using ML and DL models

Components:
• Form inputs for features:  
• Age, Gender, Ethnicity, StudyTimeWeekly, Absences, ParentalSupport, etc.  
• Radio buttons or dropdown to select model:  
• “Logistic Regression”, “Random Forest”, “XGBoost”, “Neural Network”  
• Prediction result display (text and color-coded card)  
• Optionally: display probability/confidence levels

Dash Elements:

dcc.Input(), dcc.Dropdown(), dcc.RadioItems()  
html.Button('Predict', id='predict-button')  
html.Div(id='prediction-output')

⸻

### 📈 Tab 4: Model Comparison

Purpose: Help users understand how the different models perform

Components:
• Table or graph showing:  
• Accuracy, Precision, Recall, F1-score  
• Confusion matrix visualization  
• Short textual summary of when/why each model performs better

Dash Elements:

dash_table.DataTable() # Metrics summary  
dcc.Graph() # Confusion matrix or bar chart

⸻

### 👥 Tab 5: About / Instructions

Purpose: Provide credits, contact, usage help

Components:
• Short bios of team members (Phumlani, Jeremia, Juan, Busiswe)  
• How to use the app  
• Notes on limitations or assumptions

⸻

### 🎨 UI/UX Design Tips

    •	Use dash-bootstrap-components for a modern layout
    •	Use Card components for prediction results (e.g., color-coded by GradeClass)
    •	Use Spinners (dcc.Loading) to show loading on prediction
    •	Ensure mobile responsiveness
    •	Add tooltips (html.Abbr, dbc.Tooltip) for feature explanations

⸻

### 🧪 Example Directory Structure

/app  
├── app.py ← main Dash app file  
├── assets/ ← (optional: CSS or image files)  
├── models/  
│ ├── ml_model.pkl  
│ └── dl_model.h5  
├── utils/  
│ └── predict.py ← functions for ML/DL prediction  
└── data/  
 │ └── student_performance_cleaned.csv
