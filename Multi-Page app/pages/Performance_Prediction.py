import os
from pathlib import Path
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import logging

# Register page
dash.register_page(__name__, path="/performance_prediction", suppress_callback_exceptions=True)

# Locate artifacts folder (project-root/artifacts/)
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
if not ARTIFACTS_DIR.exists():
    raise FileNotFoundError(f"Could not find artifacts/ at {ARTIFACTS_DIR}")

logging.basicConfig(level=logging.INFO)
logging.info(f"Loading models from {ARTIFACTS_DIR}")

# Define your models with their metadata
models_info = [
    {"key": "scaled_lr", "label": "Scaled Logistic Regression", "path": "ScaledLogisticRegression.joblib", "accuracy": 0.71, "recommended": True, "type": "sklearn"},
    {"key": "lr",        "label": "Logistic Regression",        "path": "LogisticRegression.joblib",          "accuracy": 0.689, "recommended": False, "type": "sklearn"},
    {"key": "rf_clf",   "label": "Random Forest Classifier",   "path": "RandomForestClassifier.joblib","accuracy": 0.733, "recommended": False, "type": "sklearn"},
    {"key": "rf_reg",   "label": "Random Forest Regressor",    "path": "RandomForestRegressor.joblib","accuracy": 0.708, "recommended": False, "type": "sklearn"},
    {"key": "xgb",      "label": "XGBoost Classifier",        "path": "XGBClassifier.joblib",           "accuracy": 0.735, "recommended": False, "type": "sklearn"},
    {"key": "nn",       "label": "Deep Learning MLP",         "path": "DeepLearningMLP.keras",          "accuracy": 0.760, "recommended": False, "type": "keras"}
]

# Load each model
loaded_models = {}
for spec in models_info:
    fullpath = ARTIFACTS_DIR / spec["path"]
    logging.info(f"Loading {spec['label']} from {fullpath}")
    if spec["type"] == "sklearn":
        loaded_models[spec["key"]] = joblib.load(fullpath)
    else:
        loaded_models[spec["key"]] = tf.keras.models.load_model(str(fullpath))

# Build model selector options
dropdown_options = []
for spec in models_info:
    label = f"{spec['label']} ({spec['accuracy']*100:.1f}%)"
    if spec['recommended']:
        label += " [Recommended]"
    dropdown_options.append({"label": label, "value": spec['key']})

# Define input fields
INPUT_FIELDS = [
    {"label": "Age", "id": "age", "component": dcc.Input,      "props": {"type": "number", "min": 0, "value": 18}},
    {"label": "Gender", "id": "gender", "component": dcc.Dropdown, "props": {"options": [{"label": "Male","value": 0},{"label": "Female","value": 1}], "value": 0}},
    {"label": "Ethnicity", "id": "ethnicity", "component": dcc.Dropdown, "props": {"options": [
        {"label": "Caucasian","value": 0}, {"label": "African American","value": 1},
        {"label": "Asian","value": 2},       {"label": "Other","value": 3}
    ], "value": 0}},
    {"label": "Parental Education", "id": "parent-edu", "component": dcc.Dropdown, "props": {"options": [
        {"label": "None","value": 0}, {"label": "High School","value": 1},
        {"label": "Some College","value": 2}, {"label": "Bachelor's","value": 3},
        {"label": "Higher Study","value": 4}
    ], "value": 0}},
    {"label": "Study Time Weekly (hrs)", "id": "study-time", "component": dcc.Input, "props": {"type": "number", "min": 0, "value": 0}},
    {"label": "Absences", "id": "absences", "component": dcc.Input, "props": {"type": "number", "min": 0, "value": 0}},
    {"label": "Tutoring", "id": "tutoring", "component": dcc.RadioItems, "props": {"options": [{"label": "Yes","value": 1},{"label": "No","value": 0}], "value": 0, "inline": True}},
    {"label": "Parental Support", "id": "parent-support", "component": dcc.RadioItems, "props": {"options": [
        {"label": "None","value": 0},{"label": "Low","value": 1},{"label": "Moderate","value": 2},
        {"label": "High","value": 3},{"label": "Very High","value": 4}
    ], "value": 0}},
    {"label": "Extracurricular", "id": "extracurricular", "component": dcc.RadioItems, "props": {"options": [{"label":"Yes","value":1},{"label":"No","value":0}], "value": 0, "inline": True}},
    {"label": "Sports", "id": "sports", "component": dcc.RadioItems, "props": {"options": [{"label":"Yes","value":1},{"label":"No","value":0}], "value": 0, "inline": True}},
    {"label": "Music", "id": "music", "component": dcc.RadioItems, "props": {"options": [{"label":"Yes","value":1},{"label":"No","value":0}], "value": 0, "inline": True}},
    {"label": "Volunteering", "id": "volunteering", "component": dcc.RadioItems, "props": {"options": [{"label":"Yes","value":1},{"label":"No","value":0}], "value": 0, "inline": True}}
]

# Helper to create form rows
def create_form_row(field):
    return dbc.Row([
        dbc.Label(field['label'], width=4, className="fw-semibold"),
        dbc.Col(field['component'](id=field['id'], **field['props']), width=8)
    ], className="mb-3")

# Page layout
layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Performance Predictions"), className="text-center mb-4")),
    dbc.Row(
        dbc.Col(
            dbc.Card([
                dbc.CardBody(
                    [
                        *[create_form_row(f) for f in INPUT_FIELDS],
                        html.Hr(),
                        create_form_row({
                            'label': 'Model',
                            'id': 'model-select',
                            'component': dcc.Dropdown,
                            'props': {'options': dropdown_options, 'value': 'scaled_lr'}
                        })
                    ], className="p-4"
                ),
                dbc.CardFooter(
                    dbc.Button(
                        "Predict Grade", id="predict-btn", color="primary", className="w-100 fw-bold", size="lg"
                    ),
                    className="p-0"
                )
            ]), width=6
        ), className="justify-content-center mb-4"
    ),
    # Prediction Result Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Prediction Result")),
        dbc.ModalBody(id='modal-body'),
        dbc.ModalFooter(
            dbc.Button("Close", id='close-modal', className="ms-auto", n_clicks=0)
        )
    ], id='prediction-modal', is_open=False)
], fluid=True)

# Callback to handle predictions and modal toggling
@callback(
    Output('prediction-modal', 'is_open'),
    Output('modal-body', 'children'),
    [
        Input('predict-btn', 'n_clicks'),
        Input('close-modal', 'n_clicks'),
        *[Input(field['id'], 'value') for field in INPUT_FIELDS],
        Input('model-select', 'value')
    ]
)
def predict(n_clicks, close_clicks, *values_and_model):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, None
    trig_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Close modal if clicked
    if trig_id == 'close-modal':
        return False, dash.no_update

    # On predict button click, compute and show prediction
    if trig_id == 'predict-btn':
        # Map inputs to DataFrame
        ids = [field['id'] for field in INPUT_FIELDS]
        cols = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTime', 'Absences',
                'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
        raw_vals = dict(zip(ids, values_and_model[:-1]))
        df = pd.DataFrame([{cols[i]: raw_vals[ids[i]] for i in range(len(ids))}])

        # Feature engineering
        flags = ['Tutoring', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
        df['Engagement'] = df[flags].sum(axis=1)
        df['FamilySupport'] = df['ParentalEducation'] * df['ParentalSupport']
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Prepare features
        X = df.values
        model_key = values_and_model[-1]
        model = loaded_models.get(model_key)
        if model is None:
            result = html.Div("Error: Model not found.", style={'color': 'red'})
        else:
            # Run prediction
            raw_pred = model.predict(X)
            # Grade mapping
            grade_map = {
                0: ("A", "GPA ≥ 3.5"),
                1: ("B", "3.0 ≤ GPA < 3.5"),
                2: ("C", "2.5 ≤ GPA < 3.0"),
                3: ("D", "2.0 ≤ GPA < 2.5"),
                4: ("F", "GPA < 2.0")
            }
            # Classification
            if (hasattr(model, 'predict_proba') or
                (isinstance(raw_pred, np.ndarray) and raw_pred.ndim > 1 and raw_pred.shape[1] > 1)):
                probs = (model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else raw_pred[0])
                class_idx = int(np.argmax(probs))
                letter, desc = grade_map.get(class_idx, ("?", ""))
                conf = float(np.max(probs))
                # Color: green for pass (not F), red for fail (F)
                color = 'green' if class_idx != 4 else 'red'
                result = html.Div([
                    html.H4(f"Predicted Grade: {letter} ({desc})", style={'color': color}),
                    html.P(f"Confidence: {conf:.2f}")
                ])
            else:
                # Regression or single output (GPA)
                pred_val = float(raw_pred[0]) if isinstance(raw_pred, np.ndarray) else float(raw_pred)
                # Color: green for pass GPA >= 2.0, red otherwise
                color = 'green' if pred_val >= 2.0 else 'red'
                result = html.Div([
                    html.H4(f"Predicted GPA: {pred_val:.2f}", style={'color': color})
                ])
        return True, result

    return False, None