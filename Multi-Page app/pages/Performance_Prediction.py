import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc  
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go

dash.register_page(__name__, path="/performance_prediction",suppress_callback_exceptions=True)


# For demonstration purposes, im going to initialize models here
logistic_model = LogisticRegression()
random_forest_model = RandomForestClassifier()
xgboost_model = XGBClassifier()
neural_network_model = MLPClassifier()

features = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation' 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Muisc', 'Vulunteering', 'GPA']


layout = dbc.Container([

    dbc.Row([
        dbc.Col([

            html.H2("Welcome to the Performance Predictions Page"),
            html.P("This page provides insights from the performance predictions."),


            html.H3("Chooose your features!"),

        ],align="center", className="text-center", style={'margin-bottom': '50px'}),
    ]),

    
    # Input Fields
    dbc.Row([
        dbc.Col([

            html.Label("Age:"),

            dcc.Input(id='age', type='number', placeholder='Enter Age'),

        ]),

        dbc.Col([

            html.Label("Gender:"),

            dcc.Dropdown(
                id ='gender-dropdown',
                options=[
                    {'label': 'Male', 'value': '1'},
                    {'label': 'Female', 'value': '0'}
                ],
                placeholder="Select Gender"
            ),

        ]),
    ]),
    
    dbc.Row([
        dbc.Col([

            html.Label("Ethnicity:"),

            dcc.Dropdown(
                id='ethnicity-dropdown',
                options=[
                    {'label': 'Group 0', 'value': '0'},
                    {'label': 'Group 1', 'value': '1'},
                    {'label': 'Group 2', 'value': '2'},
                    {'label': 'Group 3', 'value': '3'}
                ],
                placeholder="Select Ethnicity"
            ),

        ]),
    ]),

    dbc.Row([
        dbc.Col([
            
            html.Label("Study Time Weekly (hours):"),
            dcc.Input(id='study-time', type='number', placeholder='Enter Weekly Study Time'),

        ]),

        dbc.Col([

            html.Label("Absences:"),
            dcc.Input(id='absences', type='number', placeholder='Enter Absences'),

        ]),
    ]),
    
    dbc.Row([
        dbc.Col([

            html.Label("Parental Support:"),

            dcc.RadioItems(
                id='parental-support',
                options=[
                    {'label': 'Yes', 'value': '1'},
                    {'label': 'No', 'value': '0'}
                ],
                value='Yes'
            ),

        ]),
    ]),
    

    dbc.Row([
        dbc.Col([

            html.Label("Select Model:"),

            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Logistic Regression', 'value': 'logistic'},
                    {'label': 'Random Forest', 'value': 'random_forest'},
                    {'label': 'XGBoost', 'value': 'xgboost'},
                    {'label': 'Neural Network', 'value': 'neural_network'}
                ],
                value='logistic'
            ),

        ]),

        dbc.Col([

            dbc.Button('Predict', id='predict-button'),

        ]),
    ],
    justify='center',align='center', style={'margin-top': '20px'}),
    
    # Prediction Output
    html.Div(id='prediction-output')
])

# Callback for handling predictions
@callback(
    Output('prediction-output', 'children'),
    [
        Input('predict-button', 'n_clicks'),
        Input('age', 'value'),
        Input('gender-dropdown', 'value'),
        Input('ethnicity-dropdown', 'value'),
        Input('study-time', 'value'),
        Input('absences', 'value'),
        Input('parental-support', 'value'),
        Input('model-dropdown', 'value')
    ]
)

def predict_student_performance(n_clicks, age, gender, ethnicity, study_time, absences, parental_support, model_type):
    if n_clicks is None:
        return ""

    # Convert inputs into a numpy array (ensure correct data types)
    input_data = np.array([[age, gender, ethnicity, study_time, absences, parental_support]])
    
    # Preprocessing (e.g., encoding categorical variables if needed)
    # Example of encoding categorical variables:
    input_data[0, 1] = 1 if input_data[0, 1] == 'Male' else 0  # Gender: Male=1, Female=0
    input_data[0, 3] = {'Group 0': 0, 'Group 1': 1, 'Group 2': 2, 'Group 3': 3}.get(input_data[0, 3], -1)  # Encode Ethnicity

    # Select the model based on user input
    model = None
    if model_type == 'logistic':
        model = logistic_model
    elif model_type == 'random_forest':
        model = random_forest_model
    elif model_type == 'xgboost':
        model = xgboost_model
    elif model_type == 'neural_network':
        model = neural_network_model

    # Ensure the model is available (you should load models from disk if not already trained)
    if model is None:
        return "Please select a valid model."

    #model.fit(X_train, y_train) 

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]  # Get probability for the positive class

    # Display Prediction (color-coded result)
    if prediction[0] == 1:
        prediction_text = f"Predicted Grade Class: 1 (Pass) - Probability: {prediction_proba[0]:.2f}"
        color = 'green'
    else:
        prediction_text = f"Predicted Grade Class: 0 (Fail) - Probability: {prediction_proba[0]:.2f}"
        color = 'red'

    return html.Div([
        html.H4(prediction_text, style={'color': color}),
        html.Div(f"Confidence: {prediction_proba[0]:.2f}", style={'color': color})
    ])