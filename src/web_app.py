import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import flask
import sklearn
import joblib

# Load your trained model
model = joblib.load('artifacts/logistic_regression.pkl')  # Update path if needed

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1("BrightPath Grade Predictor", className="text-center my-4"),

    dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("Age"),
                dbc.Input(type="number", id="age", min=15, max=18, required=True)
            ]),
            dbc.Col([
                dbc.Label("Gender"),
                dbc.RadioItems(
                    options=[{"label": "Male", "value": 0}, {"label": "Female", "value": 1}],
                    id="gender",
                    inline=True
                )
            ])
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Ethnicity"),
                dbc.RadioItems(
                    options=[
                        {"label": "Caucasian", "value": 0},
                        {"label": "African American", "value": 1},
                        {"label": "Asian", "value": 2},
                        {"label": "Other", "value": 3}
                    ],
                    id="ethnicity",
                    inline=True
                )
            ]),
            dbc.Col([
                dbc.Label("Parental Education"),
                dbc.RadioItems(
                    options=[
                        {"label": "None", "value": 0},
                        {"label": "High School", "value": 1},
                        {"label": "Some College", "value": 2},
                        {"label": "Bachelor's", "value": 3},
                        {"label": "Higher Study", "value": 4}
                    ],
                    id="parent_ed",
                    inline=False
                )
            ])
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Study Time Weekly (hours)"),
                dbc.Input(type="number", id="study_time", min=0, max=20, required=True)
            ]),
            dbc.Col([
                dbc.Label("Absences"),
                dbc.Input(type="number", id="absences", min=0, max=30, required=True)
            ])
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Tutoring"),
                dbc.RadioItems(
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    id="tutoring",
                    inline=True
                )
            ]),
            dbc.Col([
                dbc.Label("Parental Support"),
                dbc.RadioItems(
                    options=[
                        {"label": "None", "value": 0},
                        {"label": "Low", "value": 1},
                        {"label": "Moderate", "value": 2},
                        {"label": "High", "value": 3},
                        {"label": "Very High", "value": 4}
                    ],
                    id="parental_support",
                    inline=False
                )
            ])
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Extracurricular"),
                dbc.RadioItems(
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    id="extracurricular",
                    inline=True
                )
            ]),
            dbc.Col([
                dbc.Label("Sports"),
                dbc.RadioItems(
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    id="sports",
                    inline=True
                )
            ])
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Music"),
                dbc.RadioItems(
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    id="music",
                    inline=True
                )
            ]),
            dbc.Col([
                dbc.Label("Volunteering"),
                dbc.RadioItems(
                    options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                    id="volunteering",
                    inline=True
                )
            ])
        ], className="mb-3"),

        dbc.Button("Predict Grade", id="predict-btn", color="success", className="mt-3"),
    ]),

    dbc.Modal([
        dbc.ModalHeader("Prediction Result"),
        dbc.ModalBody(id="result-text"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
        ),
    ], id="modal", is_open=False)

], style={"maxWidth": "800px"})

@app.callback(
    Output("modal", "is_open"),
    Output("result-text", "children"),
    [Input("predict-btn", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open"),
     State("age", "value"), State("gender", "value"), State("ethnicity", "value"),
     State("parent_ed", "value"), State("study_time", "value"), State("absences", "value"),
     State("tutoring", "value"), State("parental_support", "value"), State("extracurricular", "value"),
     State("sports", "value"), State("music", "value"), State("volunteering", "value")]
)
def toggle_modal(predict_clicks, close_clicks, is_open, age, gender, ethnicity, parent_ed,
                 study_time, absences, tutoring, parental_support,
                 extracurricular, sports, music, volunteering):
    ctx = dash.callback_context

    if not ctx.triggered:
        return is_open, ""
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "predict-btn" and None not in [age, gender, ethnicity, parent_ed,
                                                     study_time, absences, tutoring, parental_support,
                                                     extracurricular, sports, music, volunteering]:
        features = np.array([[age, gender, ethnicity, parent_ed,
                              study_time, absences, tutoring, parental_support,
                              extracurricular, sports, music, volunteering]])
        prediction = model.predict(features)[0]

        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
        grade = grade_map.get(prediction, "Unknown")
        result = f"Predicted Grade: {grade}"
        return True, result

    elif trigger_id == "close":
        return False, ""

    return is_open, ""

if __name__ == '__main__':
    app.run(debug=True)