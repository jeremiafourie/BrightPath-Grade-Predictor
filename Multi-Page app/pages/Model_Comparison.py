import dash
from dash import html

dash.register_page(__name__, path="/model_comparison")

layout = html.Div([
    html.H2("Welcome to the Model Comparison Page"),
    html.P("This page provides insights from the model comparison."),
    
   
])