import dash
from dash import html
import dash_bootstrap_components as dbc  

dash.register_page(__name__)

layout = dbc.Container([

    dbc.Row([
        dbc.Col([

            html.H2("About Us"),

        ],
        style={"text-align": "center", 'marginBottom': "20px"}),    
    ]),

    dbc.Row([
        dbc.Col([

             html.P(f"""Institution: Belgium Campus ITversity 
                Course: MLG382 \n
                Group S:
                Jeremia Fourie
                Juan Oosthuizen
                Busisiwe Radebe
                Phumlani Ntuli \n
                Submission Date: 22 April 2025, 12:00 AM""", 
                style={'white-space': 'pre-line', 'font-size': '18px', 'line-height': '1.5'}),

        ],
        style={"text-align": "center"}),      
    ]),

])