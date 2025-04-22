import dash
from dash import html, dcc
import dash_bootstrap_components as dbc 

app = dash.Dash( use_pages=True, suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.PULSE])
app.title = "BrightPath Grade Predictor"

server = app.server

app.layout = dbc.Container([

    dcc.Location(id='url'),

    dbc.Row([
        dbc.Col(dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("EDA Insights", href="/eda_insights")),
                dbc.NavItem(dbc.NavLink("Student Performance Prediction", href="/performance_prediction")),
                dbc.NavItem(dbc.NavLink("Model Comparison", href="/model_comparison")),
                dbc.NavItem(dbc.NavLink("About", href="/about")),
            ],
            className="navbar bg-dark navbar-expand-lg navbar-dark",
            brand="BrightPath Grade Predictor",
            brand_href="/",
        ),
        
        width=12)
    ]),
    #html.Hr(),
    html.Div([

    ],
    style={'margin': '50px'}),
    dash.page_container  # 👈 This is where our pages render
], 
fluid=True,)

if __name__ == "__main__":
    app.run(debug=False)