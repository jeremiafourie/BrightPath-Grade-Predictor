import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.graph_objects as go

import os

import dash_bootstrap_components as dbc



dash.register_page(__name__, path="/eda_insights")

df = pd.read_csv("Multi-Page app/data/student_performance_data.csv") 


layout = dbc.Container([
    html.H2("Welcome to the EDA Insights Page"),
    html.P("This page provides insights from the exploratory data analysis."),
    html.P(df['GradeClass'].unique()),

    #   html.H2("📊 EDA Insights & Visualizations"),

    html.Div([
        html.Label("Filter by Gender"),
        dcc.Dropdown(
            id='gender-filter',
            options=[{'label': "Male", 'value': 1},
                     {'label': "Female", 'value': 0}],
            value=None,
            placeholder="Select gender...",
            clearable=True,
        )
    ], style={'width': '30%', 'margin-bottom': '20px'}),

    dcc.Graph(id="gpa-vs-absences"),
    dcc.Graph(id="gradeclass-dist"),
    dcc.Graph(id="studytime-vs-gradeclass"),
    dcc.Graph(id="parent-support-vs-gpa"),
    dcc.Graph(id="heatmap-corr")
   
])

# Calculate correlation matrix
corr_matrix = df.corr()


@callback(
    Output("heatmap-corr", "figure"),
    Input("heatmap-corr", "id")
)

def update_heatmap_corr(_):

    # Create the Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1
    ))

    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        template='plotly_dark' 
    )

    return fig


#callback for gpa vs absences
@callback(
    Output("gpa-vs-absences", "figure"),
    Input("gender-filter", "value")
)
def update_gpa_vs_absences(gender):
    #filter the gender by Male and Female
    filtered = df if gender is None else df[df['Gender'] == gender]
    
    gpa_vs_absences = px.scatter(filtered, x="Absences", y="GPA", color="GradeClass", title="GPA vs Absences",category_orders={"GradeClass": [0.0, 1.0, 2.0, 3.0, 4.0]})

    return gpa_vs_absences

#callback for gradeclass distribution
@callback(
    Output("gradeclass-dist", "figure"),
    Input("gender-filter", "value")
)

def update_gradeclass_dist(gender):  
    
    #filter the gender by Male and Female
    filtered = df if gender is None else df[df['Gender'] == gender]

    gradeclass_dist = px.histogram(
    filtered,
    x="GradeClass",
    title="GradeClass Distribution (Bar Chart)",
    labels={"GradeClass": "Grade Class"},
    color="GradeClass", 
    category_orders={"GradeClass": [0.0, 1.0, 2.0, 3.0, 4.0]},
)
    return gradeclass_dist

#callback for studytime vs gradeclass
@callback(
    Output("studytime-vs-gradeclass", "figure"),
    Input("gender-filter", "value")
)

def update_studytime_vs_gradeclass(gender):

    #filter the gender by Male and Female
    filtered = df if gender is None else df[df['Gender'] == gender]

    studytime_vs_gradeclass = px.box(
    filtered,
    x="GradeClass",
    y="StudyTimeWeekly",
    title="Study Time vs GradeClass (Box Plot)",
    labels={"GradeClass": "Grade Class", "StudyTime": "Study Time (hours/week)"},
    color="GradeClass",  # optional for coloring by class
    points="all",  # shows all data points as dots (optional)
    category_orders={"GradeClass": [0.0, 1.0, 2.0, 3.0, 4.0]}
)
    return studytime_vs_gradeclass

#callback for parent support vs gpa
@callback(
    Output("parent-support-vs-gpa", "figure"),
    Input("gender-filter", "value")
)

def update_parent_support_vs_gpa(gender):

    #filter the gender by Male and Female
    filtered = df if gender is None else df[df['Gender'] == gender]

    fig_box = px.box(
    filtered,
    x="ParentalSupport",
    y="GPA",
    title="Parental Support vs GPA (Box Plot)",
    labels={"ParentalSupport": "Parental Support", "GPA": "GPA"},
    color="ParentalSupport",  # Optional: color by Parental Support
    points="all",  # Shows all individual points (optional)
    category_orders={"ParentalSupport": [0.0, 1.0, 2.0, 3.0, 4.0]}
)
    return fig_box
