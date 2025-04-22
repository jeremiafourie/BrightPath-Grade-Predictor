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

df = pd.read_csv("data/raw/student_performance_data.csv")

######################## eda.ipynb ###########################

data = pd.read_csv("data/processed/cleaned_data.csv")

# Numerical variables (continuous or discrete without inherent order)
numerical_vars = ['StudyTimeWeekly', 'Absences', 'GPA']

# Categorical variables (nominal, no inherent order)
categorical_vars = ['Gender', 'Ethnicity', 'Tutoring', 'Extracurricular', 
                    'Sports', 'Music', 'Volunteering']

# Ordinal variables (categorical with a natural order)
ordinal_vars = ['Age', 'ParentalEducation', 'ParentalSupport', 'GradeClass']

# Create separate DataFrames for numerical, categorical, and ordinal variables
num_df = data[numerical_vars].copy()
cat_df = data[categorical_vars].copy()
ord_df = data[ordinal_vars].copy()

###############################################################

def createHistograms():
    
    Histogram_list = []

    num_cols = num_df.columns.tolist()

    # Determine grid size based on the number of variables (here 5 variables)
    # We'll use 2 rows and 3 columns (6 subplots) and remove the unused subplot.
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Loop over each numerical variable to create a histogram with KDE
    # for i, col in enumerate(num_cols):
    #     sns.histplot(num_df[col], kde=True, bins=25, ax=axes[i], color='skyblue')
    #     axes[i].set_title(f"Histogram of {col}")
    #     axes[i].set_xlabel(col)
    #     axes[i].set_ylabel("Frequency")
    #     Histogram_list.append(dcc.Graph(figure=axes, id=f'Histogram-{col}'))

    for col in num_df.columns:
        fig = px.histogram(
            num_df,
            x=col,
            nbins=25,
            title=f"Histogram of {col}",
            color_discrete_sequence=["skyblue"]
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Frequency"
        )
        
        Histogram_list.append(dcc.Graph(figure=fig, id=f'Histogram-{col}'))


    # Remove any extra subplots (if there are any)
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

    plt.tight_layout()
    # plt.show()

    return Histogram_list


layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            html.H2("Welcome to the EDA Insights Page"),
            html.P("This page provides insights from the exploratory data analysis."),

    

            html.Div([
                html.Label("Filter by Gender"),
                dcc.Dropdown(
                    id='gender-filter',
                    options=[
                        {'label': "Male", 'value': 1},
                        {'label': "Female", 'value': 0}
                    ],
                    value=None,
                    placeholder="Select gender...",
                    clearable=True,
                )
            ], style={
                'width': '300px',           # fixed width helps with centering
                'margin': '0 auto 20px',    # this centers it horizontally
                'textAlign': 'left'         # optional: align label text inside box
            }),

        ],align="center",style={"text-align": "center"}),
        
    ],justify="center"),

    



    dbc.Row([
        dbc.Col([

            html.Div([
                html.H2("Distribution of Numerical Variables"),
                *createHistograms()
            
            ]),

        ]),
    ]),

    dbc.Card([
        dbc.Row([
            
            dbc.Col(
                dbc.CardBody([
                   
                    dcc.Graph(id="gpa-vs-absences"),
                    dcc.Graph(id="gradeclass-dist"),
                ]),
            ),

            dbc.Col([
                dbc.CardBody([
                    dcc.Graph(id="studytime-vs-gradeclass"),
                    dcc.Graph(id="parent-support-vs-gpa"),
                ]),
            ]),

        ]),

        dbc.Row([
            dbc.Col([
                dbc.CardBody([
                    dcc.Graph(id="heatmap-corr"),
                ])
            ]),
        ]),

    ]),        
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
