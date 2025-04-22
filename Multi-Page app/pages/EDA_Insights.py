import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def univariateNumerical():
    
    Histogram_list = []

    num_cols = num_df.columns.tolist()

    # Determine grid size based on the number of variables (here 5 variables)
    # We'll use 2 rows and 3 columns (6 subplots) and remove the unused subplot.
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    # Loop over each numerical variable to create a histogram with KDE
    for i ,col in enumerate(num_cols):
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

        fig.update_traces(
            marker_line_color='black',
            marker_line_width=1.5,
        )
        
        Histogram_list.append(

            dbc.Col(
                dcc.Graph(figure=fig, id=f'Histogram-{col}'),
                width=4,
            )

        )
        

    plt.tight_layout()

    return Histogram_list

def univariateCatigorical():
    

    # Get list of categorical column names
    cat_cols = cat_df.columns.tolist()

    # Define grid size
    n_cols = 4
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols

    # Create subplot grid for pie charts
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=[[{'type':'domain'}]*n_cols for _ in range(n_rows)],
                        subplot_titles=[f"Distribution of {col}" for col in cat_cols])

    # Add pie charts to subplots
    for idx, col in enumerate(cat_cols):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        counts = cat_df[col].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                name=col,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Pastel)
            ),
            row=row,
            col=col_pos
        )

    # Adjust layout
    fig.update_layout(
        title_text="Categorical Variable Distributions",
        height=300 * n_rows,
        showlegend=False
    )

    # Optional: display in Dash
    

    return dcc.Graph(figure=fig)

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

            html.H1("Univariate Analysis"),
            html.H2("Distribution of Numerical Variables"),

        ]),
    ]),

    dbc.Row([
        *univariateNumerical(),
    ]),

    dbc.Row([
        dbc.Col([

           html.Ul([
               html.Li("StudyTimeWeekly: The majority of students cluster between 0–10 hours/week, with a long right tail stretching to ~20 hrs. This suggests a small subset of very high‑effort students—worth flagging as “super‑studiers” or potential outliers."),
               html.Li("Absences: Absences center around 10–20 days, but there’s a tail up to 29. That tail may indicate chronic absenteeism requiring special handling or capping."),
               html.Li("GPA: GPA has a peak around 1.5–2.0, then a secondary bump near 4.0 (ceiling effect). The perfect‑4 spike suggests grade inflation or particularly high achievers."),
           ]),
           
        ]),
    ]),

    dbc.Row([
        dbc.Col([
            html.H2("Distribution of Categorical Variables"),
            univariateCatigorical(),

            html.Ul([
                html.Li("Tutoring, extracurriculars, and sports all hover around the 30–40 % mark, so they’ll each carry enough variation to help explain GradeClass—but music and volunteering, at ~20 % and ~16 %, may need to be grouped or re‑encoded if sparse classes is going to be a problem.")
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
