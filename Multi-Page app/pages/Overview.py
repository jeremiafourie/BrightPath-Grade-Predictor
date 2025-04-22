import dash
from dash import html
import dash_bootstrap_components as dbc  # optional for styling

dash.register_page(__name__, path="/")

layout = dbc.Container([
    html.H2("Welcome to the Overview Page"),
    html.P("This is a group project for Belgium Campus to predict student performance (GradeClass) at BrightPath Academy using academic, demographic, and extracurricular data. We aim to identify at-risk students, assess extracurricular impacts, and suggest support strategies through EDA and predictive modeling."),

    html.H3("Problem Statement"),
    html.P("BrightPath Academy faces challenges in identifying at-risk students early, understanding how extracurricular activities influence grades, and developing targeted support strategies. This project addresses these issues by building a predictive model for GradeClass and analyzing key factors affecting student outcomes."),

    html.H3("Hypothesis"),
    html.Ul([
        html.Li("Students with higher StudyTimeWeekly are more likely to achieve better grades."),
        html.Li("Higher Absences correlate with lower grades."),
        html.Li("Participation in Extracurricular activities positively impacts grades."),
        html.Li("ParentalSupport levels significantly influence student performance.")
    ]),
    
],style={'white-space': 'pre-line', 'font-size': '18px', 'line-height': '1.5'})