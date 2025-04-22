import dash
from dash import html, dash_table
import dash_bootstrap_components as dbc

# Register this page
dash.register_page(__name__, path="/model_comparison")

# Detailed comparison data (excluding Experiments 2 & 3)
comparison_data = [
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.762,
        "Precision": 0.74,
        "Recall": 0.76,
        "F1-Score": 0.75
    },
    {
        "Model": "Scaled Logistic Regression",
        "Accuracy": 0.8225,
        "Precision": 0.83,
        "Recall": 0.82,
        "F1-Score": 0.82
    },
    {
        "Model": "Random Forest Classifier",
        "Accuracy": 0.770,
        "Precision": 0.69,
        "Recall": 0.70,
        "F1-Score": 0.69
    },
    {
        "Model": "Random Forest Regressor",
        "Accuracy": 0.777,
        "Precision": 0.71,
        "Recall": 0.72,
        "F1-Score": 0.71
    },
    {
        "Model": "XGBoost Classifier",
        "Accuracy": 0.789,
        "Precision": 0.80,
        "Recall": 0.79,
        "F1-Score": 0.79
    },
    {
        "Model": "Deep Learning MLP",
        "Accuracy": 0.808,
        "Precision": 0.81,
        "Recall": 0.81,
        "F1-Score": 0.81
    }
]

# Determine the maximum F1-Score for highlighting
enabled_max_f1 = max(item["F1-Score"] for item in comparison_data)

# Define columns for DataTable
columns = [
    {"name": "Model",      "id": "Model",      "type": "text"},
    {"name": "Accuracy",   "id": "Accuracy",   "type": "numeric", "format": {"specifier": ".2%"}},
    {"name": "Precision",  "id": "Precision",  "type": "numeric", "format": {"specifier": ".2%"}},
    {"name": "Recall",     "id": "Recall",     "type": "numeric", "format": {"specifier": ".2%"}},
    {"name": "F1-Score",   "id": "F1-Score",   "type": "numeric", "format": {"specifier": ".2%"}}
]

layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Model Comparison"), className="text-center my-4")),
    dbc.Row(dbc.Col(
        dash_table.DataTable(
            data=comparison_data,
            columns=columns,
            style_cell={"textAlign": "center", "padding": "8px"},
            style_header={"backgroundColor": "#343a40", "color": "#f8f9fa", "fontWeight": "bold"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{Accuracy} >= 0.8", "column_id": "Accuracy"},
                    "backgroundColor": "#d4edda", "color": "#155724"
                },
                {
                    "if": {"filter_query": "{Accuracy} < 0.8", "column_id": "Accuracy"},
                    "backgroundColor": "#f8d7da", "color": "#721c24"
                },
                # Highlight the row(s) with the maximum F1-Score
                {
                    "if": {"filter_query": f"{{F1-Score}} = {enabled_max_f1}", "column_id": "F1-Score"},
                    "backgroundColor": "#cfe2ff", "color": "#084298"
                }
            ],
            page_size=6,
            sort_action="native",
            style_table={"overflowX": "auto"}
        ),
        width=8
    ), justify="center")
], fluid=True)
