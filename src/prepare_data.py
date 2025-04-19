import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the raw dataset
data = pd.read_csv("../data/raw/student_performance_data.csv")

# Remove the StudentID column, as it is a non-predictive unique identifier
data = data.drop(columns=['StudentID'])

# Display the first few rows and dataset info to verify the change
print("First few rows after removing StudentID:")
print(data.head())
print("\nDataset info:")
data.info()

# Save the cleaned dataset for future use
data.to_csv("../data/processed/cleaned_data.csv", index=False)

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, "..", "artifacts", "logistic_regression.pkl")

# Load your trained model (you should replace 'model.pkl' with your actual model path)
model = joblib.load(model_path)