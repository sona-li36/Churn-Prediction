import pandas as pd
import numpy as np
import os
from pycaret.classification import *

# Configuration
RANDOM_SEED = 142

print("Loading dataset...")
# Load Data (using relative path)
dataset = pd.read_csv("Dataset.csv")

# Preprocessing
print("Preprocessing data...")
# Convert TotalCharges to numeric, coercing errors to NaN
dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")

# Check for missing values in TotalCharges
start_len = len(dataset)
dataset = dataset.dropna(subset=['TotalCharges'])
print(f"Dropped {start_len - len(dataset)} rows with missing TotalCharges")

print("Setting up PyCaret Experiment...")
# PyCaret Setup
exp01 = setup(
    data=dataset,
    target="Churn",
    session_id=RANDOM_SEED,
    ignore_features=["customerID"],
    experiment_name='churn_prediction',
    html=False, # Disable HTML for command line
    verbose=True
)

print("Comparing Models...")
# Compare Models
best_model = compare_models()

print("\nBest Model Found:")
print(best_model)
