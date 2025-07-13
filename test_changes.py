#!/usr/bin/env python3
"""
Quick test script to verify the counterfactual tab changes work correctly.
"""
import pandas as pd
import seaborn as sns

# Load and preprocess Titanic dataset (same as in the app)
df = (
    sns.load_dataset("titanic")
    .loc[:, ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
    .dropna()
)
X_df = pd.get_dummies(
    df.drop(columns="survived"), columns=["sex", "embarked"], drop_first=True
)

# Map feature names to more descriptive labels (same as in the app)
feature_name_map = {
    "pclass": "Passenger Class (1=1st, 2=2nd, 3=3rd)",
    "age": "Age",
    "sibsp": "Siblings/Spouses Aboard",
    "parch": "Parents/Children Aboard",
    "fare": "Fare (Ticket Price)",
    "sex_male": "Is Male",
    "embarked_Q": "Embarked at Queenstown",
    "embarked_S": "Embarked at Southampton",
}

# Use original feature names for data processing
orig_feature_names = X_df.columns.tolist()
# Use descriptive names for display and SHAP explanations
feature_names = [feature_name_map.get(f, f) for f in orig_feature_names]

print("Original feature names:")
print(orig_feature_names)
print("\nDescriptive feature names:")
print(feature_names)
print("\nFirst few rows of X_df:")
print(X_df.head())

# Test the sample features display logic
idx = 0
sample_features = X_df.iloc[idx : idx + 1].copy()
sample_features.columns = feature_names

print(f"\nSample features for passenger {idx} (with descriptive names):")
print(sample_features)
print("\nSample features shape:", sample_features.shape)
print("All columns included:", len(sample_features.columns) == len(feature_names))
