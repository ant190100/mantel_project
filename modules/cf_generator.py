"""
Counterfactual Generator Module
------------------------------
Provides helpers for generating counterfactuals using DiCE and for wrapping torch models for DiCE compatibility.
All functions are type-annotated and robust to input errors.
"""

import pandas as pd
import numpy as np
import torch
import dice_ml
from typing import Any, Optional, List


class TorchWrapper:
    def __init__(self, net: torch.nn.Module, thresh: float = 0.5):
        self.net, self.thresh = net.eval(), thresh
        self.dev = next(net.parameters()).device

    def _forward(self, X: Any) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        X = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            return self.net(X).cpu().numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        logits = self._forward(X)
        if logits.shape[1] == 1:
            p1 = 1 / (1 + np.exp(-logits.ravel()))
            return np.column_stack([1 - p1, p1])
        exp = np.exp(logits - logits.max(1, keepdims=True))
        return exp / exp.sum(1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.thresh).astype(int)


def get_dice_engine(
    model: Any,
    X_train_df: pd.DataFrame,
    continuous_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    outcome_name: str = "y_dummy",
) -> dice_ml.Dice:
    """
    Create a DiCE engine for counterfactual generation.
    Args:
        model: Model compatible with DiCE.
        X_train_df: Training data DataFrame.
        continuous_features: List of continuous features.
        categorical_features: List of categorical features.
        outcome_name: Name of outcome column.
    Returns:
        DiCE engine object.
    """
    train_df = X_train_df.copy()
    train_df[outcome_name] = 0  # dummy outcome column

    # Debug: print info about sex_male before conversion
    if "sex_male" in train_df.columns:
        print(f"sex_male dtype before: {train_df['sex_male'].dtype}")
        print(f"sex_male values before: {train_df['sex_male'].unique()}")

    # Robust conversion for all columns
    for col in train_df.columns:
        print(
            f"Processing column {col}: dtype={train_df[col].dtype}, values={train_df[col].unique()}"
        )

        # Force convert all columns to numeric
        if col == outcome_name:
            continue  # Skip the dummy outcome column

        # Try multiple conversion strategies
        original_dtype = train_df[col].dtype
        converted = False

        # Strategy 1: Direct conversion to int/float
        if pd.api.types.is_bool_dtype(train_df[col]):
            train_df[col] = train_df[col].astype(int)
            converted = True
            print(f"  -> Converted bool to int")
        elif pd.api.types.is_integer_dtype(train_df[col]):
            train_df[col] = train_df[col].astype(int)
            converted = True
            print(f"  -> Kept as int")
        elif pd.api.types.is_float_dtype(train_df[col]):
            train_df[col] = train_df[col].astype(float)
            converted = True
            print(f"  -> Kept as float")
        else:
            # Strategy 2: Try converting object/other types
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors="raise")
                converted = True
                print(f"  -> Converted with pd.to_numeric")
            except Exception:
                try:
                    train_df[col] = train_df[col].astype(float)
                    converted = True
                    print(f"  -> Converted to float")
                except Exception:
                    try:
                        train_df[col] = train_df[col].astype(int)
                        converted = True
                        print(f"  -> Converted to int")
                    except Exception:
                        # Strategy 3: Map boolean-like values
                        unique_vals = set(train_df[col].unique())
                        if unique_vals <= {"True", "False"}:
                            train_df[col] = train_df[col].map({"True": 1, "False": 0})
                            converted = True
                            print(f"  -> Mapped string True/False to 1/0")
                        elif unique_vals <= {True, False}:
                            train_df[col] = train_df[col].map({True: 1, False: 0})
                            converted = True
                            print(f"  -> Mapped bool True/False to 1/0")
                        elif unique_vals <= {1, 0} or unique_vals <= {1.0, 0.0}:
                            train_df[col] = train_df[col].astype(int)
                            converted = True
                            print(f"  -> Converted 1/0 values to int")

        if not converted:
            raise ValueError(
                f"Column {col} could not be converted to int or float for DiCE. Original dtype: {original_dtype}, Values: {train_df[col].unique()}"
            )

        print(f"  -> Final dtype: {train_df[col].dtype}")

    # Debug: print info about sex_male after conversion
    if "sex_male" in train_df.columns:
        print(f"sex_male dtype after: {train_df['sex_male'].dtype}")
        print(f"sex_male values after: {train_df['sex_male'].unique()}")
    # Final check: print dtypes and raise error if any column is not int/float
    bad_cols = [
        col
        for col in train_df.columns
        if not (
            pd.api.types.is_integer_dtype(train_df[col])
            or pd.api.types.is_float_dtype(train_df[col])
        )
    ]
    if bad_cols:
        print(
            "Bad columns and their values:",
            {col: train_df[col].unique() for col in bad_cols},
        )
        raise ValueError(
            f"These columns are not int or float and will cause DiCE to fail: {bad_cols}"
        )
    # Infer categorical features (object dtype after conversion)
    if categorical_features is None:
        categorical_features = [c for c in train_df if train_df[c].dtype == "object"]
    if not categorical_features:
        categorical_features = []

    # Infer continuous features (all others except outcome and categorical)
    if continuous_features is None:
        continuous_features = [
            c for c in train_df if c not in categorical_features + [outcome_name]
        ]
    if not continuous_features:
        raise ValueError(
            "No continuous features found. Please provide a list of continuous features."
        )

    data_dice = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        outcome_name=outcome_name,
    )
    # Wrap the PyTorch model to provide predict_proba interface
    wrapped_model = TorchWrapper(model)
    model_dice = dice_ml.Model(model=wrapped_model, backend="sklearn")
    dice_engine = dice_ml.Dice(data_dice, model_dice, method="random")
    return dice_engine


def generate_counterfactuals(
    dice_engine: dice_ml.Dice,
    X_test_df: pd.DataFrame,
    row_idx: int = 0,
    k: int = 1,
) -> pd.DataFrame:
    """
    Generate counterfactuals for a given test sample using DiCE.
    Args:
        dice_engine: DiCE engine object.
        X_test_df: Test data DataFrame.
        row_idx: Index of sample to explain.
        k: Number of counterfactuals to generate.
    Returns:
        DataFrame with base, counterfactuals, and deltas.
    """
    query_df = X_test_df.iloc[[row_idx]]
    cf = dice_engine.generate_counterfactuals(
        query_instances=query_df,
        total_CFs=k,
        desired_class="opposite",
        proximity_weight=1.0,
        diversity_weight=0.0,
        verbose=False,
    )
    base = query_df.reset_index(drop=True)
    cf_rows = (
        cf.cf_examples_list[0]
        .final_cfs_df.reset_index(drop=True)
        .drop(columns=["y_dummy"], errors="ignore")
        .add_suffix(" CF")
    )
    base_repeated = pd.concat([base] * len(cf_rows), ignore_index=True)
    delta = (
        cf_rows.rename(columns=lambda c: c.replace(" CF", "")) - base_repeated
    ).add_suffix(" Δ")
    out = pd.concat([base_repeated, cf_rows, delta], axis=1)
    return out
