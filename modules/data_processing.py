"""
Data Processing Module
---------------------
Handles data loading, preprocessing, and feature engineering for the Titanic dataset.
Separates data logic from the main application.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Any


def load_titanic_data() -> pd.DataFrame:
    """
    Load the Titanic dataset from seaborn.

    Returns:
        Raw Titanic DataFrame with selected columns and missing values removed
    """
    df = (
        sns.load_dataset("titanic")
        .loc[
            :,
            ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
        ]
        .dropna()
    )
    return df


def create_feature_mappings() -> Dict[str, str]:
    """
    Create mappings from technical feature names to descriptive labels.

    Returns:
        Dictionary mapping technical names to human-readable descriptions
    """
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
    return feature_name_map


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Engineer features from the raw Titanic dataset.

    Args:
        df: Raw Titanic DataFrame

    Returns:
        Tuple of (feature DataFrame, target array)
    """
    # Separate features and target
    X_df = pd.get_dummies(
        df.drop(columns="survived"), columns=["sex", "embarked"], drop_first=True
    )
    y = df["survived"].values

    return X_df, y


def split_and_scale_data(
    X_df: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """
    Split data into train/test sets and apply scaling.

    Args:
        X_df: Feature DataFrame
        y: Target array
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train_df, X_test_df, y_train, y_test, X_train_scaled, X_test_scaled, scaler)
    """
    # Train/test split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    X_test_scaled = scaler.transform(X_test_df.values)

    return X_train_df, X_test_df, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def create_torch_tensors(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays to PyTorch tensors and move to device.

    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training targets
        y_test: Test targets
        device: PyTorch device (CPU/GPU)

    Returns:
        Tuple of (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    """
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def get_feature_names(
    X_df: pd.DataFrame, feature_name_map: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """
    Get original and descriptive feature names.

    Args:
        X_df: Feature DataFrame
        feature_name_map: Mapping from technical to descriptive names

    Returns:
        Tuple of (original_feature_names, descriptive_feature_names)
    """
    orig_feature_names = X_df.columns.tolist()
    feature_names = [feature_name_map.get(f, f) for f in orig_feature_names]

    return orig_feature_names, feature_names


def prepare_titanic_data(
    test_size: float = 0.2, random_state: int = 42, device: torch.device = None
) -> Dict[str, Any]:
    """
    Complete data preparation pipeline for the Titanic dataset.

    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        device: PyTorch device (if None, will auto-detect)

    Returns:
        Dictionary containing all processed data and metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process data
    df = load_titanic_data()
    feature_name_map = create_feature_mappings()
    X_df, y = engineer_features(df)

    # Split and scale
    X_train_df, X_test_df, y_train, y_test, X_train_scaled, X_test_scaled, scaler = (
        split_and_scale_data(X_df, y, test_size, random_state)
    )

    # Create tensors
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = create_torch_tensors(
        X_train_scaled, X_test_scaled, y_train, y_test, device
    )

    # Get feature names
    orig_feature_names, feature_names = get_feature_names(X_df, feature_name_map)

    return {
        # Raw data
        "raw_df": df,
        "feature_name_map": feature_name_map,
        # DataFrames
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        # Numpy arrays
        "y_train_np": y_train,
        "y_test_np": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        # PyTorch tensors
        "X_train": X_train_tensor,
        "X_test": X_test_tensor,
        "y_train": y_train_tensor,
        "y_test": y_test_tensor,
        # Preprocessing objects
        "scaler": scaler,
        "device": device,
        # Feature names
        "orig_feature_names": orig_feature_names,
        "feature_names": feature_names,
        # Metadata
        "train_size": len(X_train_df),
        "test_size": len(X_test_df),
        "n_features": len(orig_feature_names),
        "n_classes": len(np.unique(y)),
    }


def get_data_summary(data_dict: Dict[str, Any]) -> str:
    """
    Generate a summary string of the prepared data.

    Args:
        data_dict: Dictionary returned by prepare_titanic_data()

    Returns:
        Formatted summary string
    """
    summary = f"""
📊 Titanic Dataset Summary
========================
• Total samples: {data_dict['train_size'] + data_dict['test_size']}
• Training samples: {data_dict['train_size']}
• Test samples: {data_dict['test_size']}
• Features: {data_dict['n_features']}
• Classes: {data_dict['n_classes']} (survived/not survived)
• Device: {data_dict['device']}

Features:
{', '.join(data_dict['orig_feature_names'])}

Descriptive Names:
{', '.join(data_dict['feature_names'])}
"""
    return summary.strip()
