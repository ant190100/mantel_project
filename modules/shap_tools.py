"""
SHAP Tools Module
-----------------
Provides helpers for creating SHAP explainers and generating explanations for model predictions.
All functions are type-annotated and robust to input errors.
"""

import shap
import numpy as np
import torch
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import random
from typing import Any, Optional, List

# Set a fixed random seed for reproducibility
SEED: int = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


def create_kernel_explainer(
    model: torch.nn.Module,
    device: torch.device,
    X_train: torch.Tensor,
    feature_names: List[str],
    bg_size: Optional[int] = None,
) -> shap.KernelExplainer:
    """
    Create a SHAP KernelExplainer for a torch model.
    Args:
        model: Trained torch model.
        device: Torch device.
        X_train: Training data tensor.
        feature_names: List of feature names.
        bg_size: Number of background samples to use.
    Returns:
        SHAP KernelExplainer object.
    """
    if bg_size is None:
        bg_np = X_train.cpu().numpy()
    else:
        bg_np = X_train[:bg_size].cpu().numpy()

    def model_np(x_array: np.ndarray) -> np.ndarray:
        t = torch.tensor(x_array, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1).cpu().numpy()
        return probs[:, 1]  # return P(survived)

    explainer = shap.KernelExplainer(model_np, bg_np)
    return explainer


def explain_sample(
    explainer: shap.KernelExplainer,
    X_test: torch.Tensor,
    X_test_df: pd.DataFrame,
    y_test: Any,
    model: torch.nn.Module,
    device: torch.device,
    feature_names: List[str],
    idx: int,
    nsamples: Optional[int] = None,
    max_display: int = 10,
    feature_name_map: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Explain a sample and return SHAP values and model probabilities.
    Args:
        explainer: SHAP explainer object.
        X_test: Test data tensor.
        X_test_df: Test data DataFrame.
        y_test: Test labels.
        model: Trained torch model.
        device: Torch device.
        feature_names: List of feature names.
        idx: Index of sample to explain.
        nsamples: Number of samples for SHAP.
        max_display: Max features to display.
        feature_name_map: Optional mapping from technical feature names to human-readable names
    Returns:
        dict with raw_row, probs, exp (SHAP Explanation)
    """
    raw_row = X_test_df.iloc[idx]
    x_np = X_test[idx].cpu().numpy().reshape(1, -1)
    with torch.no_grad():
        logits = model(torch.tensor(x_np, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    if nsamples is None:
        nsamples = X_test.shape[0]
    raw_sv = explainer.shap_values(x_np, nsamples=nsamples)
    sv = raw_sv[0]
    bv = explainer.expected_value
    
    # Use human-readable feature names if provided
    display_names = feature_names
    if feature_name_map is not None:
        display_names = [feature_name_map.get(name, name) for name in feature_names]
        
    exp = shap.Explanation(
        values=sv, base_values=bv, data=x_np[0], feature_names=display_names
    )
    return {"raw_row": raw_row, "probs": probs, "exp": exp}


def create_df_explainer(
    model: torch.nn.Module,
    X_train_df: pd.DataFrame,
    sample_size: int = 100,
) -> shap.KernelExplainer:
    """
    Create a SHAP explainer for pandas DataFrame input.
    Args:
        model: Trained torch model.
        X_train_df: Training data DataFrame.
        sample_size: Number of background samples.
    Returns:
        SHAP KernelExplainer object.
    """
    background = X_train_df.sample(sample_size, random_state=42)

    def predict_df(data: Any) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(arr).to(next(model.parameters()).device)
            y = model(x_t).cpu().numpy()
        return y

    explainer = shap.Explainer(predict_df, background)
    return explainer


def compute_shap_for_row(explainer, X_test_df, row):
    """
    Compute SHAP values for a single row.
    Args:
        explainer: SHAP explainer object.
        X_test_df: Test data DataFrame.
        row: Row index to compute SHAP values for.
    Returns:
        Array of SHAP values for the row.
    """
    shap_1d = explainer(X_test_df.iloc[[row]]).values[0]
    shap_1d = shap_1d[: len(X_test_df.columns)]
    return shap_1d


def simulate_scenario(
    model_np,
    scaler,
    feature_names,
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked,
    nsamples=None,
    max_display=10,
    explainer=None,
    feature_name_map=None,
):
    """
    Simulate a custom scenario for the Titanic model and show SHAP explanation.
    Returns a dict with raw input, probabilities, and SHAP explanation.
    
    Args:
        feature_name_map: Optional mapping from technical feature names to human-readable names
    """
    import pandas as pd

    # 1) Build raw input dict
    raw = {
        "pclass": pclass,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        # one-hot encoding
        "sex_male": 1 if sex == "male" else 0,
        "embarked_Q": 1 if embarked == "Q" else 0,
        "embarked_S": 1 if embarked == "S" else 0,
    }
    raw_df = pd.DataFrame([raw])
    # 2) Scale numeric features & assemble numpy row
    X_row = raw_df[feature_names].values
    X_scaled = scaler.transform(X_row)
    # 3) Predict
    probs = model_np(X_scaled)[0]
    # 4) SHAP explanation
    sv = None
    bv = None
    exp = None
    if explainer is not None:
        # Use all samples if nsamples is None
        if nsamples is None:
            nsamples = X_scaled.shape[0]
        shap_values = explainer.shap_values(X_scaled, nsamples=nsamples)
        # For multiclass, use class 1 (survived)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            sv = shap_values[1][0]  # [1] for class 1, [0] for the single sample
        else:
            sv = shap_values[0]
        bv = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            and len(explainer.expected_value) > 1
            else explainer.expected_value
        )
        import shap

        # Use human-readable feature names if provided
        display_names = feature_names
        if feature_name_map is not None:
            display_names = [feature_name_map.get(name, name) for name in feature_names]
            
        exp = shap.Explanation(
            values=sv, base_values=bv, data=X_scaled[0], feature_names=display_names
        )
    return {"raw_df": raw_df, "probs": probs, "shap_explanation": exp}
