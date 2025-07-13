# SHAP tools for model interpretation
import shap
import numpy as np
import torch
import pandas as pd
import random

# Set a fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Helper to create a SHAP KernelExplainer for a torch model


def create_kernel_explainer(model, device, X_train, feature_names, bg_size=None):
    # Use all rows if bg_size is None
    if bg_size is None:
        bg_np = X_train.cpu().numpy()
    else:
        bg_np = X_train[:bg_size].cpu().numpy()

    def model_np(x_array):
        t = torch.tensor(x_array, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1).cpu().numpy()
        return probs[:, 1]  # return P(survived)

    explainer = shap.KernelExplainer(model_np, bg_np)
    return explainer


# Helper to explain a sample and show SHAP waterfall plot


def explain_sample(
    explainer,
    X_test,
    X_test_df,
    y_test,
    model,
    device,
    feature_names,
    idx,
    nsamples=None,
    max_display=10,
):
    raw_row = X_test_df.iloc[idx]
    # scaled array for SHAP & model
    x_np = X_test[idx].cpu().numpy().reshape(1, -1)
    # model probs
    with torch.no_grad():
        logits = model(torch.tensor(x_np, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # SHAP values
    # Use all samples if nsamples is None
    if nsamples is None:
        nsamples = X_test.shape[0]
    raw_sv = explainer.shap_values(x_np, nsamples=nsamples)
    sv = raw_sv[0]
    bv = explainer.expected_value
    exp = shap.Explanation(
        values=sv, base_values=bv, data=x_np[0], feature_names=feature_names
    )
    return {"raw_row": raw_row, "probs": probs, "exp": exp}


# Helper to create a SHAP explainer for pandas DataFrame input


def create_df_explainer(model, X_train_df, sample_size=100):
    background = X_train_df.sample(sample_size, random_state=42)

    def predict_df(data):
        arr = np.asarray(data, dtype=np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(arr).to(next(model.parameters()).device)
            y = model(x_t).cpu().numpy()
        return y

    explainer = shap.Explainer(predict_df, background)
    return explainer


# Helper to compute SHAP values for a single row


def compute_shap_for_row(explainer, X_test_df, row):
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
):
    """
    Simulate a custom scenario for the Titanic model and show SHAP explanation.
    Returns a dict with raw input, probabilities, and SHAP explanation.
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

        exp = shap.Explanation(
            values=sv, base_values=bv, data=X_scaled[0], feature_names=feature_names
        )
    return {"raw_df": raw_df, "probs": probs, "shap_explanation": exp}
