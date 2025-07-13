import pandas as pd
import numpy as np
import torch
import dice_ml


class TorchWrapper:
    def __init__(self, net, thresh=0.5):
        self.net, self.thresh = net.eval(), thresh
        self.dev = next(net.parameters()).device

    def _forward(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        X = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            return self.net(X).cpu().numpy()

    def predict_proba(self, X):
        logits = self._forward(X)
        if logits.shape[1] == 1:
            p1 = 1 / (1 + np.exp(-logits.ravel()))
            return np.column_stack([1 - p1, p1])
        exp = np.exp(logits - logits.max(1, keepdims=True))
        return exp / exp.sum(1, keepdims=True)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.thresh).astype(int)


def get_dice_engine(
    model,
    X_train_df,
    continuous_features=None,
    categorical_features=None,
    outcome_name="y_dummy",
):
    train_df = X_train_df.copy()
    train_df[outcome_name] = 0  # dummy outcome column
    # Use provided feature lists if given, else infer
    if categorical_features is None:
        categorical_features = [c for c in train_df if train_df[c].dtype == "object"]
    if continuous_features is None:
        continuous_features = [
            c for c in train_df if c not in categorical_features + [outcome_name]
        ]
    data_dice = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        outcome_name=outcome_name,
    )
    model_dice = dice_ml.Model(model=model, backend="sklearn")
    dice_engine = dice_ml.Dice(data_dice, model_dice, method="random")
    return dice_engine


def generate_counterfactuals(dice_engine, X_test_df, row_idx=0, k=1):
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
    # Repeat base to match number of counterfactuals
    base_repeated = pd.concat([base] * len(cf_rows), ignore_index=True)
    delta = (
        cf_rows.rename(columns=lambda c: c.replace(" CF", "")) - base_repeated
    ).add_suffix(" Δ")
    out = pd.concat([base_repeated, cf_rows, delta], axis=1)
    return out
