import streamlit as st
import torch
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from modules.shap_tools import (
    create_kernel_explainer,
    explain_sample,
    simulate_scenario,
)
from modules.cf_generator import TorchWrapper, get_dice_engine, generate_counterfactuals

st.title("Titanic SHAP Model Interpretation Toolbox")

# --- Load & preprocess Titanic dataset ---
df = (
    sns.load_dataset("titanic")
    .loc[:, ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
    .dropna()
)
X_df = pd.get_dummies(
    df.drop(columns="survived"), columns=["sex", "embarked"], drop_first=True
)
y = df["survived"].values

# Map feature names to more descriptive labels
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

X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
    X_df, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_df.values)
X_test_np = scaler.transform(X_test_df.values)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
y_test = torch.tensor(y_test_np, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define and train the model ---
import torch.nn as nn
import torch.optim as optim


class TitanicNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)


model = TitanicNet(X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 50
for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train.to(device))
    loss = loss_fn(logits, y_train.to(device))
    loss.backward()
    optimizer.step()
model.eval()

# --- SHAP Explainer Setup ---
if "explainer" not in st.session_state:
    st.session_state.explainer = create_kernel_explainer(
        model, device, X_train, feature_names
    )

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
tab = st.sidebar.radio(
    "Select a tool:",
    ["SHAP Sample Explainer", "What-If Scenario Simulator", "Counterfactual Generator"],
)

if tab == "SHAP Sample Explainer":
    st.subheader("Model Prediction Explanation for Individual Passengers")
    idx = st.number_input("Passenger index", min_value=0, step=1)
    max_display = st.slider(
        "Top features to display", min_value=1, max_value=8, value=8
    )

    import matplotlib.pyplot as plt
    import io

    if st.button("Explain Sample"):
        result = explain_sample(
            st.session_state.explainer,
            X_test,
            X_test_df,
            y_test,
            model,
            device,
            feature_names,
            idx,
            nsamples=500,
            max_display=max_display,
        )
        st.write("Sample feature values:")
        # Map raw_row index to descriptive feature names
        raw_row_named = result["raw_row"]
        raw_row_named.index = feature_names
        st.dataframe(raw_row_named.to_frame(name="value"))
        st.write("SHAP Waterfall Plot:")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(result["exp"], max_display=max_display, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        st.image(buf)
        st.markdown(
            """
        **How to read the SHAP Waterfall Plot:**

        - The plot explains the model's prediction for the selected passenger.
        - The leftmost value (**E[f(x)]**) is the average model output (base value) for the background data.
        - Each bar shows how a feature pushes the prediction higher or lower.
        - Features in red push the prediction up, blue push it down.
        - The rightmost value (**f(x)**) is the final model output for this passenger.
        - The sum of all feature effects plus the base value equals the model's prediction.
        - **The value next to each feature label is the standardized (scaled) value used by the model for that feature:**
          it shows how far above or below average this passenger's feature is, after preprocessing. This helps explain why the SHAP value is positive or negative.

        *Use the sliders to adjust which features are shown.*
        """
        )

elif tab == "What-If Scenario Simulator":
    st.subheader("What-If Scenario Simulator")
    pclass = st.selectbox(
        "Passenger Class (1=1st, 2=2nd, 3=3rd)", [1, 2, 3], index=2, key="pclass_whatif"
    )
    sex = st.selectbox("Sex", ["female", "male"], index=1, key="sex_whatif")
    age = st.slider(
        "Age", min_value=0.0, max_value=80.0, value=30.0, step=0.5, key="age_whatif"
    )
    sibsp = st.slider(
        "Siblings/Spouses Aboard", min_value=0, max_value=5, value=0, key="sibsp_whatif"
    )
    parch = st.slider(
        "Parents/Children Aboard", min_value=0, max_value=5, value=0, key="parch_whatif"
    )
    fare = st.slider(
        "Fare (Ticket Price)",
        min_value=0.0,
        max_value=300.0,
        value=32.0,
        step=1.0,
        key="fare_whatif",
    )
    embarked = st.selectbox(
        "Embarked (C=Cherbourg, Q=Queenstown, S=Southampton)",
        ["C", "Q", "S"],
        index=2,
        key="embarked_whatif",
    )
    max_display = st.slider(
        "Top features to display",
        min_value=1,
        max_value=8,
        value=8,
        key="max_display_whatif",
    )

    if st.button("Simulate Scenario"):
        result = simulate_scenario(
            model_np=lambda x: (
                torch.softmax(
                    (
                        lambda x_: model(
                            torch.tensor(x_, dtype=torch.float32, device=device)
                        ).detach()
                    )(x),
                    dim=1,
                )
                .cpu()
                .numpy()
            ),
            scaler=scaler,
            feature_names=orig_feature_names,  # Use original names for processing
            pclass=pclass,
            sex=sex,
            age=age,
            sibsp=sibsp,
            parch=parch,
            fare=fare,
            embarked=embarked,
            nsamples=500,
            max_display=max_display,
            explainer=st.session_state.explainer,
            # Pass descriptive names for SHAP plot via explainer
        )
        st.write("Raw input:")
        # Display raw input with descriptive names
        raw_df_named = result["raw_df"].copy()
        raw_df_named.columns = [
            feature_name_map.get(c, c) for c in raw_df_named.columns
        ]
        st.dataframe(raw_df_named.T)
        st.write("SHAP Waterfall Plot:")
        import matplotlib.pyplot as plt
        import io

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            result["shap_explanation"], max_display=max_display, show=False
        )
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        st.image(buf)
        st.markdown(
            """
            **How to read the SHAP Waterfall Plot:**
            - The plot explains the model's prediction for your custom scenario.
            - The leftmost value (**E[f(x)]**) is the average model output (base value) for the background data.
            - Each bar shows how a feature pushes the prediction higher or lower.
            - Features in red push the prediction up, blue push it down.
            - The rightmost value (**f(x)**) is the final model output for this scenario.
            - The sum of all feature effects plus the base value equals the model's prediction.
            - **The value next to each feature label is the standardized (scaled) value used by the model for that feature.**
            """
        )

elif tab == "Counterfactual Generator":
    st.subheader("Counterfactual Generator (DiCE)")
    # Ensure all columns are int or float for DiCE
    X_train_df_fixed = X_train_df.copy()
    for col in X_train_df_fixed.columns:
        # Convert all integer columns to int64, float columns to float64
        if pd.api.types.is_integer_dtype(X_train_df_fixed[col]):
            X_train_df_fixed[col] = X_train_df_fixed[col].astype(int)
        elif pd.api.types.is_float_dtype(X_train_df_fixed[col]):
            X_train_df_fixed[col] = X_train_df_fixed[col].astype(float)
        else:
            X_train_df_fixed[col] = (
                pd.to_numeric(X_train_df_fixed[col], errors="coerce")
                .fillna(0)
                .astype(int)
            )

    # Identify categorical and continuous features
    categorical = []  # All columns are now numeric, so categorical is empty
    continuous = [c for c in X_train_df_fixed.columns]

    # Build DiCE engine (cache for performance)
    if "dice_engine" not in st.session_state:
        st.session_state.dice_engine = get_dice_engine(
            TorchWrapper(model),
            X_train_df_fixed,
            continuous_features=continuous,
            categorical_features=categorical,
            outcome_name="y_dummy",
        )
    dice_engine = st.session_state.dice_engine
    row_idx = st.number_input(
        "Test row index", min_value=0, max_value=len(X_test_df) - 1, value=0, step=1
    )
    k = st.slider("Number of counterfactuals", min_value=1, max_value=5, value=1)
    if st.button("Generate Counterfactuals"):
        cf_table = generate_counterfactuals(
            dice_engine, X_test_df, row_idx=row_idx, k=k
        )
        # Separate base/sample, counterfactual, and delta columns
        base_cols = [
            c
            for c in cf_table.columns
            if not c.endswith(" CF") and not c.endswith(" Δ")
        ]
        cf_cols = [c for c in cf_table.columns if c.endswith(" CF")]
        delta_cols = [c for c in cf_table.columns if c.endswith(" Δ")]

        st.write("Sample Feature Values:")
        # Show only the first row for sample features
        st.dataframe(cf_table[base_cols].iloc[[0]].style.format(precision=3))
        st.write("Counterfactual Changes:")
        cf_and_delta = pd.concat([cf_table[cf_cols], cf_table[delta_cols]], axis=1)
        st.dataframe(cf_and_delta.style.format(precision=3))
