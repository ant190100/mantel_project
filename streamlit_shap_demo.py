import streamlit as st
import torch
import pandas as pd
import numpy as np
import shap
from modules.shap_tools import (
    create_kernel_explainer,
    explain_sample,
    simulate_scenario,
)
from modules.cf_generator import TorchWrapper, get_dice_engine, generate_counterfactuals
from modules.model_training import (
    train_model,
    evaluate_model,
    create_model_prediction_function,
)
from modules.data_processing import prepare_titanic_data

st.title("Titanic SHAP Model Interpretation Toolbox")


# --- Load & preprocess data using the new module ---
@st.cache_data
def get_processed_data():
    """Cache the processed data to avoid reprocessing on every app reload."""
    return prepare_titanic_data(test_size=0.2, random_state=42)


data = get_processed_data()

# Extract commonly used variables from the data dictionary
X_train_df = data["X_train_df"]
X_test_df = data["X_test_df"]
y_train_np = data["y_train_np"]
y_test_np = data["y_test_np"]
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
scaler = data["scaler"]
device = data["device"]
orig_feature_names = data["orig_feature_names"]
feature_names = data["feature_names"]
feature_name_map = data["feature_name_map"]


# --- Train the model using the new module ---
@st.cache_resource
def get_trained_model():
    """Cache the trained model to avoid retraining on every app reload."""
    model = train_model(X_train, y_train, device, n_epochs=50, lr=1e-3)
    return model


model = get_trained_model()

# --- Model Performance Evaluation ---
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = evaluate_model(
        model, X_train, y_train_np, X_test, y_test_np, device
    )

# --- SHAP Explainer Setup ---
if "explainer" not in st.session_state:
    st.session_state.explainer = create_kernel_explainer(
        model, device, X_train, feature_names
    )

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")

# Model Performance Metrics
with st.sidebar.expander("📊 Model Performance"):
    metrics = st.session_state.model_metrics
    st.metric("Train Accuracy", f"{metrics['train_accuracy']:.3f}")
    st.metric("Test Accuracy", f"{metrics['test_accuracy']:.3f}")

    # Classification metrics
    report = metrics["classification_report"]
    st.write("**Test Set Metrics:**")
    st.write(f"Precision: {report['weighted avg']['precision']:.3f}")
    st.write(f"Recall: {report['weighted avg']['recall']:.3f}")
    st.write(f"F1-Score: {report['weighted avg']['f1-score']:.3f}")

    # Confusion Matrix
    cm = metrics["confusion_matrix"]
    st.write("**Confusion Matrix:**")
    st.write(f"True Negatives: {cm[0,0]}")
    st.write(f"False Positives: {cm[0,1]}")
    st.write(f"False Negatives: {cm[1,0]}")
    st.write(f"True Positives: {cm[1,1]}")

# Main Navigation
tab = st.sidebar.radio(
    "Select a tool:",
    ["SHAP Sample Explainer", "What-If Scenario Simulator", "Counterfactual Generator"],
)

if tab == "SHAP Sample Explainer":
    st.subheader("Quick Access: Sample Passengers")
    # Get some interesting passenger indices with their basic info
    sample_passengers = [
        {"idx": 0, "desc": "Young male, 3rd class", "prediction": "Did not survive"},
        {"idx": 15, "desc": "Adult female, 1st class", "prediction": "Survived"},
        {"idx": 45, "desc": "Child, 3rd class", "prediction": "Survived"},
        {"idx": 67, "desc": "Elderly male, 2nd class", "prediction": "Did not survive"},
        {"idx": 102, "desc": "Young couple, 2nd class", "prediction": "Varied"},
    ]

    cols = st.columns(len(sample_passengers))
    for i, passenger in enumerate(sample_passengers):
        with cols[i]:
            if st.button(
                f"{passenger['desc']}\n(#{passenger['idx']})",
                key=f"sample_{passenger['idx']}",
            ):
                # Set the passenger index in session state for all tabs to use
                st.session_state.selected_passenger_idx = passenger["idx"]
                st.rerun()

    st.subheader("Model Prediction Explanation for Individual Passengers")

    # Use selected passenger from gallery if available
    default_idx = st.session_state.get("selected_passenger_idx", 0)
    idx = st.number_input("Passenger index", min_value=0, step=1, value=default_idx)
    max_display = st.slider(
        "Top features to display", min_value=1, max_value=8, value=8
    )

    import matplotlib.pyplot as plt
    import io

    # Button to generate sample explanation and store in session state
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
        st.session_state.sample_explanation = result
        st.session_state.llm_blurb = None  # Clear previous LLM output

    # Only show explanation and LLM button if sample_explanation exists
    if "sample_explanation" in st.session_state:
        result = st.session_state.sample_explanation
        st.write("Sample feature values:")
        raw_row_named = result["raw_row"]
        raw_row_named.index = feature_names
        # Display as a dataframe with feature names as row labels
        feature_df = raw_row_named.to_frame(name="Value")
        feature_df.index.name = "Feature"
        st.dataframe(feature_df)
        st.write("SHAP Waterfall Plot:")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(result["exp"], max_display=max_display, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        st.image(buf)
        with st.expander("ℹ️ How to read the SHAP Waterfall Plot"):
            st.markdown(
                """
                **SHAP Waterfall Plot - Quick Guide:**
                
                - **Starting point, E[f(x)]**: Average prediction across all passengers
                - **Red bars**: Features pushing prediction toward survival
                - **Blue bars**: Features pushing prediction toward non-survival
                - **Numbers on y-ticks**: Normalised feature values - mean of 0, sd of 1
                - **Final point, f(x)**: Model's prediction for this passenger
                
                The plot shows how each feature contributes to the final prediction, from most to least influential.
                """
            )
        # LLM explainer button
        if st.button("Explain with LLM"):
            from modules.llm_explainer import explain_with_citations

            st.session_state.llm_blurb = explain_with_citations(
                shap_values=result["exp"].values,
                x_row=result["exp"].data,
                feature_names=result["exp"].feature_names,
                target_name="prediction",
                top_k=max_display,
            )
        if st.session_state.get("llm_blurb"):
            st.markdown(f"**LLM Explanation:**\n\n{st.session_state.llm_blurb}")

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
        # Get prediction function from the model module
        predict_proba, predict_survival = create_model_prediction_function(
            model, device
        )

        result = simulate_scenario(
            model_np=predict_proba,  # This returns full probability matrix
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
        # Transpose to show features as rows and add proper labels
        feature_display = raw_df_named.T
        feature_display.columns = ["Value"]
        feature_display.index.name = "Feature"
        st.dataframe(feature_display)
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
        with st.expander("ℹ️ How to read the SHAP Waterfall Plot"):
            st.markdown(
                """
                **SHAP Waterfall Plot - Quick Guide:**
                
                - **Starting point, E[f(x)]**: Average prediction across all passengers
                - **Red bars**: Features pushing prediction toward survival
                - **Blue bars**: Features pushing prediction toward non-survival
                - **Numbers on y-ticks**: Normalised feature values - mean of 0, sd of 1
                - **Final point, f(x)**: Model's prediction for this passenger
                
                The plot shows how each feature contributes to the final prediction, from most to least influential.
                """
            )

else:  # Counterfactual Generator
    st.subheader("Counterfactual Explanation Generator")

    # Use selected passenger from gallery if available
    default_idx = st.session_state.get("selected_passenger_idx", 0)
    idx = st.number_input(
        "Passenger index",
        min_value=0,
        step=1,
        value=default_idx,
        key="cf_passenger_idx",
    )
    num_counterfactuals = st.slider(
        "Number of counterfactuals to generate", min_value=1, max_value=10, value=1
    )

    # --- DICE Counterfactuals ---
    st.write("### DICE Counterfactuals")
    dice_button = st.button("Generate DICE Counterfactuals")
    if dice_button:
        with st.spinner("Generating counterfactuals..."):
            # Ensure all columns are int or float for DiCE
            X_train_df_fixed = X_train_df.copy()
            for col in X_train_df_fixed.columns:
                # Convert bool to int
                if pd.api.types.is_bool_dtype(X_train_df_fixed[col]):
                    X_train_df_fixed[col] = X_train_df_fixed[col].astype(int)
                # Convert object to float (if possible)
                elif pd.api.types.is_object_dtype(X_train_df_fixed[col]):
                    try:
                        X_train_df_fixed[col] = X_train_df_fixed[col].astype(float)
                    except Exception:
                        try:
                            X_train_df_fixed[col] = X_train_df_fixed[col].astype(int)
                        except Exception:
                            # Try mapping True/False strings to 1/0
                            if set(X_train_df_fixed[col].unique()) <= {"True", "False"}:
                                X_train_df_fixed[col] = X_train_df_fixed[col].map(
                                    {"True": 1, "False": 0}
                                )
                            else:
                                raise ValueError(
                                    f"Column {col} could not be converted to int or float for DiCE. Values: {X_train_df_fixed[col].unique()}"
                                )
                # Convert anything else that's not int/float
                elif not (
                    pd.api.types.is_integer_dtype(X_train_df_fixed[col])
                    or pd.api.types.is_float_dtype(X_train_df_fixed[col])
                ):
                    try:
                        X_train_df_fixed[col] = X_train_df_fixed[col].astype(int)
                    except Exception:
                        X_train_df_fixed[col] = X_train_df_fixed[col].astype(float)
            # Final check: print dtypes and raise error if any column is not int/float
            bad_cols = [
                col
                for col in X_train_df_fixed.columns
                if not (
                    pd.api.types.is_integer_dtype(X_train_df_fixed[col])
                    or pd.api.types.is_float_dtype(X_train_df_fixed[col])
                )
            ]
            if bad_cols:
                print(
                    "Bad columns and their values:",
                    {col: X_train_df_fixed[col].unique() for col in bad_cols},
                )
                raise ValueError(
                    f"These columns are not int or float and will cause DiCE to fail: {bad_cols}"
                )
            # Setup DiCE engine if not already present
            if "dice_engine" not in st.session_state:
                st.session_state.dice_engine = get_dice_engine(model, X_train_df_fixed)
            dice_engine = st.session_state.dice_engine
            # Use the correct test dataframe and index
            result_df = generate_counterfactuals(
                dice_engine,
                X_test_df,
                row_idx=idx,
                k=num_counterfactuals,
            )

        st.write("Sample features:")
        # Display sample features with descriptive names
        sample_features = X_test_df.iloc[idx : idx + 1].copy()
        sample_features.columns = feature_names
        sample_features.index = [f"Passenger {idx}"]
        sample_features.index.name = "Passenger"
        st.dataframe(sample_features)

        st.write("Generated Counterfactuals:")
        # Display counterfactuals with descriptive names
        if result_df is not None and not result_df.empty:
            result_df_display = result_df.copy()
            # Create column names for counterfactual and delta columns only (skip original)
            if len(result_df_display.columns) == len(feature_names) * 3:
                # Skip base columns, keep only CF columns + Delta columns
                num_features = len(feature_names)
                result_df_display = result_df_display.iloc[
                    :, num_features:
                ]  # Skip first third
                new_columns = []
                new_columns.extend(
                    [f"{name} (Counterfactual)" for name in feature_names]
                )
                new_columns.extend([f"{name} (Change)" for name in feature_names])
                result_df_display.columns = new_columns
            st.dataframe(result_df_display, hide_index=True)
        else:
            st.write("No counterfactuals generated.")
