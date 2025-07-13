"""
Model Training Module
--------------------
Contains the TitanicNet model definition and training utilities.
Separates model logic from the Streamlit application.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Tuple, Any
import pandas as pd


class TitanicNet(nn.Module):
    """
    Simple neural network for Titanic survival prediction.
    Uses a single linear layer for binary classification.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    verbose: bool = False,
) -> TitanicNet:
    """
    Train the TitanicNet model on the provided data.

    Args:
        X_train: Training features tensor
        y_train: Training labels tensor
        device: Device to train on (CPU/GPU)
        n_epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print training progress

    Returns:
        Trained TitanicNet model
    """
    model = TitanicNet(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        logits = model(X_train.to(device))
        loss = loss_fn(logits, y_train.to(device))
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}")

    model.eval()
    return model


def evaluate_model(
    model: TitanicNet,
    X_train: torch.Tensor,
    y_train: np.ndarray,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate the trained model and return performance metrics.

    Args:
        model: Trained TitanicNet model
        X_train: Training features tensor
        y_train: Training labels numpy array
        X_test: Test features tensor
        y_test: Test labels numpy array
        device: Device model is on

    Returns:
        Dictionary containing performance metrics
    """
    model.eval()

    with torch.no_grad():
        # Training predictions
        train_logits = model(X_train.to(device))
        train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()
        train_accuracy = (train_preds == y_train).mean()

        # Test predictions
        test_logits = model(X_test.to(device))
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
        test_accuracy = (test_preds == y_test).mean()

    # Detailed classification metrics
    classification_report_dict = classification_report(
        y_test, test_preds, output_dict=True
    )
    confusion_matrix_result = confusion_matrix(y_test, test_preds)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "classification_report": classification_report_dict,
        "confusion_matrix": confusion_matrix_result,
        "test_predictions": test_preds,
        "train_predictions": train_preds,
    }


def create_model_prediction_function(model: TitanicNet, device: torch.device):
    """
    Create a prediction function wrapper for the model.
    Useful for SHAP and other explanation libraries.

    Args:
        model: Trained TitanicNet model
        device: Device model is on

    Returns:
        Function that takes numpy array and returns probabilities
    """

    def predict_proba(x_array: np.ndarray) -> np.ndarray:
        """Prediction function that returns survival probabilities."""
        model.eval()
        x_tensor = torch.tensor(x_array, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_survival_prob(x_array: np.ndarray) -> np.ndarray:
        """Prediction function that returns only survival probability (class 1)."""
        probs = predict_proba(x_array)
        return probs[:, 1]  # Return P(survived)

    return predict_proba, predict_survival_prob


def save_model(model: TitanicNet, filepath: str) -> None:
    """
    Save the trained model to disk.

    Args:
        model: Trained TitanicNet model
        filepath: Path to save the model
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_architecture": "TitanicNet",
            "input_dim": model.fc.in_features,
        },
        filepath,
    )


def load_model(filepath: str, device: torch.device) -> TitanicNet:
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model
        device: Device to load the model on

    Returns:
        Loaded TitanicNet model
    """
    checkpoint = torch.load(filepath, map_location=device)

    model = TitanicNet(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model
