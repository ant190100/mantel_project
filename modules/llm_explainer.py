import numpy as np
import textwrap
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
from typing import List, Any, Optional

load_dotenv()

HF_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Load API key from .env file (preferred) or environment
HF_API_KEY: Optional[str] = os.getenv("HF_API_KEY")
HF_CLIENT = InferenceClient(model=HF_MODEL, timeout=45, api_key=HF_API_KEY)


def explain_with_citations(
    shap_values: Any,
    x_row: List[Any],
    feature_names: List[str],
    target_name: str = "prediction",
    top_k: int = 5,
    temperature: float = 0.4,
    max_new_tokens: int = 500,
) -> str:
    """
    Generate a plain-English explanation of SHAP values for a sample, with inline [n] citations.
    Uses HuggingFace LLM to produce a concise, business-friendly summary.

    Args:
        shap_values: SHAP values for the sample (array-like).
        x_row: Feature values for the sample (array-like).
        feature_names: List of descriptive feature names.
        target_name: Name of the prediction target.
        top_k: Number of top features to include in the explanation.
        temperature: LLM sampling temperature.
        max_new_tokens: Maximum tokens for LLM response.

    Returns:
        str: LLM-generated explanation string.
    """
    shap_arr = np.asarray(shap_values, dtype=float).ravel()
    n = min(len(shap_arr), len(feature_names), len(x_row))
    shap_arr, feature_names, x_row = shap_arr[:n], feature_names[:n], x_row[:n]

    idx_sorted = np.argsort(np.abs(shap_arr))[::-1][:top_k]
    context = "\n".join(
        f"[{i}] **{feature_names[idx]}** = {x_row[idx]} (SHAP {shap_arr[idx]:+.3f})"
        for i, idx in enumerate(idx_sorted, 1)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that explains machine-learning predictions "
                "to non-technical business stakeholders. When referencing a feature, "
                "use the exact bold-face name provided in the facts."
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""\
                Your response must be no more than 2 sentences and under {max_new_tokens - 100} tokens.
                Below are the most influential features for one sample and their SHAP contributions.
                • Write concise plain-English sentences that explain **why** the model arrived at the current {target_name}.
                • Whenever you reference a feature, copy its bold name and append the citation key (e.g. **Age** [1]).
                • Quote all decimal figures to 2 places. 
                • Do **not** invent any information beyond the facts block.

                ### Facts
                {context}
                """
            ),
        },
    ]

    try:
        resp = HF_CLIENT.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating LLM explanation: {e}"
