"""
Deployment utilities for
Condition-Based Crop Recommendation System

- load_model: lazy-load trained pipeline
- get_feature_columns: resolve expected feature order
- predict_single: recommend best crop + top-k alternatives
- predict_batch: batch predictions with probabilities
"""

from typing import Any, Dict, List, Optional
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, FEATURE_PATH


# Caches


_model = None
_feature_cols_cache: Optional[List[str]] = None



# Load model (lazy)


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(BEST_MODEL_PATH)
    return _model



# Resolve feature columns


def get_feature_columns(model) -> List[str]:
    global _feature_cols_cache

    if _feature_cols_cache is not None:
        return _feature_cols_cache

    # 1. Try sklearn pipeline metadata
    try:
        if hasattr(model, "feature_names_in_"):
            _feature_cols_cache = list(model.feature_names_in_)
            return _feature_cols_cache
    except Exception:
        pass

    # 2. Load from feature_columns.json (preferred)
    if Path(FEATURE_PATH).exists():
        with open(FEATURE_PATH, "r", encoding="utf-8") as f:
            cols = json.load(f)
        if isinstance(cols, list):
            _feature_cols_cache = cols
            return _feature_cols_cache

    raise ValueError(
        "Could not determine feature columns. "
        "Ensure feature_columns.json exists."
    )



# Input alignment helpers


def _ensure_input_frame(input_obj: Any, feature_cols: List[str]) -> pd.DataFrame:
    if isinstance(input_obj, dict):
        row = {c: input_obj.get(c, np.nan) for c in feature_cols}
        return pd.DataFrame([row])
    elif isinstance(input_obj, pd.DataFrame):
        return input_obj.reindex(columns=feature_cols)
    else:
        raise TypeError("Input must be dict or pandas DataFrame")


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



# Single prediction (TOP-K crops)


def predict_single(
    input_data: Dict[str, Any],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Predict best crop and top-k alternatives
    """

    model = load_model()
    feature_cols = get_feature_columns(model)

    df_input = _ensure_input_frame(input_data, feature_cols)
    df_input = _coerce_numeric(df_input)

    try:
        probs = model.predict_proba(df_input)[0]
        classes = model.classes_
    except Exception as e:
        raise ValueError("Prediction failed: " + str(e)) from e

    # Sort by probability
    ranked = sorted(
        zip(classes, probs),
        key=lambda x: x[1],
        reverse=True
    )

    top_recommendations = [
        {"crop": crop, "probability": float(prob)}
        for crop, prob in ranked[:top_k]
    ]

    best_crop = top_recommendations[0]

    return {
        "best_crop": best_crop["crop"],
        "confidence": best_crop["probability"],
        "top_recommendations": top_recommendations
    }



# Batch prediction


def predict_batch(
    df: pd.DataFrame,
    top_k: int = 3
) -> pd.DataFrame:
    """
    Predict crops for a batch of inputs
    """

    model = load_model()
    feature_cols = get_feature_columns(model)

    df_input = df.reindex(columns=feature_cols)
    df_input = _coerce_numeric(df_input)

    try:
        probas = model.predict_proba(df_input)
        preds = model.predict(df_input)
        classes = model.classes_
    except Exception as e:
        raise ValueError("Batch prediction failed: " + str(e)) from e

    out = df.copy()
    out["best_crop"] = preds
    out["confidence"] = probas.max(axis=1)

