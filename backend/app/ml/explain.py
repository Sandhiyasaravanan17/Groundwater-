"""Tree-SHAP global and local explanations for fitted regressor pipelines."""

import math

import numpy as np
import pandas as pd
from scipy import sparse


class ExplanationError(RuntimeError):
    """Raised when the selected fitted model cannot be explained by Tree SHAP."""


def _feature_groups(feature_names, source_features):
    groups = {}
    for index, name in enumerate(feature_names):
        cleaned = name.split("__", 1)[-1]
        original = next(
            (
                field
                for field in source_features
                if cleaned == field or cleaned.startswith(field + "_")
            ),
            cleaned,
        )
        groups.setdefault(original, []).append(index)
    return groups


def _explain(model, source_features, rows):
    try:
        import shap
    except ImportError as exc:
        raise ExplanationError("SHAP is not installed") from exc

    frame = pd.DataFrame.from_records(rows, columns=source_features)
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["regressor"]
    transformed = preprocessor.transform(frame)
    if sparse.issparse(transformed):
        if not estimator.__class__.__module__.startswith("xgboost"):
            transformed = transformed.toarray()
    if not sparse.issparse(transformed):
        try:
            transformed = np.asarray(transformed, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ExplanationError("Transformed model features are not numeric") from exc
    finite_values = transformed.data if sparse.issparse(transformed) else transformed
    if not np.isfinite(finite_values).all():
        raise ExplanationError("Transformed model features contain non-finite values")
    feature_names = preprocessor.get_feature_names_out()
    groups = _feature_groups(feature_names, source_features)

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
    except Exception as exc:
        raise ExplanationError("Tree SHAP could not explain the selected model") from exc

    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2 or values.shape != transformed.shape:
        raise ExplanationError("SHAP returned an unsupported value shape")

    grouped_values = {}
    for field, indices in groups.items():
        grouped_values[field] = values[:, indices].sum(axis=1)
    predictions = np.asarray(model.predict(frame), dtype=float)
    base_values = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    base_value = float(base_values[0])
    return grouped_values, predictions, base_value


def explain_global(trained, model_name=None, limit=10):
    model_name = model_name or trained["best_model"]
    if model_name not in trained["models"]:
        raise ExplanationError("Select Random Forest or XGBoost")
    values, predictions, base_value = _explain(
        trained["models"][model_name],
        trained["feature_columns"],
        trained["validation_features"],
    )
    ranking = []
    for feature, contributions in values.items():
        importance = float(np.mean(np.abs(contributions)))
        ranking.append({
            "feature": feature,
            "mean_absolute_shap": importance,
            "mean_signed_shap": float(np.mean(contributions)),
        })
    ranking.sort(key=lambda item: item["mean_absolute_shap"], reverse=True)
    return {
        "model": model_name,
        "target": trained["target_column"],
        "validation_records": len(predictions),
        "base_value": base_value,
        "feature_importance": ranking[:limit],
        "method": "Tree SHAP; one-hot contributions are summed back to original predictors",
    }


def explain_local(trained, features, model_name=None):
    """Explain one model-generated estimate using grouped signed SHAP values."""
    model_name = model_name or trained["best_model"]
    if model_name not in trained["models"]:
        raise ExplanationError("Select Random Forest or XGBoost")
    expected_features = trained["feature_columns"]
    if set(features) != set(expected_features):
        raise ExplanationError("Prediction inputs do not match the trained feature set")
    rows, _, base_value = _explain(
        trained["models"][model_name], expected_features, [features]
    )
    prediction = float(
        trained["models"][model_name].predict(pd.DataFrame([features]))[0]
    )
    contributions = [
        {"feature": feature, "shap_value": float(values[0])}
        for feature, values in rows.items()
    ]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    reconstructed = base_value + sum(item["shap_value"] for item in contributions)
    if not math.isclose(reconstructed, prediction, rel_tol=1e-4, abs_tol=1e-2):
        raise ExplanationError("SHAP contributions do not reconcile to the model prediction")
    return {
        "model": model_name,
        "baseline_wqi": base_value,
        "predicted_wqi": prediction,
        "contributions": contributions,
        "interpretation": "Positive contributions increase predicted WQI relative to baseline; negative contributions decrease it.",
    }
