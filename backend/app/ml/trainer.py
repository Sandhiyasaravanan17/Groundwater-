"""Train reproducible RF/XGBoost regressors only with a numeric WQI target."""

import math


TARGET_COLUMN = "wqi"
GROUP_COLUMN = "village"
RANDOM_STATE = 42
MINIMUM_RECORDS = 10
MEASUREMENT_COLUMNS = frozenset({
    "nitrate_mg_l",
    "fluoride_mg_l",
    "chloride_mg_l",
    "tds_mg_l",
    "ph",
    "iron_mg_l",
    "alkalinity_mg_l",
})
TARGET_DERIVED_COLUMNS = frozenset({
    "quality_class_source",
    "risk_class_source",
    "wqi_class",
    "wqi_contribution",
})
UNVERIFIED_COLUMNS = frozenset({
    "source_numeric_15",
    "source_numeric_16",
    "source_numeric_17",
    "source_numeric_18",
})


class TrainingDataError(ValueError):
    """Raised when the observations cannot support valid model evaluation."""


def require_target_column(columns, target_column=TARGET_COLUMN):
    if target_column not in columns:
        raise TrainingDataError(
            "Model training requires a numeric 'wqi' target. Existing source quality "
            "and risk categories are not WQI and cannot be used as regression targets."
        )


def train_models(records, feature_columns, target_column=TARGET_COLUMN):
    """Fit RF and XGBoost with a reproducible village-group holdout split."""
    records = list(records)
    if not records:
        raise TrainingDataError("No observations were provided for model training")
    require_target_column(records[0].keys(), target_column)

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    frame = pd.DataFrame.from_records(records)
    require_target_column(frame.columns, target_column)
    if not feature_columns:
        raise TrainingDataError("Select predictor fields explicitly before model training")

    feature_columns = list(feature_columns)
    if len(feature_columns) != len(set(feature_columns)):
        raise TrainingDataError("Predictor fields must not contain duplicates")
    missing_columns = sorted(set(feature_columns) - set(frame.columns))
    if missing_columns:
        raise TrainingDataError("Missing predictor fields: " + ", ".join(missing_columns))
    for column in set(feature_columns) & MEASUREMENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    denied_features = set(feature_columns) & (
        TARGET_DERIVED_COLUMNS | UNVERIFIED_COLUMNS | {target_column}
    )
    denied_features.update(
        name
        for name in feature_columns
        if name.startswith(("raw_field_", "parameter_text_group_", "source_numeric_"))
    )
    denied_features.update({"source_record_index", "source_record_number"} & set(feature_columns))
    denied_features.update({"quality_class_source", "risk_class_source", "wqi_class"} & set(feature_columns))
    if denied_features:
        raise TrainingDataError(
            "Potentially leaking or undocumented predictors are not allowed: "
            + ", ".join(sorted(denied_features))
        )

    if GROUP_COLUMN not in frame.columns:
        raise TrainingDataError(
            f"'{GROUP_COLUMN}' is required to hold out locations during validation"
        )
    if frame[GROUP_COLUMN].isna().any():
        raise TrainingDataError("Village/location grouping contains missing values")

    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise TrainingDataError("WQI target must be numeric and finite for every training record")
    if len(frame) < MINIMUM_RECORDS:
        raise TrainingDataError(f"At least {MINIMUM_RECORDS} observations are required")
    if frame[GROUP_COLUMN].nunique() < 2:
        raise TrainingDataError("At least two villages/locations are required for grouped validation")

    predictors = frame[feature_columns]
    numeric_columns = predictors.select_dtypes(include="number").columns.tolist()
    categorical_columns = [name for name in feature_columns if name not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ))
    if not transformers:
        raise TrainingDataError("No usable predictor columns were selected")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_indices, test_indices = next(
        splitter.split(predictors, target, groups=frame[GROUP_COLUMN])
    )
    if len(test_indices) < 2:
        raise TrainingDataError("Grouped test split must contain at least two observations")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    estimators = {
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    fitted_models = {}
    metrics = {}
    x_train = predictors.iloc[train_indices]
    x_test = predictors.iloc[test_indices]
    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]
    training_feature_ranges = {
        column: {
            "minimum": float(x_train[column].min()),
            "maximum": float(x_train[column].max()),
        }
        for column in numeric_columns
    }
    for name, estimator in estimators.items():
        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("regressor", estimator),
        ])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        fitted_models[name] = model
        metrics[name] = {
            "r2": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(rmse),
            "test_records": int(len(test_indices)),
            "validation": "grouped holdout by village/location",
        }

    best_model = min(metrics, key=lambda name: (metrics[name]["rmse"], metrics[name]["mae"]))
    return {
        "models": fitted_models,
        "metrics": metrics,
        "best_model": best_model,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "random_state": RANDOM_STATE,
        "train_records": int(len(train_indices)),
        "test_records": int(len(test_indices)),
        "validation_features": x_test.to_dict(orient="records"),
        "training_feature_ranges": training_feature_ranges,
    }