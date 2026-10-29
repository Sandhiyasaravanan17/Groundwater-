"""FastAPI endpoints for the groundwater observations currently available."""

import csv
import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .gis.sampling import sample_map_points
from .ml.explain import ExplanationError, explain_global, explain_local
from .ml.trainer import train_models
from .preprocessing.pipeline import MEASUREMENT_FIELDS, run_pipeline
from .temporal.analysis import analyze_temporal_records
from .wqi.analysis import (
    DEFAULT_METHODOLOGY,
    join_wqi_results,
    load_methodology,
    write_wqi_outputs,
)


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "groundwater_observations.csv"
AUDIT_FILE = PROCESSED_DIR / "preprocessing_audit.json"
PAGE_SIZE_MAX = 100
MAP_POINT_LIMIT = 500
MODEL_FEATURES = (
    "year",
    "season",
    "well_type",
    "block",
)
CHEMISTRY_FEATURES = (
    "nitrate_mg_l",
    "fluoride_mg_l",
    "chloride_mg_l",
    "tds_mg_l",
    "ph",
    "iron_mg_l",
    "alkalinity_mg_l",
)
PARAMETER_GUIDELINES = {
    "nitrate_mg_l": {
        "label": "Nitrate",
        "unit": "mg/L",
        "acceptable_limit": 45.0,
        "permissible_limit": None,
        "no_relaxation": True,
        "explanation": "Elevated nitrate can pose health risks, particularly for infants.",
    },
    "fluoride_mg_l": {
        "label": "Fluoride",
        "unit": "mg/L",
        "acceptable_limit": 1.0,
        "permissible_limit": 1.5,
        "explanation": "Long-term excess fluoride can cause dental and skeletal fluorosis.",
    },
    "chloride_mg_l": {
        "label": "Chloride",
        "unit": "mg/L",
        "acceptable_limit": 250.0,
        "permissible_limit": 1000.0,
        "explanation": "High chloride can affect taste and may indicate salinity or contamination.",
    },
    "tds_mg_l": {
        "label": "TDS",
        "unit": "mg/L",
        "acceptable_limit": 500.0,
        "permissible_limit": 2000.0,
        "explanation": "High dissolved solids can affect taste and indicate elevated dissolved minerals.",
    },
    "ph": {
        "label": "pH",
        "unit": "unitless",
        "acceptable_range": [6.5, 8.5],
        "permissible_range": None,
        "no_relaxation": True,
        "explanation": "pH outside the range can affect corrosion, scaling, and treatment.",
    },
    "iron_mg_l": {
        "label": "Iron",
        "unit": "mg/L",
        "acceptable_limit": 0.3,
        "permissible_limit": None,
        "no_relaxation": True,
        "explanation": "Elevated iron can cause staining, deposits, and taste problems.",
    },
    "alkalinity_mg_l": {
        "label": "Alkalinity",
        "unit": "mg/L",
        "acceptable_limit": 200.0,
        "permissible_limit": 600.0,
        "basis": "as CaCO3",
        "explanation": "Alkalinity affects buffering and treatment; comparison assumes the result is reported as CaCO3.",
    },
}
ILLUSTRATIVE_SAMPLES = [
    {
        "id": "good-quality",
        "label": "Good-quality example",
        "values": {
            "nitrate_mg_l": 8.0, "fluoride_mg_l": 0.5, "chloride_mg_l": 60.0,
            "tds_mg_l": 180.0, "ph": 7.2, "iron_mg_l": 0.1,
            "alkalinity_mg_l": 140.0,
        },
    },
    {
        "id": "moderate-quality",
        "label": "Moderate-quality example",
        "values": {
            "nitrate_mg_l": 20.0, "fluoride_mg_l": 0.6, "chloride_mg_l": 150.0,
            "tds_mg_l": 300.0, "ph": 7.4, "iron_mg_l": 0.14,
            "alkalinity_mg_l": 200.0,
        },
    },
    {
        "id": "poor-quality",
        "label": "Poor-quality example",
        "values": {
            "nitrate_mg_l": 40.0, "fluoride_mg_l": 1.2, "chloride_mg_l": 240.0,
            "tds_mg_l": 490.0, "ph": 8.4, "iron_mg_l": 0.15,
            "alkalinity_mg_l": 350.0,
        },
    },
    {
        "id": "high-tds",
        "label": "High-TDS example",
        "values": {
            "nitrate_mg_l": 15.0, "fluoride_mg_l": 0.5, "chloride_mg_l": 400.0,
            "tds_mg_l": 650.0, "ph": 7.2, "iron_mg_l": 0.2,
            "alkalinity_mg_l": 300.0,
        },
    },
    {
        "id": "high-nitrate",
        "label": "High-nitrate example",
        "values": {
            "nitrate_mg_l": 50.0, "fluoride_mg_l": 0.6, "chloride_mg_l": 80.0,
            "tds_mg_l": 220.0, "ph": 7.1, "iron_mg_l": 0.15,
            "alkalinity_mg_l": 160.0,
        },
    },
    {
        "id": "high-fluoride",
        "label": "High-fluoride example",
        "values": {
            "nitrate_mg_l": 10.0, "fluoride_mg_l": 1.6, "chloride_mg_l": 90.0,
            "tds_mg_l": 240.0, "ph": 7.0, "iron_mg_l": 0.15,
            "alkalinity_mg_l": 180.0,
        },
    },
]
WQI_FILE = PROCESSED_DIR / "water_quality_index.csv"
WQI_SUMMARY_FILE = PROCESSED_DIR / "wqi_summary.json"


class PredictionRequest(BaseModel):
    year: int | None = Field(None, ge=1900, le=2200)
    season: str | None = None
    well_type: str | None = None
    block: str | None = None
    nitrate_mg_l: float | None = Field(None, ge=0)
    fluoride_mg_l: float | None = Field(None, ge=0)
    chloride_mg_l: float | None = Field(None, ge=0)
    tds_mg_l: float | None = Field(None, ge=0)
    ph: float | None = Field(None, ge=0, le=14)
    iron_mg_l: float | None = Field(None, ge=0)
    alkalinity_mg_l: float | None = Field(None, ge=0)
    class Config:
        extra = "forbid"
        allow_inf_nan = False


app = FastAPI(
    title="Groundwater Quality Research API",
    version="0.1.0",
    description="Observed groundwater summaries, calculated WQI analysis, and exploratory model estimates.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):517[0-9]+",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_records():
    if not PROCESSED_FILE.exists():
        try:
            run_pipeline(ROOT / "dataset", PROCESSED_DIR)
        except Exception as exc:
            raise RuntimeError("Groundwater data could not be prepared") from exc
    try:
        with PROCESSED_FILE.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise RuntimeError("Processed groundwater data could not be read") from exc


@lru_cache(maxsize=1)
def load_wqi_records():
    observations = load_records()
    output_files = (WQI_FILE, WQI_SUMMARY_FILE)
    source_files = (PROCESSED_FILE, DEFAULT_METHODOLOGY)
    outputs_are_stale = any(
        not output.exists()
        or any(source.stat().st_mtime_ns > output.stat().st_mtime_ns for source in source_files)
        for output in output_files
    )
    if outputs_are_stale:
        try:
            write_wqi_outputs(PROCESSED_FILE, PROCESSED_DIR)
        except Exception as exc:
            raise RuntimeError("WQI could not be calculated from the observations") from exc
    try:
        with WQI_FILE.open("r", encoding="utf-8", newline="") as stream:
            wqi_results = list(csv.DictReader(stream))
        for result in wqi_results:
            result["wqi"] = float(result["wqi"])
            result["wqi_contributions"] = json.loads(result["wqi_contributions"])
        return join_wqi_results(observations, wqi_results)
    except (OSError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("Calculated WQI results could not be read") from exc


@lru_cache(maxsize=1)
def get_wqi_summary():
    load_wqi_records()
    try:
        return json.loads(WQI_SUMMARY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("WQI summary could not be read") from exc


@lru_cache(maxsize=1)
def get_trained_models():
    try:
        return train_models(load_wqi_records(), MODEL_FEATURES)
    except Exception as exc:
        raise RuntimeError("WQI prediction models could not be trained") from exc


@lru_cache(maxsize=1)
def get_chemistry_models():
    try:
        return train_models(load_wqi_records(), CHEMISTRY_FEATURES)
    except Exception as exc:
        raise RuntimeError("Chemistry-based WQI models could not be trained") from exc


@lru_cache(maxsize=2)
def get_global_explanation(model_name):
    try:
        return explain_global(get_trained_models(), model_name)
    except Exception as exc:
        raise RuntimeError("SHAP global explanation is unavailable") from exc


def model_selection_details(trained):
    selected_model = trained["best_model"]
    alternative_model = next(
        model_name for model_name in trained["metrics"] if model_name != selected_model
    )
    selected = trained["metrics"][selected_model]
    alternative = trained["metrics"][alternative_model]
    return {
        "selected_model": selected_model,
        "alternative_model": alternative_model,
        "criteria": "Lowest grouped-holdout RMSE; MAE is the tie-breaker. Both models use the same village/location holdout split.",
        "explanation": (
            f"{selected_model} was selected because its held-out RMSE "
            f"({selected['rmse']:.2f}) is lower than {alternative_model} "
            f"({alternative['rmse']:.2f}); its MAE is "
            f"{selected['mae']:.2f} compared with {alternative['mae']:.2f}."
        ),
        "reliable": selected["r2"] >= 0,
        "all_models_negative_r2": all(
            metric["r2"] < 0 for metric in trained["metrics"].values()
        ),
    }


def assess_chemistry(measurements):
    assessments = []
    for field in CHEMISTRY_FEATURES:
        value = float(measurements[field])
        guideline = PARAMETER_GUIDELINES[field]
        assessment = {
            "field": field,
            "label": guideline["label"],
            "value": value,
            "unit": guideline["unit"],
            "explanation": guideline["explanation"],
            "basis": guideline.get("basis"),
        }
        if field == "ph":
            lower, upper = guideline["acceptable_range"]
            assessment["reference"] = f"{lower:g}–{upper:g} (no relaxation)"
            assessment["status"] = (
                "Below guideline" if value < lower else
                "Above guideline" if value > upper else
                "Within guideline"
            )
            assessment["tone"] = "alert" if value < lower or value > upper else "good"
        elif field == "alkalinity_mg_l":
            acceptable = guideline["acceptable_limit"]
            permissible = guideline["permissible_limit"]
            assessment["reference"] = (
                f"{acceptable:g} mg/L acceptable; {permissible:g} mg/L permissible as CaCO3"
            )
            assessment["status"] = (
                "Within guideline" if value <= acceptable else
                "Above acceptable; within permissible" if value <= permissible else
                "Above permissible"
            )
            assessment["tone"] = "good" if value <= acceptable else (
                "caution" if value <= permissible else "alert"
            )
            assessment["requires_basis_review"] = True
        else:
            acceptable = guideline["acceptable_limit"]
            permissible = guideline["permissible_limit"]
            assessment["reference"] = (
                f"{acceptable:g} mg/L acceptable (no relaxation)"
                if permissible is None else
                f"{acceptable:g} mg/L acceptable; {permissible:g} mg/L permissible"
            )
            assessment["status"] = (
                "Within guideline" if value <= acceptable else
                "Above acceptable; within permissible"
                if permissible is not None and value <= permissible else
                "Above guideline"
            )
            assessment["tone"] = "good" if value <= acceptable else (
                "caution"
                if permissible is not None and value <= permissible
                else "alert"
            )
        assessments.append(assessment)
    return assessments


def get_audit():
    import json

    try:
        return json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Preprocessing audit could not be read") from exc


def filtered_records(year=None, season=None, block=None, quality_class=None, search=None, wqi_class=None):
    records = load_wqi_records()
    search_value = search.casefold().strip() if search else None
    result = []
    for record in records:
        if year is not None and record.get("year") != str(year):
            continue
        if season and record.get("season") != season:
            continue
        if block and record.get("block") != block:
            continue
        if quality_class and record.get("quality_class_source") != quality_class:
            continue
        if wqi_class and record.get("wqi_class") != wqi_class:
            continue
        if search_value and not any(
            search_value in (record.get(field) or "").casefold()
            for field in ("village", "block", "source_dataset")
        ):
            continue
        result.append(record)
    return result


def parse_parameter_means(records):
    result = {}
    for field in MEASUREMENT_FIELDS:
        values = []
        for record in records:
            try:
                value = float(record[field])
            except (TypeError, ValueError):
                continue
            if value == value and abs(value) != float("inf"):
                values.append(value)
        result[field] = {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
        }
    return result


@app.get("/api/health")
def health():
    try:
        records = load_records()
        return {"status": "ok", "records_available": len(records)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Groundwater data is currently unavailable") from exc


@app.get("/api/summary")
def summary():
    try:
        records = load_records()
        audit = get_audit()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Unable to load groundwater data") from exc

    source_classes = {}
    calculated_classes = {}
    for record in load_wqi_records():
        label = record.get("quality_class_source") or "Unlabeled"
        source_classes[label] = source_classes.get(label, 0) + 1
        calculated = record.get("wqi_class") or "Unclassified"
        calculated_classes[calculated] = calculated_classes.get(calculated, 0) + 1
    wqi = get_wqi_summary()
    return {
        "record_count": len(records),
        "village_count": len({record.get("village") for record in records if record.get("village")}),
        "block_count": len({record.get("block") for record in records if record.get("block")}),
        "year_count": len({record.get("year") for record in records if record.get("year")}),
        "year_range": [
            min(int(record["year"]) for record in records if record.get("year")),
            max(int(record["year"]) for record in records if record.get("year")),
        ] if records else None,
        "source_quality_class_distribution": source_classes,
        "calculated_wqi_class_distribution": calculated_classes,
        "wqi_summary": wqi,
        "mean_observed_parameters": parse_parameter_means(records),
        "wqi": {"status": "calculated", "method": wqi["method"], "reason": wqi["interpretation_note"]},
        "models": {"status": "available", "reason": "Regression models use temporal, well-type, and block predictors; chemistry inputs used to calculate WQI are excluded to avoid target leakage."},
        "preprocessing": audit.get("summary", {}),
    }


@app.get("/api/dataset")
def dataset_page(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=PAGE_SIZE_MAX),
    year: int | None = Query(None, ge=1900, le=2200),
    season: str | None = None,
    block: str | None = None,
    quality_class: str | None = None,
    wqi_class: str | None = None,
    search: str | None = Query(None, max_length=100),
):
    try:
        records = filtered_records(year, season, block, quality_class, search, wqi_class)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Unable to load groundwater data") from exc

    selected_fields = (
        "source_dataset", "source_record_number", "village", "block", "latitude", "longitude",
        "year", "season", "well_type", *MEASUREMENT_FIELDS, "quality_class_source", "risk_class_source",
        "wqi", "wqi_class", "wqi_contributions",
    )
    start = (page - 1) * page_size
    page_records = []
    for record in records[start:start + page_size]:
        normalized = {}
        for field in selected_fields:
            value = record.get(field)
            if value in (None, ""):
                normalized[field] = None
            elif field in {"year", "source_record_number"}:
                normalized[field] = int(value)
            elif field == "wqi_contributions":
                normalized[field] = json.loads(value) if isinstance(value, str) else value
            elif field in {"latitude", "longitude", *MEASUREMENT_FIELDS}:
                normalized[field] = float(value)
            else:
                normalized[field] = value
        page_records.append(normalized)
    return {
        "page": page,
        "page_size": page_size,
        "total_records": len(records),
        "records": page_records,
        "filters": {
            "years": sorted({int(record["year"]) for record in load_records() if record.get("year")}),
            "seasons": sorted({record["season"] for record in load_records() if record.get("season")}),
            "blocks": sorted({record["block"] for record in load_records() if record.get("block")}),
            "source_quality_classes": sorted({record["quality_class_source"] for record in load_records() if record.get("quality_class_source")}),
            "wqi_classes": sorted({record["wqi_class"] for record in load_wqi_records() if record.get("wqi_class")}),
        },
    }


@app.get("/api/spatial")
def spatial(
    year: int | None = Query(None, ge=1900, le=2200),
    season: str | None = None,
    block: str | None = None,
    quality_class: str | None = None,
    wqi_class: str | None = None,
    search: str | None = Query(None, max_length=100),
):
    try:
        records = filtered_records(year, season, block, quality_class, search, wqi_class)
        all_records = load_records()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Unable to load groundwater data") from exc
    result = sample_map_points(records, limit=MAP_POINT_LIMIT, class_field="wqi_class")
    for point in result["points"]:
        point["wqi"] = next(
            record["wqi"] for record in records
            if record["source_dataset"] == point["source_dataset"]
            and record["source_record_number"] == point["source_record_number"]
        )
        point["wqi_class"] = next(
            record["wqi_class"] for record in records
            if record["source_dataset"] == point["source_dataset"]
            and record["source_record_number"] == point["source_record_number"]
        )
    result["metadata"]["available_filters"] = {
        "years": sorted({int(record["year"]) for record in all_records if record.get("year")}),
        "seasons": sorted({record["season"] for record in all_records if record.get("season")}),
        "blocks": sorted({record["block"] for record in all_records if record.get("block")}),
        "source_quality_classes": sorted({record["quality_class_source"] for record in all_records if record.get("quality_class_source")}),
        "wqi_classes": sorted({record["wqi_class"] for record in load_wqi_records() if record.get("wqi_class")}),
    }
    return result


@app.get("/api/temporal")
def temporal():
    try:
        return analyze_temporal_records(load_wqi_records())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Unable to load groundwater data") from exc


@app.get("/api/model-metrics")
def model_metrics(input_type: str = Query("chemistry", pattern="^(chemistry|metadata)$")):
    try:
        trained = (
            get_chemistry_models()
            if input_type == "chemistry"
            else get_trained_models()
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Model evaluation is unavailable") from exc
    return {
        "input_type": input_type,
        "metrics": trained["metrics"],
        "best_model": trained["best_model"],
        "model_selection": model_selection_details(trained),
        "feature_columns": trained["feature_columns"],
        "training_feature_ranges": trained["training_feature_ranges"],
        "target_column": trained["target_column"],
        "train_records": trained["train_records"],
        "test_records": trained["test_records"],
        "random_state": trained["random_state"],
        "model_caveat": (
            "Validation holds out villages. Negative R2 indicates these predictors "
            "do not explain the held-out WQI variation well; prediction quality is limited."
            if input_type == "metadata"
            else "Chemistry predictors include the measurements used to calculate WQI, "
            "so model fit is not independent evidence of predictive performance."
        ),
    }


@app.get("/api/shap/global")
def shap_global(model: str | None = None):
    try:
        return get_global_explanation(model or get_trained_models()["best_model"])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Global model explanation is unavailable") from exc


@app.get("/api/wqi")
def wqi_page(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=PAGE_SIZE_MAX),
    wqi_class: str | None = None,
    year: int | None = Query(None, ge=1900, le=2200),
    season: str | None = None,
    block: str | None = None,
):
    try:
        records = filtered_records(year, season, block, wqi_class=wqi_class)
        summary_data = get_wqi_summary()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Calculated WQI results are unavailable") from exc
    start = (page - 1) * page_size
    fields = (
        "source_dataset", "source_record_number", "village", "block", "year", "season",
        "wqi", "wqi_class", "wqi_contributions", "quality_class_source",
    )
    page_records = []
    for record in records[start:start + page_size]:
        item = {field: record.get(field) for field in fields}
        item["source_record_number"] = int(item["source_record_number"])
        item["year"] = int(item["year"])
        item["wqi"] = float(item["wqi"])
        page_records.append(item)
    return {
        "page": page,
        "page_size": page_size,
        "total_records": len(records),
        "summary": summary_data,
        "records": page_records,
    }


@app.get("/api/prediction/options")
def prediction_options():
    return {
        "feature_fields": list(CHEMISTRY_FEATURES),
        "metadata_feature_fields": list(MODEL_FEATURES),
        "samples_are_illustrative": True,
        "sample_profiles": ILLUSTRATIVE_SAMPLES,
        "guideline_source": (
            "BIS IS 10500:2012, Table 2 (acceptable and permissible limits); "
            "alkalinity comparison assumes results are expressed as CaCO3."
        ),
    }


@app.post("/api/prediction")
def prediction(request: PredictionRequest):
    metadata_values = [getattr(request, field) for field in MODEL_FEATURES]
    chemistry_values = [getattr(request, field) for field in CHEMISTRY_FEATURES]
    has_metadata = any(value is not None for value in metadata_values)
    has_chemistry = any(value is not None for value in chemistry_values)

    if has_metadata == has_chemistry:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one complete input set: metadata or chemistry",
        )
    if has_metadata:
        if any(value is None for value in metadata_values) or any(
            not value.strip()
            for value in metadata_values
            if isinstance(value, str)
        ):
            raise HTTPException(
                status_code=422,
                detail="Metadata predictions require year, season, well_type, and block",
            )
        input_type = "metadata"
    else:
        if any(value is None for value in chemistry_values):
            raise HTTPException(
                status_code=422,
                detail="Chemistry predictions require all seven measurement fields",
            )
        input_type = "chemistry"

    try:
        trained = (
            get_trained_models()
            if input_type == "metadata"
            else get_chemistry_models()
        )
        methodology, _ = load_methodology()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="WQI prediction is currently unavailable") from exc

    model_name = trained["best_model"]

    if input_type == "metadata":
        features = {
            field: str(getattr(request, field))
            for field in trained["feature_columns"]
        }
    else:
        features = {
            field: float(getattr(request, field))
            for field in trained["feature_columns"]
        }
    import pandas as pd

    predicted_wqi = float(trained["models"][model_name].predict(pd.DataFrame([features]))[0])
    classification = methodology.classify(predicted_wqi)
    if input_type == "chemistry":
        calculated = methodology.calculate(features)
        parameter_assessments = assess_chemistry(features)
        training_feature_ranges = trained["training_feature_ranges"]
        out_of_domain = [
            {
                "field": field,
                "value": value,
                **training_feature_ranges[field],
            }
            for field, value in features.items()
            if field in training_feature_ranges
            and not (
                training_feature_ranges[field]["minimum"]
                <= value
                <= training_feature_ranges[field]["maximum"]
            )
        ]
    else:
        calculated = None
        parameter_assessments = []
        out_of_domain = []
    selection = model_selection_details(trained)
    try:
        local_explanation = explain_local(trained, features, model_name)
    except ExplanationError as exc:
        raise HTTPException(status_code=503, detail="Local model explanation is unavailable") from exc
    return {
        "prediction_type": "predicted",
        "input_type": input_type,
        "predicted_wqi": predicted_wqi,
        "predicted_class": classification,
        "model": model_name,
        "model_selection": selection,
        "model_performance": trained["metrics"][model_name],
        "local_explanation": local_explanation,
        "input_features": features,
        "calculated_wqi": calculated["wqi"] if calculated else None,
        "calculated_class": calculated["classification"] if calculated else None,
        "parameter_assessments": parameter_assessments,
        "training_domain": trained.get("training_feature_ranges", {}),
        "out_of_domain_features": out_of_domain,
        "guideline_source": (
            "BIS IS 10500:2012, Table 2; alkalinity reference assumes the input is reported as CaCO3."
        ),
        "performance_caveat": (
            (
                "The selected model has negative grouped-holdout R2 and may be less accurate than predicting the test-set mean. "
                if trained["metrics"][model_name]["r2"] < 0
                else "Grouped-holdout performance is not external validation. "
            )
            + "The model estimates a formula-derived WQI from the same chemistry measurements; it is not independent evidence of water quality or potability."
            if input_type == "chemistry"
            else "This model estimate is exploratory, not a measurement or potability result."
        ),
    }


@app.get("/api/reports")
def reports():
    try:
        trained = get_trained_models()
        global_shap = get_global_explanation(trained["best_model"])
        return {
            "summary": summary(),
            "temporal": temporal(),
            "preprocessing_audit": get_audit(),
            "model_metrics": {
                "metrics": trained["metrics"],
                "best_model": trained["best_model"],
                "feature_columns": trained["feature_columns"],
                "train_records": trained["train_records"],
                "test_records": trained["test_records"],
            },
            "global_shap": global_shap,
            "limitations": [
                "The source files are headerless and were mapped using an inferred positional schema.",
                "Observed coordinates span 8.57592–8.63584 latitude and 77.36503–77.42496 longitude; verify this extent against the reported Cuddalore blocks. No official study-area boundary is available.",
                "Source quality labels are undocumented and are not calculated WQI classes.",
                "WQI is calculated using the configured weighted-arithmetic method and BIS acceptable limits; source labels remain separate.",
                "Model validation currently has negative R2 with available non-chemistry predictors; predictions are exploratory and weakly predictive.",
            ],
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Report analysis is currently unavailable") from exc