# Vellar Groundwater Atlas — Full Project Report

**Prepared:** 29 October 2026  
**Project type:** Research-oriented groundwater data exploration and analysis web application  
**Scope:** Data preparation, observed-data exploration, weighted-arithmetic Water Quality Index (WQI), exploratory model estimates, spatial and temporal summaries, and report/provenance outputs

## 1. Executive summary

Vellar Groundwater Atlas is a browser-based research workspace for the groundwater records supplied for Melbhuvanagiri and Kurinjipadi. A Python API prepares and serves the source records, calculates an explicitly configured WQI, summarizes the measurements, evaluates Random Forest and XGBoost regressors, and exposes both global and per-estimate SHAP explanations. A React application presents the results in dashboard, dataset, map, temporal analysis, prediction, and reports views.

The supplied source files are headerless tab-delimited text files with `.csv` extensions. Their schema is inferred from a 20-field layout. The pipeline produced **1,138 normalized observations**, quarantined **two malformed rows**, and removed **zero exact duplicates**. It retains original fields and an audit record for traceability.

The calculated WQI is a deterministic formula based on six measurements: nitrate, fluoride, chloride, total dissolved solids (TDS), pH, and iron. Alkalinity is excluded from that formula because the source does not establish its reporting basis. The chemistry regression uses seven measured fields—including alkalinity—to estimate this formula-derived WQI. It therefore approximates a score computed from its own inputs; its good holdout metrics are not independent evidence of real-world forecasting or water safety. Metadata-only models have negative grouped-holdout R² and are weak exploratory estimates.

The application is intended for inspection and research, not drinking-water certification. Coordinates, source quality/risk labels, inferred source schema, units, and WQI method should be independently reviewed before scientific or geographic conclusions are made.

## 2. Project goals and delivered functions

The application provides:

1. **Data normalization and provenance:** Convert the two source exports into a consistent dataset while retaining the original 20 values, tracking data issues, and quarantining malformed rows.
2. **Observed-data dashboard:** Present sample counts, observed measurement summaries, source labels, calculated WQI classifications, and map context.
3. **Dataset exploration:** Search and filter records by year, season, block, supplied quality label, and calculated WQI class; retrieve records in pages.
4. **Spatial exploration:** Display a deterministic, representative sample of valid unique coordinates, capped at 500 map points, with observation details.
5. **Temporal summaries:** Summarize observed chemistry, calculated WQI, source labels, and WQI classes by year and season. Results are descriptive, not causal.
6. **WQI calculation:** Apply the configured weighted-arithmetic method and preserve parameter-level ratings and contributions.
7. **Exploratory model estimates:** Evaluate Random Forest and XGBoost models on a shared location-grouped holdout; choose the lower-RMSE model, using MAE as a tie-breaker.
8. **Explainability and checks:** Return Tree-SHAP explanations, out-of-training-range warnings, and separate BIS parameter-level guideline checks for chemistry inputs.
9. **Report output:** Return a machine-readable JSON report from the API and provide a frontend download action.

## 3. Repository layout

| Path | Role |
|---|---|
| [`README.md`](./README.md) | Project overview, startup instructions, data interpretation caveats, and checks |
| [`PROJECT_PERFORMANCE_REPORT.md`](./PROJECT_PERFORMANCE_REPORT.md) | Focused WQI/model performance analysis and recommendations |
| [`PROJECT_REPORT.md`](./PROJECT_REPORT.md) | This full implementation and project report |
| [`dataset/Melbhuvanagiri.csv`](./dataset/Melbhuvanagiri.csv), [`dataset/Kurinjipadi.csv`](./dataset/Kurinjipadi.csv) | Original, headerless, tab-delimited input files |
| [`backend/app/main.py`](./backend/app/main.py) | FastAPI application, API routes, cached loading, prediction and guideline orchestration |
| [`backend/app/preprocessing/pipeline.py`](./backend/app/preprocessing/pipeline.py) | Source parsing, validation, normalization, audit, duplicate handling |
| [`backend/app/wqi/calculator.py`](./backend/app/wqi/calculator.py) | Validated WQI methodology and scoring implementation |
| [`backend/app/wqi/methodology.json`](./backend/app/wqi/methodology.json) | Method references, standard limits, class bands, excluded parameter note |
| [`backend/app/wqi/analysis.py`](./backend/app/wqi/analysis.py) | WQI output generation, summary statistics, and record joining |
| [`backend/app/ml/trainer.py`](./backend/app/ml/trainer.py) | Feature validation, grouped holdout, model training and metrics |
| [`backend/app/ml/explain.py`](./backend/app/ml/explain.py) | Global and local Tree-SHAP explanations |
| [`backend/app/gis/sampling.py`](./backend/app/gis/sampling.py) | Deterministic coordinate deduplication and representative map sampling |
| [`backend/app/temporal/analysis.py`](./backend/app/temporal/analysis.py) | Observed-data annual/seasonal summaries |
| [`backend/tests/`](./backend/tests/) | Unit and API tests for preprocessing-dependent behavior, WQI, models, GIS, and temporal analysis |
| [`data/processed/groundwater_observations.csv`](./data/processed/groundwater_observations.csv) | Normalized records, including raw source fields and validation notes |
| [`data/processed/preprocessing_audit.json`](./data/processed/preprocessing_audit.json) | Input counts, transformation operations, issues, quarantined rows, and field dictionary |
| [`data/processed/water_quality_index.csv`](./data/processed/water_quality_index.csv), [`data/processed/wqi_summary.json`](./data/processed/wqi_summary.json) | Per-record scores/contributions and aggregate WQI results |
| [`frontend/src/App.tsx`](./frontend/src/App.tsx) | React app pages, API integration, filters, prediction form, report download |
| [`frontend/src/SpatialMap.tsx`](./frontend/src/SpatialMap.tsx) | Leaflet map visualization |
| [`frontend/src/TemporalChart.tsx`](./frontend/src/TemporalChart.tsx) | Recharts annual trend visualization |
| [`frontend/src/App.css`](./frontend/src/App.css), [`frontend/src/index.css`](./frontend/src/index.css) | Application and global presentation styles |
| [`frontend/package.json`](./frontend/package.json) | Frontend dependencies and development/build/lint scripts |
| [`backend/requirements.txt`](./backend/requirements.txt) | Backend runtime dependency constraints |

There is no relational database in the current implementation. CSV and JSON files are the persistent data and analysis artifacts; in-process caches avoid recalculating/loading them for every request.

## 4. Data sources, schema, and preparation

### 4.1 Inputs and inferred schema

The preprocessing script expects exactly two files named `Melbhuvanagiri.csv` and `Kurinjipadi.csv`. It reads each using a tab delimiter and UTF-8 (with an optional byte-order mark). The inferred 20-position layout is:

| Source positions | Inferred or preserved meaning |
|---|---|
| 1–4 | State, district, block, village/location |
| 5–6 | Latitude and longitude |
| 7–8 | Year and season |
| 9–10 | Source record index and well type |
| 11–14 | Two parameter-text groups, each paired with a duplicate text field |
| 15–18 | Numeric source fields with undocumented meanings/units |
| 19–20 | Supplied quality and risk category labels |

Because the source has no header, these meanings are inferences, not confirmed source metadata. The pipeline records a field dictionary in the audit file and preserves every accepted source field in `raw_field_01` through `raw_field_20`.

### 4.2 Normalization and validation workflow

Implementation in [`pipeline.py`](./backend/app/preprocessing/pipeline.py):

1. Check that both expected inputs exist.
2. Read rows strictly as tab-delimited records.
3. Quarantine any row whose field count is not exactly 20.
4. Remove only exact duplicates across all 20 fields, across both inputs.
5. Normalize numeric values only when finite; normalize seasons to the project’s four canonical labels.
6. Parse parameter strings in `Name[value unit]` form using aliases for nitrate, fluoride, chloride, TDS, pH, iron, and alkalinity.
7. Check that duplicate chemistry text fields agree before extracting measurements. If they disagree, withhold the extracted values and record a validation issue.
8. Check latitude/longitude against broad WGS84 bounds and check block name against the source filename.
9. Write normalized observations and a JSON audit; do not modify source files.

The current audit records:

| Input | Rows read | Normalized rows | Malformed rows quarantined | Exact duplicates removed |
|---|---:|---:|---:|---:|
| Melbhuvanagiri | 505 | 504 | 1 | 0 |
| Kurinjipadi | 635 | 634 | 1 | 0 |
| **Total** | **1,140** | **1,138** | **2** | **0** |

The normalized measurement fields are `nitrate_mg_l`, `fluoride_mg_l`, `chloride_mg_l`, `tds_mg_l`, `ph`, `iron_mg_l`, and `alkalinity_mg_l`. Concentrations are represented in mg/L, while pH is unitless. Source fields 15–18 remain preserved but are not given interpreted names or used as model predictors.

### 4.3 Generated data products

- `groundwater_observations.csv`: normalized source records, measurements, validation issues, and original raw fields.
- `preprocessing_audit.json`: row-level quarantine/duplicate/validation details, per-input counts, operations, caveats, and field dictionary.
- `water_quality_index.csv`: record keys, calculated WQI, class, and JSON parameter contributions.
- `wqi_summary.json`: method references, parameter coverage, count, min/mean/max, and class distribution.

On the first API access, missing normalized observations are generated from `dataset/`; WQI outputs are generated if missing or older than their observation/methodology inputs. Source files are not changed.

## 5. WQI methodology and implementation

### 5.1 Configured method

The configurable methodology is stored in [`methodology.json`](./backend/app/wqi/methodology.json) and implemented in [`calculator.py`](./backend/app/wqi/calculator.py). It uses the weighted-arithmetic WQI method referenced to Ramakrishnaiah, Sadashivaiah, and Ranganna (2009), with acceptable limits from BIS IS 10500:2012, Table 2.

For a concentration parameter \(i\), the implementation uses:

```text
qi = 100 * (Ci - Vi) / (Si - Vi)
Wi = (1 / Si) / sum(1 / Sj)
WQI = sum(Wi * qi)
```

`Ci` is the observed value, `Vi` is the ideal value, `Si` is the configured standard limit, `qi` is its quality rating, and `Wi` is its normalized inverse-limit weight. For pH, the method computes deviation from the ideal pH of 7 using the appropriate lower or upper limit.

| Included parameter | Acceptable limit / range | Ideal |
|---|---:|---:|
| Nitrate | 45 mg/L | 0 |
| Fluoride | 1 mg/L | 0 |
| Chloride | 250 mg/L | 0 |
| TDS | 500 mg/L | 0 |
| pH | 6.5–8.5 | 7.0 |
| Iron | 0.3 mg/L | 0 |

Alkalinity is deliberately excluded from WQI because its source reporting basis (for example, as CaCO₃) is not confirmed. It remains a normalized measurement and is used by the chemistry model and parameter guideline panel. The methodology validator requires all six supported parameters, nonempty references, valid limits/units, and ordered continuous classification bands. The calculator rejects missing/non-finite observations and returns each rating, weight, and contribution so the total can be inspected.

Configured class bands are:

| Score | Index class |
|---:|---|
| 0–25 | Excellent |
| >25–50 | Good |
| >50–75 | Poor |
| >75–100 | Very Poor |
| >100 | Unsuitable for Drinking |

The lower boundary of a band is exclusive except for the first open-ended band; the upper boundary is inclusive. These are index classes and are not a potability certification.

### 5.2 Current WQI results

The current saved summary reports 1,138 records:

| Measure/class | Count or value |
|---|---:|
| Mean WQI | 174.71 |
| Minimum WQI | 24.34 |
| Maximum WQI | 658.35 |
| Unsuitable for Drinking | 903 (79.4%) |
| Very Poor | 111 (9.8%) |
| Poor | 78 (6.9%) |
| Good | 45 (4.0%) |
| Excellent | 1 (0.1%) |

Inverse-limit weighting puts most weight on iron (approximately 74.4%) and fluoride (approximately 22.3%) under this configuration. This materially affects the score and should be considered when interpreting, comparing, or revising the index. The calculated classes differ markedly from the supplied source labels, which are kept separate and are not treated as WQI.

## 6. Backend and API design

### 6.1 Runtime structure

The backend is a FastAPI application in [`main.py`](./backend/app/main.py). It uses Pydantic for request validation and FastAPI query constraints for pagination and filter validation. Loading, WQI generation, model training, and global SHAP results are cached in process using `functools.lru_cache`. The API’s CORS policy permits local Vite development origins on localhost/127.0.0.1 ports in the 5170–5179 range and permits GET and POST.

Errors such as unavailable processed data, WQI, model evaluation, and SHAP explanations are surfaced through API errors rather than presented as valid results. Prediction input disallows extra fields, NaN, and infinity; concentrations must be non-negative and pH must be between 0 and 14.

### 6.2 API routes

| Method and route | Purpose |
|---|---|
| `GET /api/health` | Report API/data availability and record count |
| `GET /api/summary` | Return record, location, year, source-label, WQI, and mean-parameter summaries |
| `GET /api/dataset` | Paginated normalized records with filters for year, season, block, source quality class, WQI class, and text search |
| `GET /api/spatial` | Filter and return at most 500 unique representative map points plus counts, bounds, and available filters |
| `GET /api/temporal` | Observed-only annual, seasonal, and year-season parameter and class summaries |
| `GET /api/wqi` | Paginated WQI scores and contribution breakdowns; filterable by year, season, block, and WQI class |
| `GET /api/model-metrics` | Compare RF/XGBoost evaluation; defaults to chemistry, with `input_type=metadata` for metadata-only models |
| `GET /api/shap/global` | Return a global Tree-SHAP feature ranking for the metadata model |
| `GET /api/prediction/options` | Return chemistry feature names, illustrative sample profiles, and guideline source |
| `POST /api/prediction` | Estimate WQI using exactly one complete input type: chemistry or metadata |
| `GET /api/reports` | Return report-oriented summary, temporal, preprocessing audit, metadata model metrics, global SHAP, and limitations |

Interactive OpenAPI documentation is available at `/docs` when the API is running.

### 6.3 Prediction behavior

The prediction endpoint accepts **either**:

- Metadata: year, season, well type, and block; or
- Chemistry: nitrate, fluoride, chloride, TDS, pH, iron, and alkalinity.

It rejects mixed, incomplete, or empty input sets. Chemistry prediction returns both the selected model estimate and deterministic formula-calculated WQI; the latter is the direct method for calculating this project’s WQI from those measurements. The response includes selected-model metrics, a local SHAP explanation, model-selection reason, parameter-by-parameter guideline status, training ranges, out-of-range feature warnings, and explicit limitations.

The chemistry guideline panel uses BIS IS 10500:2012 acceptable and permissible limits separately where available. Alkalinity comparison is conditional on the input being expressed as CaCO₃. Parameter guideline assessments are independent of the model estimate; neither a guideline panel nor the WQI constitutes certification.

## 7. Machine-learning implementation and results

### 7.1 Training design

[`trainer.py`](./backend/app/ml/trainer.py) requires a numeric finite `wqi` target; the supplied quality and risk labels are explicitly not valid substitutes. The default metadata feature set is year, season, well type, and block. The chemistry feature set is, in fixed order:

```text
nitrate_mg_l, fluoride_mg_l, chloride_mg_l, tds_mg_l,
ph, iron_mg_l, alkalinity_mg_l
```

Both model families use a single reproducible `GroupShuffleSplit`: 80/20 holdout grouped by village/location, random seed 42. A group must not be missing and at least two groups and ten observations are required. Numeric values are imputed with the median; categorical fields use most-frequent imputation and one-hot encoding with unknown categories ignored. Random Forest uses 400 trees and a minimum leaf size of 2. XGBoost uses 300 estimators, depth 4, learning rate 0.05, 0.9 subsampling and column sampling, and squared-error objective. The selected model minimizes held-out RMSE, with MAE as a tie-breaker.

Leakage safeguards disallow target-derived classes/contributions, raw and undocumented source fields, source record indices, and `wqi` as predictors. Numeric chemistry fields are explicitly parsed as continuous values.

### 7.2 Reported evaluation

The existing performance report records 896 training and 242 held-out observations for the model evaluation. The chemistry evaluation uses the seven chemistry predictors; metadata evaluation uses year, season, well type, and block.

| Predictor set → target | Model | R² | MAE | RMSE |
|---|---|---:|---:|---:|
| Chemistry → formula-derived WQI | XGBoost (selected) | 0.9986 | 1.90 | 2.42 |
| Chemistry → formula-derived WQI | Random Forest | 0.9898 | 2.39 | 6.67 |
| Metadata → formula-derived WQI | XGBoost | -0.7357 | 73.78 | 86.81 |
| Metadata → formula-derived WQI | Random Forest | -0.8593 | 75.47 | 89.84 |

These are internal grouped-holdout evaluations documented in [`PROJECT_PERFORMANCE_REPORT.md`](./PROJECT_PERFORMANCE_REPORT.md), not external validation. Chemistry features are the measurements used to calculate the target, so the high scores quantify formula-surrogate agreement and do not demonstrate independent predictive power. Negative R² for metadata means those models perform worse on the held-out observations than a constant mean baseline. Use deterministic WQI calculation for a score derived from measured chemistry; treat metadata predictions as weak exploratory estimates.

### 7.3 SHAP explanations

[`explain.py`](./backend/app/ml/explain.py) uses Tree SHAP on the fitted tree estimator. The preprocessing pipeline’s transformed feature names are grouped back to original input fields (including summing one-hot category contributions). Global explanations rank mean absolute contributions on the validation rows. Local explanations return signed contributions for one estimate and verify that baseline plus contributions reconciles to the model output within a tolerance. Positive local values increase estimated WQI relative to the baseline; negative values reduce it.

The API’s global SHAP/report view explains the metadata model. Chemistry predictions request a local explanation for their selected chemistry model.

## 8. Spatial and temporal analysis

### 8.1 Spatial map

[`sampling.py`](./backend/app/gis/sampling.py) removes invalid coordinates and deduplicates exact coordinate pairs. If a location repeats, the deterministic time/source ordering selects the representative record. If the unique points exceed 500, sampling first allocates class-stratified quotas, then distributes selections across coordinate grid cells and time-orders records. Metadata includes input/unique/selected counts, per-class representation, coordinate bounds, and a note that coordinates require independent verification.

The frontend uses React Leaflet and OpenStreetMap tiles. Point colors represent the calculated WQI class; tooltips show village, block, year, season, observed WQI, source label, and coordinates. The map is a visualization of observed records; its point cap is not a reduction in dataset/API availability.

The stored observations’ coordinate values span approximately **8.57592–8.63584 latitude** and **77.36503–77.42496 longitude**. The project documentation flags this extent for verification against the reported Cuddalore locations. No official study-area boundary is included.

### 8.2 Temporal analysis

[`analysis.py`](./backend/app/temporal/analysis.py) summarizes observed measurements and calculated WQI using year and season, with mean, median, count, and class distributions. The available years are 2018–2025 and the expected canonical seasons are Pre-Monsoon, SW Monsoon, NE Monsoon, and Post-Monsoon. Rows with invalid years are excluded from temporal groupings and counted. The output explicitly notes that there are no exact sampling dates, unequal counts/repeated sites can affect comparability, and no statistical significance or causal/region-wide trend is established.

The React chart uses Recharts; the user can choose the analyte, and the chart displays annual observed means. Seasonal/year-season views remain descriptive summaries.

## 9. Frontend and user workflow

The frontend is a React 19 and TypeScript single-page application built with Vite. Its six navigation views are:

- **Dashboard:** High-level record/WQI summaries and map preview.
- **Dataset:** Searchable, filterable, paginated observation table.
- **Spatial Analysis:** Interactive map and map-sampling metadata.
- **Temporal Analysis:** Annual trends and season/year summaries of observed measurements and WQI.
- **WQI Prediction:** Chemistry entry form, illustrative example profiles, model estimate, formula calculation, local model explanation, parameter guideline assessments, and domain warnings.
- **Reports:** Data and provenance overview with download of the `/api/reports` JSON payload.

The app loads data from `VITE_API_BASE` if set, otherwise `http://127.0.0.1:8001/api`. Large map and temporal components are lazy-loaded. React effects fetch data on page/filter changes and cancel obsolete requests with `AbortController`. The reports view downloads JSON; it does not generate a PDF or export the raw source files.

Frontend libraries include React, React DOM, React Leaflet/Leaflet, Recharts, and Lucide React. Development/build tools are TypeScript, Vite, the Vite React plugin, and Oxlint.

## 10. Tools, libraries, and techniques

### Backend

| Tool/library | Use |
|---|---|
| Python | API, transformations, statistical summaries, analysis orchestration |
| FastAPI | HTTP API, route/query validation, OpenAPI documentation |
| Pydantic | Typed validation of prediction requests |
| Uvicorn | ASGI development/runtime server |
| Python `csv`, `json`, `pathlib`, `re`, `statistics`, `collections`, `functools` | Source parsing, output files, validation, summary calculations, caching |
| NumPy and pandas | Numeric checks, feature frames, model/SHAP input |
| scikit-learn | Pipelines, imputation, one-hot encoding, grouped split, Random Forest and metrics |
| XGBoost | Gradient-boosted regression model |
| SHAP | Tree-based feature attribution |

Backend dependency version constraints are declared in [`requirements.txt`](./backend/requirements.txt): FastAPI `>=0.115,<1.0`, Uvicorn `>=0.30,<1.0`, NumPy `>=1.24`, pandas `>=2.0`, scikit-learn `>=1.4`, XGBoost `>=2.0`, and SHAP `>=0.46`.

### Frontend

| Tool/library | Use |
|---|---|
| React 19 + TypeScript | Typed component UI and state |
| Vite 8 | Development server and production bundling |
| React Leaflet + Leaflet | Interactive spatial map |
| Recharts | Annual parameter chart |
| Lucide React | Interface icons |
| Oxlint | Frontend linting |

Exact declared frontend package ranges are in [`package.json`](./frontend/package.json); the repository specifies dependency ranges rather than pinning a full environment in this report.

### Testing and developer commands

Backend tests use Python `unittest` and FastAPI `TestClient`. Test modules cover API responses and validation, WQI formula/configuration/class boundaries, model target requirements, deterministic map sampling, and temporal summaries. Frontend scripts provide Vite build and Oxlint.

Commands from the workspace root on Windows PowerShell:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001
```

In another terminal:

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

Reproducible checks:

```powershell
python -m unittest discover -s backend/tests
npm --prefix frontend run build
npm --prefix frontend run lint
```

Open the frontend URL printed by Vite (normally `http://localhost:5173`). The API is at `http://127.0.0.1:8001`; interactive API documentation is at `http://127.0.0.1:8001/docs`.

## 11. Representative code patterns

The pipeline intentionally rejects malformed row lengths and preserves a trace of what it did:

```python
if len(values) != EXPECTED_FIELDS:
    audit["quarantined_rows"].append({
        "source_file": file_name,
        "source_record_number": record_number,
        "field_count": len(values),
        "raw_fields": values,
        "reason": f"expected {EXPECTED_FIELDS} fields, found {len(values)}",
    })
    continue
```

The WQI implementation computes inverse-limit weights and auditable parameter contributions:

```python
inverse_limits = {
    name: 1.0 / self.parameter_limits[name].standard_limit
    for name in REQUIRED_PARAMETERS
}
inverse_limit_sum = sum(inverse_limits.values())

weight = inverse_limits[name] / inverse_limit_sum
contribution = rating * weight
score += contribution
```

The model trainer validates the numeric target and uses grouped splits by location:

```python
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(
    splitter.split(predictors, target, groups=frame["village"])
)
```

The full implementations and additional guards are in [`pipeline.py`](./backend/app/preprocessing/pipeline.py), [`calculator.py`](./backend/app/wqi/calculator.py), and [`trainer.py`](./backend/app/ml/trainer.py).

## 12. Testing and verification coverage

The repository includes tests that assert, among other behaviors:

- Real source data produces 1,138 normalized observations and finite auditable WQI scores.
- WQI parameter contributions reconcile to the total; normalized weights sum to 1; classification boundaries are inclusive/exclusive as documented.
- The source category fields alone cannot satisfy the numeric WQI target requirement.
- Map sampling is deterministic, unique-coordinate, class-representative, capped, and marked as observed.
- Temporal analysis covers 2018–2025 and the four seasonal labels.
- API summary, pagination/filtering, WQI output, model metrics/predictions, report provenance, and map coordinate caveats are exposed.

The commands above are the project’s available checks. This report documents source/test coverage; it does not claim that those commands were executed as part of authoring this document.

## 13. Limitations, interpretation, and risk controls

1. **Inferred source schema:** The input has no headers; position-based meanings need confirmation from the source provider.
2. **Undocumented fields:** Source numeric positions 15–18 have unknown definitions/units and are retained without analytical use.
3. **Chemistry provenance:** Parameter strings are parsed from duplicate source text only when corresponding parsed groups agree. Source analytical units/methods should be independently confirmed.
4. **WQI choice and weighting:** The configured inverse-limit method yields unusually high influence for iron and fluoride. Domain experts should confirm whether this method, standard edition, and parameter selection are appropriate.
5. **Source classes:** Supplied quality and risk labels have undocumented methods and are not interchangeable with calculated WQI classes.
6. **Model target dependence:** Chemistry model target is calculated from the same measurements used as model features; metrics are formula-reproduction performance, not independent prediction.
7. **Metadata model weakness:** Negative R² means current metadata predictors perform poorly on held-out locations.
8. **No external validation:** There is no independent laboratory target, external dataset, or external validation split in the reported evaluation.
9. **Geographic uncertainty:** Coordinate range needs verification against the named Cuddalore locations; there is no verified boundary/sub-basin geometry.
10. **Temporal limitations:** The data provide years and seasons, not exact sampling dates; counts and repeated locations may be unbalanced.
11. **Guidelines are not certification:** Neither aggregate WQI, model predictions, nor parameter comparisons alone certify water safety; laboratory testing and appropriate professional review are needed.
12. **No causal conclusions:** Dashboard trends and distributions are descriptive and should not be read as causal effects or statistically significant regional trends.

## 14. Recommended next steps

1. Obtain the original data dictionary, source methodology, laboratory methods, units, and coordinate reference/location metadata; confirm the inferred schema and geographic extent.
2. Confirm the WQI formula, limits, and parameter set with domain experts; examine sensitivity to the inverse-limit weighting, especially iron and fluoride.
3. Investigate why calculated WQI classes diverge from supplied quality/risk labels, without assuming either set is correct.
4. Add independent measured outcomes and suitable external spatial/temporal validation before presenting model performance as predictive evidence.
5. Keep deterministic WQI as the preferred calculation when all six required measurements are available; clearly label model estimates and examples as exploratory.
6. Consider adding verified study-area boundary data only after coordinate provenance is established.
7. Continue testing changes to data validation, endpoint contracts, and interpretation safeguards.

## 15. Related project documentation

- Project setup and concise interpretation notes: [`README.md`](./README.md)
- WQI formula and science caveats: [`backend/app/wqi/README.md`](./backend/app/wqi/README.md)
- Focused measured WQI/model results: [`PROJECT_PERFORMANCE_REPORT.md`](./PROJECT_PERFORMANCE_REPORT.md)
- Dataset transformation details: [`data/processed/preprocessing_audit.json`](./data/processed/preprocessing_audit.json)
- Current WQI summary: [`data/processed/wqi_summary.json`](./data/processed/wqi_summary.json)
