# Vellar Groundwater Atlas

A research-oriented workspace for inspecting the supplied Melbhuvanagiri and Kurinjipadi groundwater records. It exposes observed-data summaries, calculated weighted-arithmetic WQI, a capped spatial sample, temporal summaries, paginated records, Random Forest/XGBoost evaluation, predictions, and Tree-SHAP explanations.

For the detailed project overview, architecture, implementation, tools, results, and limitations, see [PROJECT_REPORT.md](./PROJECT_REPORT.md). WQI and model performance results are also discussed in [PROJECT_PERFORMANCE_REPORT.md](./PROJECT_PERFORMANCE_REPORT.md).

## Start the application

From the workspace root, install backend requirements and start the API on port 8001:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001
```

In a second terminal:

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

Open the Vite URL printed by the frontend server, normally `http://localhost:5173`. The API is available at `http://127.0.0.1:8001`; interactive API documentation is at `http://127.0.0.1:8001/docs`.

The first API request prepares `data/processed/groundwater_observations.csv` and its audit from the two source files under `dataset/`. Source files are never modified.

## Data and interpretation status

- The inputs are headerless, tab-delimited `.csv` files; the preprocessing schema is inferred from the inspected 20-field positional layout.
- Complete chemistry strings are extracted while all original source fields are retained. Truncated final rows are quarantined in the preprocessing audit.
- Existing quality and risk fields remain source labels; they are not numeric WQI measurements and are not overwritten by calculated classes.
- WQI uses six chemistry parameters and BIS IS 10500:2012 acceptable limits; alkalinity is excluded because the reporting basis is unspecified. Details and citations are in `backend/app/wqi/README.md`.
- The chemistry regression uses seven measurements in the order nitrate, fluoride, chloride, TDS, pH, iron, and alkalinity; it predicts the six-parameter formula-derived WQI, not an independently measured target. The prediction page automatically selects the lower-RMSE model on the shared village-grouped holdout, compares both models, shows Tree-SHAP contributions, guideline checks, and warns about inputs outside the training range. Example profiles are illustrative only.
- BIS guideline checks keep acceptable limits distinct from permissible limits. Alkalinity comparison is conditional on the entered result being expressed as CaCO₃. Parameter-level comparisons and model estimates are not potability certification; use laboratory testing for drinking-water decisions.
- Calculated WQI classes differ markedly from the source labels; review the method and contributions before interpretation. WQI classes are index categories, not potability certification.
- Observed coordinate values span 8.57592–8.63584 latitude and 77.36503–77.42496 longitude. Verify these against the reported Cuddalore locations before geographic interpretation. No study-area boundary is included.
- The map API returns at most 500 representative unique coordinates; the dataset API retains access to all normalized records through pagination and filters.
- Temporal views summarize observed chemistry and calculated WQI only. They do not make causal or statistical-significance claims.
- Model evaluation, reports, and global SHAP use year, season, well type, and block; these metadata-only models have negative grouped-holdout R² and are weak exploratory estimates. Chemistry prediction uses the seven ordered measurements above; numeric fields are parsed as continuous values, and both candidate models share the same village-grouped holdout. The model estimates a target calculated from these same chemistry values, so even strong holdout metrics are not independent predictive evidence or external validation. `/api/model-metrics?input_type=metadata` returns metadata-model evaluation; chemistry metrics are the default.

## Checks

```powershell
python -m unittest discover -s backend/tests
npm --prefix frontend run build
npm --prefix frontend run lint
```
