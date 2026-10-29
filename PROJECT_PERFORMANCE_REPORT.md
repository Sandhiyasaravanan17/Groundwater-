# Vellar Groundwater Atlas — Project Performance Report

**Prepared:** 22 October 2026  
**Scope:** WQI calculation, chemistry-based WQI estimation, model evaluation, and data limitations

## Executive summary

The project standardizes two groundwater datasets, calculates an auditable weighted-arithmetic Water Quality Index (WQI), and provides exploratory Random Forest and XGBoost estimates. There are 1,138 normalized observations with calculated WQI scores.

The most important distinction is that the WQI is **calculated directly from six chemistry measurements**. The chemistry ML models are trained to reproduce that formula-derived score, not to predict independently measured water quality or future groundwater conditions. Their grouped-holdout scores are excellent as formula surrogates, but are not independent evidence of real-world forecasting performance. In contrast, metadata-only models have negative R² and perform poorly on held-out locations.

## 1. Project and data overview

- Inputs: `Melbhuvanagiri.csv` and `Kurinjipadi.csv`; the source exports are headerless, tab-delimited files despite their `.csv` extensions.
- The preprocessing pipeline retained 1,138 records, quarantined two malformed rows, and found no exact duplicates.
- Chemistry is extracted only when repeated source chemistry fields agree. Original source fields are retained for traceability.
- The WQI uses nitrate, fluoride, chloride, total dissolved solids (TDS), pH, and iron. Alkalinity is excluded from the index because its reporting basis is not confirmed, but it is included as an ML input.
- The available observations span 2018–2025. Coordinates have not been independently verified against the reported study area.

## 2. WQI calculation formula

The project implements the weighted-arithmetic method documented in [`backend/app/wqi/README.md`](./backend/app/wqi/README.md), using acceptable limits from BIS IS 10500:2012, Table 2.

For each one-sided concentration parameter \(i\):

\[
q_i = 100 \times \frac{C_i - V_i}{S_i - V_i}
\]

\[
W_i = \frac{1/S_i}{\sum_{j=1}^{n}(1/S_j)}
\qquad
\mathrm{WQI} = \sum_{i=1}^{n} W_i q_i
\]

Where:

- \(C_i\) is the observed measurement.
- \(S_i\) is the acceptable standard limit.
- \(V_i\) is the ideal value (zero for the five concentration parameters).
- \(q_i\) is the parameter quality rating.
- \(W_i\) is the normalized parameter weight; all weights sum to 1.

For pH, the ideal is 7.0, the acceptable range is 6.5–8.5, and rating is calculated against the relevant side of the range:

\[
q_{\mathrm{pH}} =
\begin{cases}
100 \times (7.0-C_{\mathrm{pH}})/(7.0-6.5), & C_{\mathrm{pH}} < 7.0 \\
100 \times (C_{\mathrm{pH}}-7.0)/(8.5-7.0), & C_{\mathrm{pH}} \ge 7.0
\end{cases}
\]

### Limits and resulting normalized weights

| Parameter | Acceptable limit \(S_i\) | Normalized weight \(W_i\) |
|---|---:|---:|
| Nitrate | 45 mg/L | 0.004961 (0.496%) |
| Fluoride | 1 mg/L | 0.223254 (22.325%) |
| Chloride | 250 mg/L | 0.000893 (0.089%) |
| TDS | 500 mg/L | 0.000447 (0.045%) |
| pH | 8.5 upper limit; 6.5 lower limit | 0.026265 (2.627%) |
| Iron | 0.3 mg/L | 0.744180 (74.418%) |

**Interpretation note:** Because the method assigns weights inversely proportional to acceptable limits, iron receives about 74% of the total weight and fluoride about 22%. Thus, the index is strongly influenced by these two parameters. That follows from the configured formula and limits; it should be considered when interpreting scores or comparing this index with other WQI methods.

### Classification bands

| WQI score | Project class |
|---:|---|
| 0–25 | Excellent |
| >25–50 | Good |
| >50–75 | Poor |
| >75–100 | Very Poor |
| >100 | Unsuitable for Drinking |

These are the configured index classes, not a potability certification.

### Worked example

For the illustrative chemistry profile nitrate 8, fluoride 0.5, chloride 60, TDS 180, pH 7.2, and iron 0.1 (concentrations in mg/L):

- Formula-calculated WQI: **36.44**
- Resulting class: **Good**
- Iron contributes about **24.81** WQI points, while fluoride contributes about **11.16** points.

Alkalinity does not enter this calculation.

## 3. Observed WQI results

The stored project summary reports 1,138 calculated scores:

| Measure / class | Result |
|---|---:|
| Mean WQI | 174.71 |
| Minimum WQI | 24.34 |
| Maximum WQI | 658.35 |
| Unsuitable for Drinking | 903 (79.4%) |
| Very Poor | 111 (9.8%) |
| Poor | 78 (6.9%) |
| Good | 45 (4.0%) |
| Excellent | 1 (0.1%) |

Most scores fall in the highest WQI band. This is a result of the configured index and input data, not a standalone determination that samples are safe or unsafe. The project documentation also notes that the calculated classes differ markedly from the supplied source quality/risk labels; those labels have not been treated as numeric WQI or overwritten.

## 4. Prediction model performance

Both Random Forest and XGBoost were evaluated on the same reproducible 80/20 village-grouped holdout (896 training records; 242 held-out records; random seed 42). Model selection uses the lowest RMSE, with MAE as a tie-breaker.

| Input / target | Model | R² | MAE (WQI points) | RMSE (WQI points) |
|---|---|---:|---:|---:|
| Chemistry → formula-derived WQI | XGBoost (selected) | 0.9986 | 1.90 | 2.42 |
| Chemistry → formula-derived WQI | Random Forest | 0.9898 | 2.39 | 6.67 |
| Year, season, well type, block → formula-derived WQI | XGBoost | -0.7357 | 73.78 | 86.81 |
| Year, season, well type, block → formula-derived WQI | Random Forest | -0.8593 | 75.47 | 89.84 |

### What these metrics mean

- **Chemistry model:** The high R² and low errors show that the selected model closely approximates the WQI calculation on the held-out records. However, the target is calculated from the same chemistry measurements supplied as model features. This is a formula-reproduction task, not independent prediction, and the ML estimate should not replace the deterministic WQI formula.
- **Metadata model:** Negative R² means both models perform worse on held-out villages than a constant prediction equal to the test-set mean. These estimates are weak and exploratory, and should not be presented as reliable site-level prediction.
- These results are internal grouped-holdout evaluation only; there is no independent laboratory target or external dataset validation.

The application also reports per-prediction model performance, SHAP contributions, parameter guideline checks, and warnings for chemistry inputs outside training ranges.

## 5. Performance conclusion and recommendations

**Current status:** The data preparation and formula-based WQI pipeline provide traceable, inspectable calculations. Chemistry-based regression has strong agreement with the calculated target, while metadata-only prediction has poor generalization.

Recommended next steps:

1. Prefer the deterministic weighted-arithmetic formula for a WQI calculated from measured chemistry. Keep the ML output explicitly labeled as an estimate or formula surrogate.
2. Confirm the WQI method and parameter weights with the study’s domain experts, particularly the high iron and fluoride weights. Consider a sensitivity analysis before drawing scientific conclusions.
3. Validate units, analytical methods, source labels, and geographic coordinates against original records or laboratory documentation.
4. Evaluate genuine predictive performance only against independently measured WQI/quality targets, with a temporal or spatial external holdout where possible.
5. Do not use WQI class alone for drinking-water decisions; assess individual parameters and consult appropriate laboratory testing.

## 6. Reproducibility and project references

- WQI formula, sources, and interpretation: [`backend/app/wqi/README.md`](./backend/app/wqi/README.md)
- Configured limits and classification bands: [`backend/app/wqi/methodology.json`](./backend/app/wqi/methodology.json)
- WQI result summary: [`data/processed/wqi_summary.json`](./data/processed/wqi_summary.json)
- Preprocessing audit: [`data/processed/preprocessing_audit.json`](./data/processed/preprocessing_audit.json)
- Model evaluation and prediction implementation: [`backend/app/ml/trainer.py`](./backend/app/ml/trainer.py)
- Project overview and caveats: [`README.md`](./README.md)

To reproduce the project checks, run `python -m unittest discover -s backend/tests` from the project root. The evaluated estimators and metrics are produced by the `/api/model-metrics` endpoint; chemistry metrics are its default and metadata metrics are available with `input_type=metadata`.
