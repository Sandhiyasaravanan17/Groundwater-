# WQI Methodology

The dataset does not contain numeric WQI values or its original calculation
reference. The implementation therefore calculates a new, explicit index and
retains the supplied quality/risk labels independently; it does not infer the
formula from those labels.

`methodology.json` configures a **weighted-arithmetic WQI** using the acceptable
limits in BIS IS 10500:2012, Table 2. The method reference is Ramakrishnaiah,
Sadashivaiah, and Ranganna (2009), *Assessment of Water Quality Index for the
Groundwater in Tumkur Taluk, Karnataka State, India*, *Journal of Chemistry*,
6(2), 523-530, DOI: [10.1155/2009/757424](https://doi.org/10.1155/2009/757424).

For one-sided concentration parameters, the quality rating is:

```text
qi = 100 * (Ci - Vi) / (Si - Vi)
Wi = (1 / Si) / sum(1 / Sj)
WQI = sum(Wi * qi)
```

Here, `Ci` is the observed concentration, `Si` is the acceptable limit, and
`Vi` is the ideal value (zero for concentrations). The six included values are
nitrate 45 mg/L, fluoride 1 mg/L, chloride 250 mg/L, TDS 500 mg/L, iron
0.3 mg/L, and pH 6.5-8.5 (ideal pH 7). For pH below 7, the lower limit is used
to rate deviation; above 7, the upper limit is used. Weights use the inverse
acceptable limit and are normalized to sum to one.

The class bands follow the cited weighted-arithmetic groundwater WQI reference:
0-25 Excellent, >25-50 Good, >50-75 Poor, >75-100 Very Poor, and >100
Unsuitable for Drinking. Bounds are continuous: each upper bound is inclusive,
and the next class begins strictly above it.

To configure it, provide a mapping with:

- `method_reference`: bibliographic or authoritative source for the formula.
- `standard_source`: standard, edition, and table used for each parameter limit.
- `classification_source`: source for the full WQI class scale.
- `parameters`: exactly the six configured index parameters, each with a
  `standard_limit`, `ideal_value`, and unit; pH also has `lower_limit`.
- `classification_bands`: ordered contiguous bands covering the full score axis;
  use `null` for the open lower bound of the first band and open upper bound of
  the last band.

The calculator rejects incomplete configuration or observations. It returns the
observed values, limits, normalized weights, individual quality ratings, and
per-parameter WQI contributions so results remain inspectable. Alkalinity is
excluded because the source does not specify its reporting basis (for example,
as CaCO3); it is retained as an observed measurement. This index is not a
potability certification and does not replace parameter-by-parameter
comparison with the cited standard. WHO drinking-water guidance is health-based
guidance, not a WQI class scheme; see [WHO GDWQ, 4th edition, first addendum](https://www.who.int/publications/i/item/9789241549950).

## Chemistry prediction and guideline checks

The regression dashboard accepts seven continuous chemistry features in this
fixed training order: nitrate, fluoride, chloride, TDS, pH, iron, and
alkalinity. The target remains the WQI calculated from the six parameters above;
alkalinity is a model input but is not part of that formula. The two regressors
are compared on the same village-grouped holdout and selection minimizes RMSE,
using MAE as a tie-breaker. Continuous chemistry values are parsed numerically
before preprocessing. Training ranges are calculated from the training split
and inputs outside those ranges are flagged as extrapolations.

The interface separately compares entered values with BIS IS 10500:2012,
Table 2 acceptable and permissible limits where both are specified: nitrate
45 mg/L (no relaxation); fluoride 1.0/1.5 mg/L; chloride 250/1,000 mg/L;
TDS 500/2,000 mg/L; pH 6.5–8.5 (no relaxation); iron 0.3 mg/L (no relaxation);
and alkalinity 200/600 mg/L as CaCO3. Alkalinity comparison is conditional on
the reporting basis being confirmed. Exceeding an acceptable limit but
remaining within a stated permissible limit is shown separately from exceeding
the permissible limit. These checks are not inferred from the model estimate
and do not certify potability.

The chemistry regression learns a target calculated from the same entered
measurements, so its validation is not independent evidence of water quality.
Sample profiles are illustrative, not records from a named well or village.
Always confirm drinking-water quality through appropriate laboratory testing.

## Current Status

WQI scores are calculated for all 1,138 complete observations into
`data/processed/water_quality_index.csv`; summary and method details are in
`data/processed/wqi_summary.json`. The source labels remain unverified and are
not overwritten. On this configuration, calculated classes differ markedly
from the supplied labels, so the comparison should be reviewed before
scientific interpretation.