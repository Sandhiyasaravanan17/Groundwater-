"""Configurable weighted-arithmetic WQI with explicit standard limits.

For one-sided parameters, qi = 100 * (Ci - Vi) / (Si - Vi). For pH,
the rating uses the lower or upper standard limit according to the direction
of deviation from the ideal. Weights are Wi = (1 / Si) / sum(1 / Sj), and
WQI = sum(Wi * qi). The configured method and standards are exposed in results.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


REQUIRED_PARAMETERS = (
    "nitrate_mg_l",
    "fluoride_mg_l",
    "chloride_mg_l",
    "tds_mg_l",
    "ph",
    "iron_mg_l",
)
PARAMETER_UNITS = {
    "nitrate_mg_l": "mg/l",
    "fluoride_mg_l": "mg/l",
    "chloride_mg_l": "mg/l",
    "tds_mg_l": "mg/l",
    "ph": "unitless",
    "iron_mg_l": "mg/l",
    "alkalinity_mg_l": "mg/l",
}


class WQIMethodologyError(ValueError):
    """Raised when a WQI configuration is incomplete or internally invalid."""


def _finite_number(value, label):
    if isinstance(value, bool):
        raise WQIMethodologyError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise WQIMethodologyError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise WQIMethodologyError(f"{label} must be a finite number")
    return number


@dataclass(frozen=True)
class ParameterLimit:
    standard_limit: float
    ideal_value: float
    unit: str
    lower_limit: Optional[float] = None


@dataclass(frozen=True)
class ClassificationBand:
    name: str
    lower_bound: Optional[float]
    upper_bound: Optional[float]


@dataclass(frozen=True)
class WQIMethodology:
    method_reference: str
    standard_source: str
    classification_source: str
    parameter_limits: Mapping[str, ParameterLimit]
    classification_bands: Tuple[ClassificationBand, ...]

    @classmethod
    def from_mapping(cls, configuration):
        if not isinstance(configuration, Mapping):
            raise WQIMethodologyError("A WQI methodology configuration is required")

        references = {}
        for key in ("method_reference", "standard_source", "classification_source"):
            value = configuration.get(key)
            if not isinstance(value, str) or not value.strip():
                raise WQIMethodologyError(f"{key} must identify a documented source")
            references[key] = value.strip()

        configured_limits = configuration.get("parameters")
        if not isinstance(configured_limits, Mapping):
            raise WQIMethodologyError("parameters must define limits for all required measurements")
        missing_parameters = sorted(set(REQUIRED_PARAMETERS) - set(configured_limits))
        extra_parameters = sorted(set(configured_limits) - set(REQUIRED_PARAMETERS))
        if missing_parameters or extra_parameters:
            details = []
            if missing_parameters:
                details.append("missing: " + ", ".join(missing_parameters))
            if extra_parameters:
                details.append("unsupported: " + ", ".join(extra_parameters))
            raise WQIMethodologyError("Parameter configuration mismatch (" + "; ".join(details) + ")")

        parameter_limits = {}
        for name in REQUIRED_PARAMETERS:
            raw_limit = configured_limits[name]
            if not isinstance(raw_limit, Mapping):
                raise WQIMethodologyError(f"parameters.{name} must be an object")
            standard_limit = _finite_number(
                raw_limit.get("standard_limit"), f"parameters.{name}.standard_limit"
            )
            ideal_value = _finite_number(
                raw_limit.get("ideal_value"), f"parameters.{name}.ideal_value"
            )
            raw_lower_limit = raw_limit.get("lower_limit")
            lower_limit = (
                None
                if raw_lower_limit is None
                else _finite_number(raw_lower_limit, f"parameters.{name}.lower_limit")
            )
            unit = raw_limit.get("unit")
            if not isinstance(unit, str) or unit.strip().casefold() != PARAMETER_UNITS[name]:
                raise WQIMethodologyError(
                    f"parameters.{name}.unit must be {PARAMETER_UNITS[name]}"
                )
            if standard_limit <= 0 or standard_limit == ideal_value:
                raise WQIMethodologyError(
                    f"parameters.{name}.standard_limit must be positive and differ from ideal_value"
                )
            if name == "ph":
                if lower_limit is None or not lower_limit < ideal_value < standard_limit:
                    raise WQIMethodologyError(
                        "parameters.ph must define lower_limit < ideal_value < standard_limit"
                    )
            elif lower_limit is not None or ideal_value < 0 or ideal_value >= standard_limit:
                raise WQIMethodologyError(
                    f"parameters.{name} must have 0 <= ideal_value < standard_limit and no lower_limit"
                )
            parameter_limits[name] = ParameterLimit(
                standard_limit=standard_limit,
                ideal_value=ideal_value,
                unit=PARAMETER_UNITS[name],
                lower_limit=lower_limit,
            )

        bands = cls._parse_bands(configuration.get("classification_bands"))
        return cls(
            method_reference=references["method_reference"],
            standard_source=references["standard_source"],
            classification_source=references["classification_source"],
            parameter_limits=parameter_limits,
            classification_bands=bands,
        )

    @staticmethod
    def _parse_bands(raw_bands):
        if not isinstance(raw_bands, list) or not raw_bands:
            raise WQIMethodologyError("classification_bands must define the complete WQI class scale")

        bands = []
        for index, raw_band in enumerate(raw_bands):
            if not isinstance(raw_band, Mapping):
                raise WQIMethodologyError("Each classification band must be an object")
            name = raw_band.get("name")
            if not isinstance(name, str) or not name.strip():
                raise WQIMethodologyError("Each classification band must have a name")
            lower = raw_band.get("lower_bound")
            upper = raw_band.get("upper_bound")
            lower = None if lower is None else _finite_number(lower, f"classification_bands[{index}].lower_bound")
            upper = None if upper is None else _finite_number(upper, f"classification_bands[{index}].upper_bound")
            if lower is not None and upper is not None and lower >= upper:
                raise WQIMethodologyError(f"Classification band {name!r} must have lower_bound < upper_bound")
            bands.append(ClassificationBand(name.strip(), lower, upper))

        if bands[0].lower_bound is not None or bands[-1].upper_bound is not None:
            raise WQIMethodologyError("Classification bands must cover values from negative to positive infinity")
        for previous, current in zip(bands, bands[1:]):
            if previous.upper_bound is None or previous.upper_bound != current.lower_bound:
                raise WQIMethodologyError("Classification bands must be ordered, contiguous, and non-overlapping")
        return tuple(bands)

    def calculate(self, measurements):
        if not isinstance(measurements, Mapping):
            raise WQIMethodologyError("Observed measurements must be provided as a mapping")

        missing = [name for name in REQUIRED_PARAMETERS if name not in measurements]
        if missing:
            raise WQIMethodologyError("Missing observed measurements: " + ", ".join(missing))

        values = {
            name: _finite_number(measurements[name], f"measurements.{name}")
            for name in REQUIRED_PARAMETERS
        }
        inverse_limits = {
            name: 1.0 / self.parameter_limits[name].standard_limit
            for name in REQUIRED_PARAMETERS
        }
        inverse_limit_sum = sum(inverse_limits.values())

        contributions = {}
        score = 0.0
        for name in REQUIRED_PARAMETERS:
            limit = self.parameter_limits[name]
            if name == "ph" and values[name] < limit.ideal_value:
                rating = 100.0 * (limit.ideal_value - values[name]) / (
                    limit.ideal_value - limit.lower_limit
                )
            else:
                rating = 100.0 * (values[name] - limit.ideal_value) / (
                    limit.standard_limit - limit.ideal_value
                )
            weight = inverse_limits[name] / inverse_limit_sum
            contribution = rating * weight
            contributions[name] = {
                "observed_value": values[name],
                "unit": limit.unit,
                "standard_limit": limit.standard_limit,
                "lower_limit": limit.lower_limit,
                "ideal_value": limit.ideal_value,
                "quality_rating": rating,
                "normalized_weight": weight,
                "wqi_contribution": contribution,
            }
            score += contribution

        classification = self.classify(score)
        return {
            "wqi": score,
            "classification": classification,
            "method": "weighted_arithmetic",
            "references": {
                "method": self.method_reference,
                "standards": self.standard_source,
                "classification": self.classification_source,
            },
            "contributions": contributions,
        }

    def classify(self, score):
        score = _finite_number(score, "wqi")
        for band in self.classification_bands:
            if (band.lower_bound is None or score > band.lower_bound) and (
                band.upper_bound is None or score <= band.upper_bound
            ):
                return band.name
        raise WQIMethodologyError("WQI score is outside the configured classification bands")