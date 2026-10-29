"""Observed-only temporal summaries for the normalized groundwater records."""

import math
from collections import Counter, defaultdict
from statistics import fmean, median

from ..preprocessing.pipeline import MEASUREMENT_FIELDS


SEASON_ORDER = ("Pre-Monsoon", "SW Monsoon", "NE Monsoon", "Post-Monsoon")
TEMPORAL_PARAMETER_FIELDS = (*MEASUREMENT_FIELDS, "wqi")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _year(value):
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if 1000 <= result <= 9999 else None


def _parameter_summary(values):
    result = {}
    for field in TEMPORAL_PARAMETER_FIELDS:
        valid_values = values[field]
        result[field] = {
            "count": len(valid_values),
            "mean": fmean(valid_values) if valid_values else None,
            "median": median(valid_values) if valid_values else None,
        }
    return result


def _group_summaries(rows, group_fields, order_key):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        if any(value is None for value in key):
            continue
        group = groups.setdefault(key, {
            "record_count": 0,
            "parameters": defaultdict(list),
        })
        group["record_count"] += 1
        for field in TEMPORAL_PARAMETER_FIELDS:
            value = _number(row.get(field))
            if value is not None:
                group["parameters"][field].append(value)

    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: order_key(item[0])):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["record_count"] = group["record_count"]
        summary["parameters"] = _parameter_summary(group["parameters"])
        summaries.append(summary)
    return summaries


def _class_distribution(rows, group_fields, order_key, class_field, output_field):
    counts = Counter()
    for row in rows:
        group = tuple(row.get(field) for field in group_fields)
        label = row.get(class_field)
        if any(value is None for value in group) or not label:
            continue
        counts[group + (str(label),)] += 1

    result = []
    for key, count in sorted(
        counts.items(),
        key=lambda item: (order_key(item[0][:-1]), item[0][-1]),
    ):
        record = {field: value for field, value in zip(group_fields, key[:-1])}
        record[output_field] = key[-1]
        record["record_count"] = count
        result.append(record)
    return result


def analyze_temporal_records(records):
    """Summarize observed chemistry and calculated WQI by year and season."""
    records = list(records)
    valid_rows = []
    invalid_year_count = 0
    for record in records:
        year = _year(record.get("year"))
        if year is None:
            invalid_year_count += 1
            continue
        season_value = record.get("season")
        season = str(season_value).strip() if season_value is not None else ""
        valid_rows.append({**record, "year": year, "season": season or None})

    years = sorted({row["year"] for row in valid_rows})
    seasons_present = {row["season"] for row in valid_rows if row["season"]}
    seasons = [season for season in SEASON_ORDER if season in seasons_present]
    seasons.extend(sorted(seasons_present - set(SEASON_ORDER)))

    def year_order(key):
        return key

    def year_season_order(key):
        season_rank = {season: index for index, season in enumerate(SEASON_ORDER)}
        return key[0], season_rank.get(key[1], len(SEASON_ORDER)), key[1]

    record_count_by_year = Counter(row["year"] for row in valid_rows)
    annual_parameter_summaries = _group_summaries(
        valid_rows, ("year",), year_order
    )
    seasonal_parameter_summaries = _group_summaries(
        [row for row in valid_rows if row["season"]],
        ("season",),
        lambda key: (SEASON_ORDER.index(key[0]) if key[0] in SEASON_ORDER else len(SEASON_ORDER), key[0]),
    )
    year_season_parameter_summaries = _group_summaries(
        [row for row in valid_rows if row["season"]],
        ("year", "season"),
        year_season_order,
    )

    limitations = [
        "Only observed measurements are summarized; no predicted values are included.",
        "The source provides year and season, not exact sampling dates or a verified time index.",
        "Unequal observation counts and repeated locations may affect aggregate comparability; the sampling design is not established.",
        "Source quality labels remain separate from calculated WQI classes.",
        "WQI summaries are descriptive aggregates; no statistical significance is claimed.",
        "These summaries describe the available records and do not establish causal or region-wide trends.",
    ]
    return {
        "observed_only": True,
        "time_coverage": {
            "first_year": years[0] if years else None,
            "last_year": years[-1] if years else None,
            "years": years,
            "seasons": seasons,
        },
        "record_count_by_year": [
            {"year": year, "record_count": record_count_by_year[year]}
            for year in years
        ],
        "annual_parameter_summary": annual_parameter_summaries,
        "seasonal_parameter_summary": seasonal_parameter_summaries,
        "year_season_parameter_summary": year_season_parameter_summaries,
        "annual_source_class_distribution": _class_distribution(
            valid_rows, ("year",), year_order,
            "quality_class_source", "source_quality_label",
        ),
        "seasonal_source_class_distribution": _class_distribution(
            [row for row in valid_rows if row["season"]],
            ("season",),
            lambda key: (SEASON_ORDER.index(key[0]) if key[0] in SEASON_ORDER else len(SEASON_ORDER), key[0]),
            "quality_class_source", "source_quality_label",
        ),
        "annual_wqi_class_distribution": _class_distribution(
            valid_rows, ("year",), year_order,
            "wqi_class", "wqi_class",
        ),
        "seasonal_wqi_class_distribution": _class_distribution(
            [row for row in valid_rows if row["season"]],
            ("season",),
            lambda key: (SEASON_ORDER.index(key[0]) if key[0] in SEASON_ORDER else len(SEASON_ORDER), key[0]),
            "wqi_class", "wqi_class",
        ),
        "input_records": len(records),
        "records_with_invalid_year": invalid_year_count,
        "limitations": limitations,
    }