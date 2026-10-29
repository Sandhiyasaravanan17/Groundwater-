"""Calculate and persist observed WQI results using the configured methodology."""

import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from .calculator import WQIMethodology
from ..preprocessing.pipeline import MEASUREMENT_FIELDS


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_METHODOLOGY = MODULE_DIR / "methodology.json"
WQI_OUTPUT_FIELDS = (
    "source_dataset",
    "source_record_number",
    "village",
    "block",
    "year",
    "season",
    "wqi",
    "wqi_class",
    "wqi_contributions",
)


@lru_cache(maxsize=4)
def load_methodology(configuration_path=None):
    path = Path(configuration_path) if configuration_path else DEFAULT_METHODOLOGY
    configuration = json.loads(path.read_text(encoding="utf-8"))
    return WQIMethodology.from_mapping(configuration), configuration


def score_observations(records, methodology):
    results = []
    for record in records:
        values = {}
        for field in methodology.parameter_limits:
            try:
                values[field] = float(record[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot calculate WQI for {record.get('source_dataset')} record "
                    f"{record.get('source_record_number')}: missing or invalid {field}"
                ) from exc
        score = methodology.calculate(values)
        results.append({
            "source_dataset": record.get("source_dataset"),
            "source_record_number": record.get("source_record_number"),
            "village": record.get("village"),
            "block": record.get("block"),
            "year": record.get("year"),
            "season": record.get("season"),
            "wqi": score["wqi"],
            "wqi_class": score["classification"],
            "wqi_contributions": score["contributions"],
        })
    return results


def summarize_scores(results, configuration):
    class_counts = Counter(result["wqi_class"] for result in results)
    scores = [result["wqi"] for result in results]
    return {
        "method": "weighted_arithmetic",
        "method_reference": configuration["method_reference"],
        "standard_source": configuration["standard_source"],
        "classification_source": configuration["classification_source"],
        "included_parameters": list(configuration["parameters"]),
        "excluded_parameters": configuration.get("excluded_parameters", {}),
        "interpretation_note": configuration.get("interpretation_note"),
        "record_count": len(results),
        "mean_wqi": fmean(scores) if scores else None,
        "minimum_wqi": min(scores) if scores else None,
        "maximum_wqi": max(scores) if scores else None,
        "class_distribution": dict(class_counts),
        "classification_bands": configuration["classification_bands"],
    }


def write_wqi_outputs(observations_path, output_directory, configuration_path=None):
    methodology, configuration = load_methodology(configuration_path)
    observations_path = Path(observations_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with observations_path.open("r", encoding="utf-8", newline="") as stream:
        observations = list(csv.DictReader(stream))
    results = score_observations(observations, methodology)
    summary = summarize_scores(results, configuration)

    output_path = output_directory / "water_quality_index.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=WQI_OUTPUT_FIELDS)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["wqi_contributions"] = json.dumps(row["wqi_contributions"], separators=(",", ":"))
            writer.writerow(row)

    summary_path = output_directory / "wqi_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def join_wqi_results(observations, wqi_results):
    by_source = {
        (result["source_dataset"], str(result["source_record_number"])): result
        for result in wqi_results
    }
    joined = []
    for observation in observations:
        key = (observation.get("source_dataset"), str(observation.get("source_record_number")))
        result = by_source.get(key)
        if result is None:
            raise ValueError(f"No WQI result found for source observation {key}")
        joined.append({**observation, **result})
    return joined
