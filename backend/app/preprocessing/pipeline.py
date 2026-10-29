"""Normalize the supplied headerless groundwater TSV exports."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_FILES = ("Melbhuvanagiri.csv", "Kurinjipadi.csv")
EXPECTED_FIELDS = 20
SEASONS = {
    "pre-monsoon": "Pre-Monsoon",
    "sw monsoon": "SW Monsoon",
    "ne monsoon": "NE Monsoon",
    "post-monsoon": "Post-Monsoon",
}
PARAMETER_ALIASES = {
    "nitrate": "nitrate_mg_l",
    "fluoride": "fluoride_mg_l",
    "chloride": "chloride_mg_l",
    "tds": "tds_mg_l",
    "ph": "ph",
    "iron": "iron_mg_l",
    "alkalinity": "alkalinity_mg_l",
}
PARAMETER_PATTERN = re.compile(
    r"([^,\[\]]+)\[([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s*([^\]]+))?\]"
)
NUMERIC_FIELDS = (
    ("latitude", 5),
    ("longitude", 6),
    ("source_record_index", 9),
    ("source_numeric_15", 15),
    ("source_numeric_16", 16),
    ("source_numeric_17", 17),
    ("source_numeric_18", 18),
)
MEASUREMENT_FIELDS = (
    "nitrate_mg_l",
    "fluoride_mg_l",
    "chloride_mg_l",
    "tds_mg_l",
    "ph",
    "iron_mg_l",
    "alkalinity_mg_l",
)
FIELD_NAMES = (
    "state",
    "district",
    "block",
    "village",
    "latitude",
    "longitude",
    "year",
    "season",
    "source_record_index",
    "well_type",
    "parameter_text_group_1_raw",
    "parameter_text_group_1_duplicate_raw",
    "parameter_text_group_2_raw",
    "parameter_text_group_2_duplicate_raw",
    "source_numeric_15",
    "source_numeric_16",
    "source_numeric_17",
    "source_numeric_18",
    "quality_class_source",
    "risk_class_source",
)
RAW_FIELD_NAMES = tuple(f"raw_field_{index:02d}" for index in range(1, EXPECTED_FIELDS + 1))
OUTPUT_FIELDS = (
    "source_dataset",
    "source_record_number",
    *FIELD_NAMES[:10],
    *MEASUREMENT_FIELDS,
    *FIELD_NAMES[10:14],
    *FIELD_NAMES[14:],
    "validation_issues",
    *RAW_FIELD_NAMES,
)


def parse_number(value):
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_parameters(text):
    parameters = {}
    issues = []
    matches = list(PARAMETER_PATTERN.finditer(text))
    for match in matches:
        source_name, raw_value, raw_unit = match.groups()
        field_name = PARAMETER_ALIASES.get(source_name.strip().casefold())
        if field_name is None:
            issues.append(f"unrecognized parameter name: {source_name.strip()}")
            continue
        if field_name in parameters:
            issues.append(f"duplicate parameter in text: {source_name.strip()}")
            continue
        number = parse_number(raw_value)
        if number is None:
            issues.append(f"invalid numeric value for {source_name.strip()}")
            continue
        unit = raw_unit.strip().casefold().replace(" ", "") if raw_unit else ""
        expected_unit = "" if field_name == "ph" else "mg/l"
        if unit != expected_unit:
            issues.append(f"unexpected unit for {source_name.strip()}: {raw_unit or '(none)'}")
            continue
        parameters[field_name] = number
    if text.strip() and not matches:
        issues.append("parameter text did not match the expected Name[value unit] format")
    return parameters, issues


def number_or_issue(value, field_name, row_issues):
    number = parse_number(value)
    if value.strip() and number is None:
        row_issues.append(f"{field_name} is not a finite number")
    return number


def normalize_record(values, source_name, record_number):
    row_issues = []
    cleaned = [value.strip() for value in values]
    output = {
        "source_dataset": source_name,
        "source_record_number": record_number,
    }
    for index, field_name in enumerate(FIELD_NAMES):
        output[field_name] = cleaned[index]

    for field_name, source_position in NUMERIC_FIELDS:
        output[field_name] = number_or_issue(
            cleaned[source_position - 1], field_name, row_issues
        )

    year_value = cleaned[6]
    if year_value:
        try:
            year_number = int(year_value)
            if str(year_number) != year_value or not 1000 <= year_number <= 9999:
                raise ValueError
            output["year"] = year_number
        except ValueError:
            output["year"] = None
            row_issues.append("year is not a four-digit integer")
    else:
        output["year"] = None

    output["season"] = SEASONS.get(cleaned[7].casefold(), cleaned[7])
    if cleaned[7] and cleaned[7].casefold() not in SEASONS:
        row_issues.append(f"unrecognized season: {cleaned[7]}")

    if output["latitude"] is not None and not -90 <= output["latitude"] <= 90:
        row_issues.append("latitude is outside WGS84 bounds")
        output["latitude"] = None
    if output["longitude"] is not None and not -180 <= output["longitude"] <= 180:
        row_issues.append("longitude is outside WGS84 bounds")
        output["longitude"] = None

    if cleaned[2].casefold() != source_name.casefold():
        row_issues.append("block value does not match the source filename")

    group_one, group_one_issues = parse_parameters(cleaned[10])
    group_one_duplicate, group_one_duplicate_issues = parse_parameters(cleaned[11])
    group_two, group_two_issues = parse_parameters(cleaned[12])
    group_two_duplicate, group_two_duplicate_issues = parse_parameters(cleaned[13])
    row_issues.extend(
        group_one_issues
        + group_one_duplicate_issues
        + group_two_issues
        + group_two_duplicate_issues
    )

    if group_one != group_one_duplicate:
        row_issues.append("duplicate parameter-text group 1 fields disagree; extracted values withheld")
        group_one = {}
    if group_two != group_two_duplicate:
        row_issues.append("duplicate parameter-text group 2 fields disagree; extracted values withheld")
        group_two = {}

    for field_name in MEASUREMENT_FIELDS:
        output[field_name] = group_one.get(field_name, group_two.get(field_name))

    output["validation_issues"] = "; ".join(row_issues)
    for index, raw_value in enumerate(values, start=1):
        output[f"raw_field_{index:02d}"] = raw_value
    return output


def field_dictionary():
    descriptions = {
        "state": "Inferred state name from positional values; source has no header.",
        "district": "Inferred district name from positional values; source has no header.",
        "block": "Inferred block name from positional values; source has no header.",
        "village": "Inferred village/location from positional values; source has no header.",
        "latitude": "Inferred from numeric field 5; requires independent study-area verification.",
        "longitude": "Inferred from numeric field 6; requires independent study-area verification.",
        "year": "Inferred integer year from field 7.",
        "season": "Season category from field 8.",
        "source_record_index": "Integer field 9; source meaning is undocumented.",
        "well_type": "Source/well category from field 10.",
        "parameter_text_group_1_raw": "Raw parameter-list text in field 11; group meaning is undocumented.",
        "parameter_text_group_1_duplicate_raw": "Repeated parameter-list text in field 12; preserved as supplied.",
        "parameter_text_group_2_raw": "Raw parameter-list text in field 13; group meaning is undocumented.",
        "parameter_text_group_2_duplicate_raw": "Repeated parameter-list text in field 14; preserved as supplied.",
        "source_numeric_15": "Numeric field 15; meaning and unit are not provided.",
        "source_numeric_16": "Numeric field 16; meaning and unit are not provided.",
        "source_numeric_17": "Numeric field 17; meaning and unit are not provided.",
        "source_numeric_18": "Numeric field 18; meaning and unit are not provided.",
        "quality_class_source": "Existing text category in field 19; not a numeric WQI.",
        "risk_class_source": "Existing text category in field 20; method is undocumented.",
    }
    result = {}
    for index, field_name in enumerate(FIELD_NAMES, start=1):
        result[field_name] = {
            "source_field_position": index,
            "description": descriptions[field_name],
        }
    for field_name in MEASUREMENT_FIELDS:
        result[field_name] = {
            "source_field_positions": [11, 12] if field_name in MEASUREMENT_FIELDS[:5] else [13, 14],
            "description": "Extracted numeric parameter; raw source text is retained in the raw fields.",
            "unit": "mg/L" if field_name.endswith("_mg_l") else "unitless",
        }
    for index, field_name in enumerate(RAW_FIELD_NAMES, start=1):
        result[field_name] = {
            "source_field_position": index,
            "description": "Unmodified source field retained for traceability.",
        }
    return result


def run_pipeline(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_files = [name for name in EXPECTED_FILES if not (input_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Required dataset files not found: {', '.join(missing_files)}")

    output_path = output_dir / "groundwater_observations.csv"
    audit_path = output_dir / "preprocessing_audit.json"
    seen_records = set()
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {},
        "operations": [
            "Read headerless tab-delimited source files without modifying them.",
            "Map the inspected positional schema and normalize numeric and season fields.",
            "Extract labeled chemistry values only when duplicate source fields agree.",
            "Preserve every source field in raw_field_01 through raw_field_20.",
            "Quarantine rows that do not contain exactly 20 fields.",
            "Remove only exact duplicate source records across the combined inputs.",
        ],
        "limitations": [
            "The source files have no headers; field meanings are inferred from their values and order.",
            "The meanings and units of source fields 15 through 18 are unknown and remain unlabeled.",
            "WGS84 coordinate bounds are checked, but no independent study-area boundary is available for validation.",
            "Existing quality and risk categories are retained as supplied; no WQI is calculated in this phase.",
        ],
        "field_dictionary": field_dictionary(),
        "quarantined_rows": [],
        "removed_duplicates": [],
        "validation_issues": [],
    }

    with output_path.open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.DictWriter(output_stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for file_name in EXPECTED_FILES:
            source_path = input_dir / file_name
            source_name = source_path.stem
            input_count = 0
            accepted_count = 0
            malformed_count = 0
            duplicate_count = 0
            with source_path.open("r", encoding="utf-8-sig", newline="") as source_stream:
                reader = csv.reader(source_stream, delimiter="\t", strict=True)
                for record_number, values in enumerate(reader, start=1):
                    input_count += 1
                    if len(values) != EXPECTED_FIELDS:
                        malformed_count += 1
                        audit["quarantined_rows"].append({
                            "source_file": file_name,
                            "source_record_number": record_number,
                            "field_count": len(values),
                            "raw_fields": values,
                            "reason": f"expected {EXPECTED_FIELDS} fields, found {len(values)}",
                        })
                        continue

                    identity = tuple(values)
                    if identity in seen_records:
                        duplicate_count += 1
                        audit["removed_duplicates"].append({
                            "source_file": file_name,
                            "source_record_number": record_number,
                            "reason": "exact match across all 20 source fields",
                        })
                        continue
                    seen_records.add(identity)

                    normalized = normalize_record(values, source_name, record_number)
                    if normalized["validation_issues"]:
                        audit["validation_issues"].append({
                            "source_file": file_name,
                            "source_record_number": record_number,
                            "issues": normalized["validation_issues"],
                        })
                    writer.writerow(normalized)
                    accepted_count += 1

            audit["inputs"][file_name] = {
                "records_read": input_count,
                "normalized_records_written": accepted_count,
                "quarantined_malformed_records": malformed_count,
                "removed_exact_duplicates": duplicate_count,
            }

    audit["summary"] = {
        "normalized_records_written": sum(
            item["normalized_records_written"] for item in audit["inputs"].values()
        ),
        "quarantined_malformed_records": len(audit["quarantined_rows"]),
        "removed_exact_duplicates": len(audit["removed_duplicates"]),
        "records_with_validation_issues": len(audit["validation_issues"]),
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main():
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=repository_root / "dataset")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "data" / "processed")
    arguments = parser.parse_args()
    audit = run_pipeline(arguments.input_dir, arguments.output_dir)
    print(json.dumps(audit["summary"], indent=2))


if __name__ == "__main__":
    main()