"""Select a deterministic, spatially representative observed-point map sample."""

import math
from collections import Counter, defaultdict

MAP_POINT_FIELDS = (
    "source_dataset",
    "source_record_number",
    "village",
    "block",
    "latitude",
    "longitude",
    "year",
    "season",
    "well_type",
    "quality_class_source",
    "risk_class_source",
    "wqi",
    "wqi_class",
)


def _coordinate(record, field):
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _class_key(record, class_field):
    value = record.get(class_field)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _allocate_class_quotas(class_counts, limit):
    ordered_classes = sorted(class_counts, key=lambda value: (value is not None, str(value)))
    if limit < len(ordered_classes):
        chosen = sorted(
            ordered_classes,
            key=lambda value: (-class_counts[value], value is not None, str(value)),
        )[:limit]
        return {value: 1 for value in chosen}

    quotas = {value: 1 for value in ordered_classes}
    remaining_slots = limit - len(ordered_classes)
    remaining_capacity = {value: class_counts[value] - 1 for value in ordered_classes}
    total_capacity = sum(remaining_capacity.values())
    if not remaining_slots or not total_capacity:
        return quotas

    allocations = {
        value: remaining_slots * remaining_capacity[value] / total_capacity
        for value in ordered_classes
    }
    for value in ordered_classes:
        addition = min(remaining_capacity[value], math.floor(allocations[value]))
        quotas[value] += addition

    slots_left = limit - sum(quotas.values())
    remainder_order = sorted(
        ordered_classes,
        key=lambda value: (
            -(allocations[value] - math.floor(allocations[value])),
            value is not None,
            str(value),
        ),
    )
    while slots_left:
        progressed = False
        for value in remainder_order:
            if quotas[value] < class_counts[value]:
                quotas[value] += 1
                slots_left -= 1
                progressed = True
                if not slots_left:
                    break
        if not progressed:
            break
    return quotas


def _time_order(record):
    year = record.get("year")
    try:
        year = int(year)
    except (TypeError, ValueError):
        year = -1
    return (
        year,
        str(record.get("season") or ""),
        str(record.get("village") or ""),
        str(record.get("source_dataset") or ""),
        str(record.get("source_record_number") or ""),
    )


def _sample_class(records, quota, grid_size, bounds, latitude_field, longitude_field):
    min_latitude, max_latitude, min_longitude, max_longitude = bounds
    latitude_span = max_latitude - min_latitude
    longitude_span = max_longitude - min_longitude
    cells = defaultdict(list)
    for record in records:
        latitude = float(record[latitude_field])
        longitude = float(record[longitude_field])
        row = 0 if latitude_span == 0 else min(
            grid_size - 1,
            int((latitude - min_latitude) / latitude_span * grid_size),
        )
        column = 0 if longitude_span == 0 else min(
            grid_size - 1,
            int((longitude - min_longitude) / longitude_span * grid_size),
        )
        cells[(row, column)].append(record)

    cell_keys = sorted(cells)
    cell_quotas = {key: 0 for key in cell_keys}
    allocated = 0
    while allocated < quota:
        progressed = False
        for key in cell_keys:
            if cell_quotas[key] < len(cells[key]):
                cell_quotas[key] += 1
                allocated += 1
                progressed = True
                if allocated == quota:
                    break
        if not progressed:
            break

    selected = []
    for key in cell_keys:
        bucket = sorted(cells[key], key=_time_order)
        take = cell_quotas[key]
        for rank in range(take):
            index = min(len(bucket) - 1, math.floor((rank + 0.5) * len(bucket) / take))
            selected.append(bucket[index])
    return selected


def sample_map_points(
    records,
    limit=500,
    latitude_field="latitude",
    longitude_field="longitude",
    class_field="quality_class_source",
):
    """Return at most ``limit`` valid unique coordinates and sampling metadata."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")

    records = list(records)
    unique_by_coordinate = {}
    invalid_coordinates = 0
    duplicate_coordinates = 0
    for record in records:
        latitude = _coordinate(record, latitude_field)
        longitude = _coordinate(record, longitude_field)
        if (
            latitude is None
            or longitude is None
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            invalid_coordinates += 1
            continue
        coordinate = (latitude, longitude)
        candidate = dict(record)
        candidate[latitude_field] = latitude
        candidate[longitude_field] = longitude
        previous = unique_by_coordinate.get(coordinate)
        if previous is None:
            unique_by_coordinate[coordinate] = candidate
            continue
        duplicate_coordinates += 1
        if _time_order(candidate) > _time_order(previous):
            unique_by_coordinate[coordinate] = candidate

    unique_records = list(unique_by_coordinate.values())
    coordinate_bounds = None
    if unique_records:
        coordinate_bounds = {
            "latitude_min": min(record[latitude_field] for record in unique_records),
            "latitude_max": max(record[latitude_field] for record in unique_records),
            "longitude_min": min(record[longitude_field] for record in unique_records),
            "longitude_max": max(record[longitude_field] for record in unique_records),
        }
    selected_records = unique_records
    if len(unique_records) > limit:
        class_groups = defaultdict(list)
        for record in unique_records:
            class_groups[_class_key(record, class_field)].append(record)
        class_counts = Counter({key: len(value) for key, value in class_groups.items()})
        quotas = _allocate_class_quotas(class_counts, limit)
        bounds = (
            min(record[latitude_field] for record in unique_records),
            max(record[latitude_field] for record in unique_records),
            min(record[longitude_field] for record in unique_records),
            max(record[longitude_field] for record in unique_records),
        )
        grid_size = max(1, math.ceil(math.sqrt(limit)))
        selected_records = []
        for class_value in sorted(quotas, key=lambda value: (value is not None, str(value))):
            selected_records.extend(_sample_class(
                class_groups[class_value],
                quotas[class_value],
                grid_size,
                bounds,
                latitude_field,
                longitude_field,
            ))

    points = []
    for record in selected_records:
        point = {field: record.get(field) for field in MAP_POINT_FIELDS}
        point["observation_status"] = "observed"
        points.append(point)

    class_counts = Counter(_class_key(record, class_field) for record in unique_records)
    selected_class_counts = Counter(_class_key(record, class_field) for record in points)
    truncated = len(unique_records) > len(points)
    return {
        "points": points,
        "metadata": {
            "input_records": len(records),
            "invalid_coordinate_records": invalid_coordinates,
            "duplicate_coordinate_records": duplicate_coordinates,
            "unique_coordinate_records": len(unique_records),
            "selected_records": len(points),
            "display_limit": limit,
            "is_limited": truncated,
            "message": (
                f"Showing {len(points)} representative locations from the full dataset."
                if truncated
                else f"Showing all {len(points)} valid unique locations."
            ),
            "class_counts": dict(class_counts),
            "display_class_counts": dict(selected_class_counts),
            "coordinate_bounds": coordinate_bounds,
            "coordinate_note": (
                "Coordinates are taken from inferred source fields. Their observed extent "
                "must be verified against the reported Cuddalore blocks; no study-area "
                "boundary or sub-basin coverage has been validated."
            ),
        },
    }