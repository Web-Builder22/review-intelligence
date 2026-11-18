"""
Dataset loading and inspection.

`inspect_dataset` never fabricates statistics: it reads the configured CSV
and reports exactly what is on disk, or raises a clear error explaining
what is missing or misconfigured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DatasetConfig
from preprocess import normalize_review_for_dedup


class DatasetError(Exception):
    """Raised when the dataset is missing, unreadable, or misconfigured."""


def _to_json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas/path objects into standard JSON-serializable Python types."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        return None if (math.isnan(fval) or math.isinf(fval)) else fval
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return _to_json_safe(vars(value))
    return str(value)


def _normalize_non_empty_strings(series: pd.Series) -> pd.Series:
    """Strip strings and convert empty or whitespace-only values to pd.NA."""
    s = series.astype("string").str.strip()
    return s.mask(s.isna() | (s.str.len() == 0), pd.NA)


@dataclass
class ColumnReport:
    name: str
    dtype: str
    non_null: int
    missing: int
    unique: int


@dataclass
class InspectionReport:
    file_path: str
    exists: bool
    total_rows: int = 0
    total_columns: int = 0
    columns: list[ColumnReport] = field(default_factory=list)
    label_classes: dict[str, int] = field(default_factory=dict)
    label_percentages: dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_reviews: int = 0
    conflicting_duplicate_reviews: int = 0
    missing_text: int = 0
    missing_labels: int = 0
    text_column_found: bool = False
    label_column_found: bool = False
    configured_correctly: bool = False
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        raw = {
            "file_path": self.file_path,
            "exists": self.exists,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "columns": [c.__dict__ for c in self.columns],
            "label_classes": self.label_classes,
            "label_percentages": self.label_percentages,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_reviews": self.duplicate_reviews,
            "conflicting_duplicate_reviews": self.conflicting_duplicate_reviews,
            "missing_text": self.missing_text,
            "missing_labels": self.missing_labels,
            "text_column_found": self.text_column_found,
            "label_column_found": self.label_column_found,
            "configured_correctly": self.configured_correctly,
            "issues": self.issues,
            "notes": self.notes,
        }
        return _to_json_safe(raw)


def validate_dataset_config(config: DatasetConfig) -> list[str]:
    """Validate a DatasetConfig and return a list of human-readable issues."""
    if not isinstance(config, DatasetConfig):
        raise DatasetError("Invalid config object: expected DatasetConfig.")
    return config.validate()


def load_raw_dataframe(config: DatasetConfig) -> pd.DataFrame:
    """Load the configured CSV verbatim (no cleaning). Raises DatasetError on failure."""
    if not config.csv_path or not str(config.csv_path).strip():
        raise DatasetError("Dataset CSV path is empty. Set DATASET_CSV_PATH in .env.")
    path = config.resolve_csv_path()
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found at '{path}'. "
            "Place the CSV in ml-service/data/ or set DATASET_CSV_PATH in .env."
        )
    if path.is_dir():
        raise DatasetError(f"Configured dataset path '{path}' is a directory, not a CSV file.")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise DatasetError(f"Dataset file at '{path}' is empty.") from e
    except Exception as e:  # unreadable encoding / malformed CSV
        raise DatasetError(f"Could not parse '{path}' as CSV: {e}") from e

    if len(df.columns) == 0:
        raise DatasetError(f"Dataset file at '{path}' has no columns.")
    return df


def _normalize_rating_mapping(config: DatasetConfig) -> dict[float, str]:
    """Validate and normalize rating_to_sentiment keys to floats and values to lowercase strings."""
    if not isinstance(config.rating_to_sentiment, dict) or len(config.rating_to_sentiment) == 0:
        raise DatasetError(
            "Invalid rating_to_sentiment configuration: mapping must be a non-empty dictionary "
            "when ratings_are_labels=True."
        )
    expected_set = {c.strip().lower() for c in config.expected_classes if isinstance(c, str) and c.strip()}
    normalized: dict[float, str] = {}
    for k, v in config.rating_to_sentiment.items():
        if isinstance(k, bool) or not isinstance(k, (int, float)):
            raise DatasetError(
                f"Invalid rating_to_sentiment key {k!r}: rating keys must be numeric."
            )
        if not isinstance(v, str) or not v.strip():
            raise DatasetError(
                f"Invalid rating_to_sentiment target for rating {k!r}: sentiment label must be a non-empty string."
            )
        target = v.strip().lower()
        if expected_set and target not in expected_set:
            raise DatasetError(
                f"rating_to_sentiment maps rating {k!r} to '{target}', which is not in "
                f"expected_classes {list(config.expected_classes)}."
            )
        normalized[float(k)] = target
    return normalized


def map_labels(df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
    """Return normalized string labels, applying the rating mapping only when
    explicitly configured (`ratings_are_labels=True`).

    Empty strings and whitespace-only strings are treated as missing (`pd.NA`).
    """
    if config.ratings_are_labels:
        if not config.rating_column:
            raise DatasetError(
                "Rating mapping is enabled (ratings_are_labels=True), but rating_column is not configured."
            )
        if config.rating_column not in df.columns:
            raise DatasetError(
                f"Configured rating column '{config.rating_column}' not found in dataframe columns: "
                f"{list(df.columns)}."
            )
        norm_mapping = _normalize_rating_mapping(config)
        raw_series = _normalize_non_empty_strings(df[config.rating_column])
        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        mapped = numeric_series.map(norm_mapping).astype("string")
        return mapped.mask(mapped.isna() | (mapped.str.strip().str.len() == 0), pd.NA)

    if config.label_column:
        if config.label_column not in df.columns:
            raise DatasetError(
                f"Configured label column '{config.label_column}' not found in dataframe columns: "
                f"{list(df.columns)}."
            )
        cleaned = _normalize_non_empty_strings(df[config.label_column])
        return cleaned.str.lower()

    raise DatasetError("No label column available: check label_column / rating settings.")


def inspect_dataset(config: DatasetConfig) -> InspectionReport:
    """Inspect the configured CSV and return a factual report. Never fabricates data."""
    path = config.resolve_csv_path()
    config_issues = validate_dataset_config(config)

    if not path.exists() or path.is_dir():
        report = InspectionReport(file_path=str(path), exists=False)
        report.issues.append("File does not exist.")
        for issue in config_issues:
            if issue not in report.issues:
                report.issues.append(issue)
        return report

    try:
        df = load_raw_dataframe(config)
    except DatasetError as exc:
        report = InspectionReport(file_path=str(path), exists=True)
        report.issues.append(str(exc))
        for issue in config_issues:
            if issue not in report.issues:
                report.issues.append(issue)
        return report

    report = InspectionReport(file_path=str(path), exists=True)
    report.total_rows = int(len(df))
    report.total_columns = int(len(df.columns))

    report.columns = []
    for c in df.columns:
        col_series = df[c]
        if pd.api.types.is_string_dtype(col_series) or col_series.dtype == object:
            cleaned_col = _normalize_non_empty_strings(col_series)
            non_null_count = int(cleaned_col.notna().sum())
            missing_count = int(cleaned_col.isna().sum())
            unique_count = int(cleaned_col.nunique(dropna=True))
        else:
            non_null_count = int(col_series.notna().sum())
            missing_count = int(col_series.isna().sum())
            unique_count = int(col_series.nunique(dropna=True))
        report.columns.append(
            ColumnReport(
                name=str(c),
                dtype=str(col_series.dtype),
                non_null=non_null_count,
                missing=missing_count,
                unique=unique_count,
            )
        )

    for issue in config_issues:
        if issue not in report.issues:
            report.issues.append(issue)

    if report.total_rows == 0:
        report.issues.append("Dataset CSV has headers but contains 0 data rows.")

    # --- configured text column checks ---
    text_col = config.review_text_column
    norm_text: pd.Series | None = None
    dedup_keys: pd.Series | None = None

    if text_col:
        if text_col not in df.columns:
            report.issues.append(
                f"Configured text column '{text_col}' not found. Available columns: "
                f"{', '.join(map(str, df.columns))}."
            )
        else:
            report.text_column_found = True
            norm_text = _normalize_non_empty_strings(df[text_col])
            # Also treat texts that normalize to empty (e.g. only punctuation/HTML) as missing
            dedup_keys = norm_text.map(
                lambda v: normalize_review_for_dedup(v) if pd.notna(v) else ""
            )
            empty_after_norm_mask = norm_text.isna() | (dedup_keys.str.len() == 0)
            report.missing_text = int(empty_after_norm_mask.sum())

            valid_dedup_keys = dedup_keys[~empty_after_norm_mask]
            report.duplicate_reviews = int(valid_dedup_keys.duplicated().sum())

            if report.total_rows > 0 and len(valid_dedup_keys) == 0:
                report.issues.append(
                    f"Configured text column '{text_col}' contains no valid (non-empty) review text."
                )

    # --- configured label / rating column checks ---
    resolved_labels: pd.Series | None = None

    if config.ratings_are_labels:
        if config.rating_column and config.rating_column not in df.columns:
            report.issues.append(
                f"Configured rating column '{config.rating_column}' not found in CSV. Available columns: "
                f"{', '.join(map(str, df.columns))}."
            )
        elif config.rating_column and config.rating_column in df.columns:
            report.label_column_found = True
            try:
                norm_mapping = _normalize_rating_mapping(config)
            except DatasetError as exc:
                if str(exc) not in report.issues:
                    report.issues.append(str(exc))
                norm_mapping = {}

            raw_ratings = _normalize_non_empty_strings(df[config.rating_column])
            missing_rating_mask = raw_ratings.isna()
            numeric_ratings = pd.to_numeric(raw_ratings, errors="coerce")

            if norm_mapping:
                mapped = numeric_ratings.map(norm_mapping).astype("string")
                mapped = mapped.mask(mapped.isna() | (mapped.str.strip().str.len() == 0), pd.NA)
                resolved_labels = mapped
                report.missing_labels = int(mapped.isna().sum())

                unmapped_mask = (~missing_rating_mask) & mapped.isna()
                unmapped = int(unmapped_mask.sum())
                unmapped_values = int(raw_ratings[unmapped_mask].nunique(dropna=True)) if unmapped else 0
                if unmapped:
                    report.issues.append(
                        f"{unmapped} row(s) have rating values with no mapping in rating_to_sentiment "
                        f"({unmapped_values} distinct unmapped value(s)) — review the mapping config."
                    )

                valid_mapped = mapped.dropna()
                counts = valid_mapped.value_counts()
                total_valid = int(counts.sum())
                report.label_classes = {str(k): int(v) for k, v in counts.items()}
                report.label_percentages = (
                    {str(k): round(float(v) / total_valid * 100, 2) for k, v in counts.items()}
                    if total_valid > 0
                    else {}
                )
                if report.total_rows > 0 and total_valid == 0:
                    report.issues.append(
                        f"Configured rating column '{config.rating_column}' produced 0 valid mapped sentiment labels."
                    )
    elif config.label_column:
        if config.label_column not in df.columns:
            report.issues.append(
                f"Configured label column '{config.label_column}' not found. Available columns: "
                f"{', '.join(map(str, df.columns))}."
            )
        else:
            report.label_column_found = True
            labels = map_labels(df, config)
            resolved_labels = labels
            report.missing_labels = int(labels.isna().sum())

            valid_labels = labels.dropna()
            counts = valid_labels.value_counts()
            total_valid = int(counts.sum())
            report.label_classes = {str(k): int(v) for k, v in counts.items()}
            report.label_percentages = (
                {str(k): round(float(v) / total_valid * 100, 2) for k, v in counts.items()}
                if total_valid > 0
                else {}
            )

            if report.total_rows > 0 and total_valid == 0:
                report.issues.append(
                    f"Configured label column '{config.label_column}' contains no valid (non-empty) labels."
                )

            expected_set = set(config.expected_classes)
            unexpected = set(report.label_classes.keys()) - expected_set
            if unexpected:
                report.issues.append(
                    f"Unexpected label values {sorted(unexpected)} not in expected classes "
                    f"{list(config.expected_classes)} — update expected_classes or the rating mapping."
                )

    # --- conflicting duplicate detection (same normalized text, different labels) ---
    if dedup_keys is not None and resolved_labels is not None and len(df) > 0:
        valid_pairs = pd.DataFrame({"_key": dedup_keys, "_label": resolved_labels})
        valid_pairs = valid_pairs[
            valid_pairs["_key"].notna()
            & (valid_pairs["_key"].str.len() > 0)
            & valid_pairs["_label"].notna()
        ]
        if not valid_pairs.empty:
            nunique_per_key = valid_pairs.groupby("_key")["_label"].nunique(dropna=True)
            conflicting_keys = nunique_per_key[nunique_per_key > 1]
            report.conflicting_duplicate_reviews = int(len(conflicting_keys))
            if report.conflicting_duplicate_reviews > 0:
                report.notes.append(
                    f"{report.conflicting_duplicate_reviews} normalized review text(s) appear with "
                    "multiple distinct sentiment labels."
                )

    report.duplicate_rows = int(df.duplicated().sum())
    report.configured_correctly = bool(
        report.exists
        and report.total_rows > 0
        and report.text_column_found
        and report.label_column_found
        and len(report.issues) == 0
    )

    if report.configured_correctly:
        report.notes.append("Dataset present and both required columns resolved; ready for preprocessing/training.")
    return report
