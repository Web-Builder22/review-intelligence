"""
Central configuration for the ML service.

Every dataset-specific setting lives here so a new dataset can be adopted
without touching any core code. Override any value with environment
variables (all optional) — see .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", BASE_DIR / "reports"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

for _dir in (MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR, DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def _clean_optional_str(value: str | None) -> str | None:
    if value is None or not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


@dataclass
class DatasetConfig:
    """Configuration for locating and interpreting the training CSV.

    Column names are intentionally NOT assumed: set them once here (or via
    env vars) when the dataset is supplied. `review_text_column` and
    `label_column` default to Summary and Sentiment respectively.
    """

    # Path to the CSV. Accepts an absolute path or one relative to ml-service/.
    csv_path: str = field(
        default_factory=lambda: os.getenv("DATASET_CSV_PATH", str(DATA_DIR / "reviews.csv"))
    )

    # Column containing the review text. Defaults to "Summary".
    review_text_column: str | None = field(
        default_factory=lambda: _clean_optional_str(os.getenv("DATASET_TEXT_COLUMN", "Summary"))
    )

    # Column containing the sentiment label. Defaults to "Sentiment".
    label_column: str | None = field(
        default_factory=lambda: _clean_optional_str(os.getenv("DATASET_LABEL_COLUMN", "Sentiment"))
    )

    # Product column is optional (used for per-product insights later).
    product_column: str | None = field(
        default_factory=lambda: _clean_optional_str(os.getenv("DATASET_PRODUCT_COLUMN"))
    )

    # Expected label values after any mapping. Classes not listed here are
    # reported during inspection as unexpected.
    expected_classes: tuple[str, ...] = ("negative", "neutral", "positive")

    # Only used when the dataset carries ratings instead of sentiment labels.
    rating_column: str | None = field(
        default_factory=lambda: _clean_optional_str(os.getenv("DATASET_RATING_COLUMN"))
    )
    rating_to_sentiment: dict[int, str] = field(
        default_factory=lambda: {
            1: "negative",
            2: "negative",
            3: "neutral",
            4: "positive",
            5: "positive",
        }
    )
    ratings_are_labels: bool = field(
        default_factory=lambda: os.getenv("DATASET_RATINGS_ARE_LABELS", "false").strip().lower() == "true"
    )

    def __post_init__(self) -> None:
        self.csv_path = self.csv_path.strip() if isinstance(self.csv_path, str) else ""
        self.review_text_column = _clean_optional_str(self.review_text_column)
        self.label_column = _clean_optional_str(self.label_column)
        self.product_column = _clean_optional_str(self.product_column)
        self.rating_column = _clean_optional_str(self.rating_column)
        if isinstance(self.expected_classes, (list, tuple, set)):
            cleaned_classes = tuple(
                str(c).strip().lower()
                for c in self.expected_classes
                if isinstance(c, str) and c.strip()
            )
            self.expected_classes = cleaned_classes
        else:
            self.expected_classes = ()

    def resolve_csv_path(self) -> Path:
        if not self.csv_path:
            return BASE_DIR / "data" / "reviews.csv"
        p = Path(self.csv_path)
        return p if p.is_absolute() else BASE_DIR / p

    def validate(self) -> list[str]:
        """Validate dataset configuration fields and return a list of issues."""
        errors: list[str] = []

        if not self.csv_path:
            errors.append("Dataset CSV path (csv_path) is empty.")

        if not self.review_text_column:
            errors.append("Review text column is not configured (DATASET_TEXT_COLUMN unset).")

        if not self.expected_classes:
            errors.append("expected_classes must contain at least one valid sentiment class name.")

        expected_set = set(self.expected_classes)

        if self.ratings_are_labels:
            if not self.rating_column:
                errors.append(
                    "Rating mapping is enabled (ratings_are_labels=True), but rating_column "
                    "(DATASET_RATING_COLUMN) is not configured."
                )
            elif self.review_text_column and self.review_text_column == self.rating_column:
                errors.append(
                    f"Invalid configuration: review_text_column and rating_column cannot be the same "
                    f"column ('{self.review_text_column}')."
                )

            if not isinstance(self.rating_to_sentiment, dict) or len(self.rating_to_sentiment) == 0:
                errors.append(
                    "Invalid rating_to_sentiment mapping: a non-empty dictionary is required when "
                    "ratings_are_labels=True."
                )
            else:
                invalid_keys: list[Any] = []
                empty_targets: list[Any] = []
                unknown_targets: set[str] = set()
                for k, v in self.rating_to_sentiment.items():
                    if isinstance(k, bool) or not isinstance(k, (int, float)):
                        invalid_keys.append(k)
                    if not isinstance(v, str) or not v.strip():
                        empty_targets.append(k)
                    else:
                        norm_target = v.strip().lower()
                        if expected_set and norm_target not in expected_set:
                            unknown_targets.add(norm_target)

                if invalid_keys:
                    errors.append(
                        f"Invalid rating_to_sentiment keys {invalid_keys}: rating keys must be numeric."
                    )
                if empty_targets:
                    errors.append(
                        f"Invalid rating_to_sentiment values for key(s) {empty_targets}: target sentiment "
                        "labels must be non-empty strings."
                    )
                if unknown_targets:
                    errors.append(
                        f"rating_to_sentiment maps to unknown sentiment class(es) {sorted(unknown_targets)} "
                        f"not in expected_classes {list(self.expected_classes)}."
                    )
        else:
            if not self.label_column:
                if self.rating_column:
                    errors.append(
                        "No label column configured (DATASET_LABEL_COLUMN unset). "
                        "rating_column is set, but ratings_are_labels is False."
                    )
                else:
                    errors.append("No label column configured (DATASET_LABEL_COLUMN unset).")
            elif self.review_text_column and self.review_text_column == self.label_column:
                errors.append(
                    f"Invalid configuration: review_text_column and label_column cannot be the same "
                    f"column ('{self.review_text_column}')."
                )

        return errors

    def has_required_columns(self) -> bool:
        """Return True only when required columns and label/rating settings are valid."""
        return len(self.validate()) == 0


def get_dataset_config() -> DatasetConfig:
    """Build the active dataset configuration (env vars already applied)."""
    return DatasetConfig()


# --- Service settings ---

API_HOST = os.getenv("ML_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("ML_API_PORT", "8000"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_REVIEW_CHARS", "5000"))
