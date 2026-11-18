"""
Comprehensive Phase B tests covering:
1. Missing NLTK resources (no download on import, graceful fallback when stopwords/wordnet are missing)
2. Negation preservation ("not", "no", "never", "nor", "without", etc.)
3. Contractions ("don't", "doesn't", "didn't", "isn't", "wasn't", "can't", "won't", curly apostrophes)
4. Missing text (None, NaN)
5. Missing labels (None, NaN)
6. Empty strings and whitespace-only strings in text and labels
7. Duplicate reviews with different capitalization and conflicting vs. matching sentiment labels
8. Invalid dataset configuration (missing columns, identical text/label columns, invalid rating mappings)
9. Missing CSV
10. Valid dataset inspection & JSON serializability (using temporary test CSVs only; real dataset path untouched)
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure ml-service directory is importable
ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

import config as config_module  # noqa: E402
import dataset as dataset_module  # noqa: E402
import main as main_module  # noqa: E402
import preprocess as preprocess_module  # noqa: E402
from config import DatasetConfig  # noqa: E402
from dataset import DatasetError, inspect_dataset, map_labels  # noqa: E402
from preprocess import (  # noqa: E402
    clean_dataframe,
    get_stopwords,
    lemmatize,
    normalize_review_for_dedup,
    preprocess_text,
    reset_nltk_caches,
)


# ---------------------------------------------------------------------------
# 1. Missing NLTK resources
# ---------------------------------------------------------------------------

def test_no_nltk_download_on_module_import():
    """Importing preprocess must never call nltk.download."""
    with patch("nltk.download", side_effect=AssertionError("nltk.download should not be called on import")) as mock_dl:
        importlib.reload(preprocess_module)
        assert mock_dl.call_count == 0


def test_missing_nltk_resources_fallback_and_api_resilience():
    """When NLTK stopwords and wordnet raise LookupError, preprocessing and /preprocess API still succeed."""
    reset_nltk_caches()

    with patch("nltk.data.find", side_effect=LookupError("NLTK corpora missing")):
        reset_nltk_caches()
        assert preprocess_module.check_nltk_resources() == {
            "stopwords": False,
            "wordnet": False,
            "omw-1.4": False,
        }
        stops = get_stopwords()
        assert "the" in stops
        assert "not" not in stops
        assert "don't" not in stops

        processed = preprocess_text("This product is not good and the boxes arrived damaged")
        assert "not" in processed.split()
        assert "product" in processed.split()
        assert "good" in processed.split()
        assert "box" in processed.split()

        # Verify FastAPI /preprocess endpoint does not crash when NLTK corpora are missing
        client = TestClient(main_module.app)
        resp = client.post("/preprocess", json={"text": "I don't recommend this product"})
        assert resp.status_code == 200
        tokens = resp.json()["data"]["processed"].split()
        assert "don't" in tokens
        assert "recommend" in tokens
        assert "product" in tokens

    reset_nltk_caches()


# ---------------------------------------------------------------------------
# 2 & 3. Negation preservation and contractions
# ---------------------------------------------------------------------------

def test_required_negation_and_sentiment_examples():
    """Verify the four required review examples preserve sentiment-critical tokens."""
    ex1 = preprocess_text("This product is good")
    ex2 = preprocess_text("This product is not good")
    ex3 = preprocess_text("I don't recommend this product")
    ex4 = preprocess_text("I really like this product")

    assert ex1 == "product good"
    assert ex2 == "product not good"
    assert ex3 == "don't recommend product"
    assert ex4 == "really like product"
    assert ex1 != ex2


@pytest.mark.parametrize(
    "contraction",
    ["don't", "doesn't", "didn't", "isn't", "wasn't", "can't", "won't", "wouldn't", "shouldn't", "couldn't"],
)
def test_contractions_preserved(contraction: str):
    """All negative contractions must survive tokenization, stopword removal, and lemmatization."""
    sentence = f"I {contraction} buy this item again"
    processed = preprocess_text(sentence)
    tokens = processed.split()
    assert contraction in tokens, f"Expected '{contraction}' to be preserved in {tokens}"


def test_curly_apostrophe_contractions_preserved():
    """Curly/typographic apostrophes in contractions (e.g. don’t) must normalize to ASCII and be preserved."""
    processed = preprocess_text("I don’t recommend this product, it isn’t working")
    tokens = processed.split()
    assert "don't" in tokens
    assert "isn't" in tokens


@pytest.mark.parametrize("negation", ["not", "no", "never", "nor", "without", "cannot"])
def test_core_negation_words_preserved(negation: str):
    """Core negation words must never be removed as stopwords."""
    processed = preprocess_text(f"The package arrived {negation} quality")
    tokens = processed.split()
    assert negation in tokens


# ---------------------------------------------------------------------------
# 4, 5 & 6. Missing text, missing labels, and empty/whitespace strings
# ---------------------------------------------------------------------------

def test_missing_text_missing_labels_and_empty_strings(tmp_path: Path):
    """Both inspect_dataset and clean_dataframe must handle None, NaN, empty, and whitespace-only values."""
    csv_file = tmp_path / "missing_and_empty.csv"
    df_raw = pd.DataFrame({
        "review": [
            "This product is great",
            None,
            "",
            "    ",
            "Terrible battery life",
            "Decent for the price",
            "Fast shipping",
        ],
        "sentiment": [
            "positive",
            "positive",
            "negative",
            "neutral",
            None,
            "",
            "   ",
        ],
    })
    df_raw.to_csv(csv_file, index=False)

    cfg = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="review",
        label_column="sentiment",
    )
    report = inspect_dataset(cfg)

    # 3 rows have missing/empty/whitespace text (None, "", "    ")
    assert report.missing_text == 3
    # 3 rows have missing/empty/whitespace labels (None, "", "   ")
    assert report.missing_labels == 3
    # Empty/whitespace labels must NOT appear as a "" class in label_classes
    assert "" not in report.label_classes

    cleaned_df, stats = clean_dataframe(df_raw, "review", "sentiment")
    assert stats["dropped_missing_or_empty_text"] == 3
    assert stats["dropped_missing_labels"] == 3
    assert stats["remaining_rows"] == 1
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["review"] == "This product is great"


def test_preprocess_text_empty_and_whitespace():
    """preprocess_text returns empty string for empty, whitespace-only, or non-string inputs."""
    assert preprocess_text("") == ""
    assert preprocess_text("    \t\n  ") == ""
    assert preprocess_text("!!!") == ""


# ---------------------------------------------------------------------------
# 7. Duplicate reviews with different capitalization & conflicting labels
# ---------------------------------------------------------------------------

def test_duplicate_reviews_case_insensitive_consistency(tmp_path: Path):
    """Duplicate reviews differing only in capitalization/whitespace/punctuation are detected consistently."""
    csv_file = tmp_path / "duplicates.csv"
    df_raw = pd.DataFrame({
        "review": [
            "This product is good",
            "this product is good",
            "  THIS PRODUCT IS GOOD! ",
            "I don't recommend this product",
        ],
        "sentiment": [
            "positive",
            "positive",
            "positive",
            "negative",
        ],
    })
    df_raw.to_csv(csv_file, index=False)

    cfg = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="review",
        label_column="sentiment",
    )
    report = inspect_dataset(cfg)
    assert report.duplicate_reviews == 2
    assert report.conflicting_duplicate_reviews == 0

    cleaned_df, stats = clean_dataframe(df_raw, "review", "sentiment")
    assert stats["dropped_duplicate_reviews"] == 2
    assert stats["conflicting_label_duplicates"] == 0
    assert len(cleaned_df) == 2


def test_duplicate_reviews_with_different_sentiment_labels_reported_and_not_silently_dropped(tmp_path: Path):
    """Reviews with identical normalized text but different sentiment labels are reported and not silently dropped."""
    csv_file = tmp_path / "conflicting_duplicates.csv"
    df_raw = pd.DataFrame({
        "review": [
            "The battery lasts about five hours",
            "the battery lasts about five hours",
            "Awesome sound quality",
        ],
        "sentiment": [
            "positive",
            "negative",
            "positive",
        ],
    })
    df_raw.to_csv(csv_file, index=False)

    cfg = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="review",
        label_column="sentiment",
    )
    report = inspect_dataset(cfg)
    assert report.duplicate_reviews == 1
    assert report.conflicting_duplicate_reviews == 1
    assert any("multiple distinct sentiment labels" in note for note in report.notes)

    cleaned_df, stats = clean_dataframe(df_raw, "review", "sentiment")
    # Conflicting sentiment rows are reported and preserved by default
    assert stats["conflicting_label_duplicates"] == 1
    assert stats["conflicting_label_rows"] == 2
    assert stats["dropped_duplicate_reviews"] == 0
    assert len(cleaned_df) == 3


# ---------------------------------------------------------------------------
# 8. Invalid dataset configuration & rating mapping validation
# ---------------------------------------------------------------------------

def test_invalid_dataset_configurations(tmp_path: Path):
    """Invalid column configurations and invalid rating mappings return meaningful errors."""
    csv_file = tmp_path / "sample.csv"
    pd.DataFrame({
        "review_body": ["Great item", "Bad item"],
        "label": ["positive", "negative"],
        "stars": [5, 99],
    }).to_csv(csv_file, index=False)

    # Case A: Unset text and label columns
    cfg_unset = DatasetConfig(csv_path=str(csv_file), review_text_column=None, label_column=None)
    assert not cfg_unset.has_required_columns()
    rep_unset = inspect_dataset(cfg_unset)
    assert not rep_unset.configured_correctly
    assert any("DATASET_TEXT_COLUMN" in issue for issue in rep_unset.issues)
    assert any("DATASET_LABEL_COLUMN" in issue for issue in rep_unset.issues)

    # Case B: Non-existent column names
    cfg_wrong_cols = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="wrong_text",
        label_column="wrong_label",
    )
    rep_wrong_cols = inspect_dataset(cfg_wrong_cols)
    assert not rep_wrong_cols.configured_correctly
    assert not rep_wrong_cols.text_column_found
    assert not rep_wrong_cols.label_column_found
    assert any("wrong_text" in issue for issue in rep_wrong_cols.issues)
    assert any("wrong_label" in issue for issue in rep_wrong_cols.issues)

    # Case C: Same column configured for both text and label
    cfg_same_col = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="review_body",
        label_column="review_body",
    )
    assert not cfg_same_col.has_required_columns()
    rep_same = inspect_dataset(cfg_same_col)
    assert not rep_same.configured_correctly
    assert any("cannot be the same column" in issue for issue in rep_same.issues)

    # Case D: ratings_are_labels=True but rating_column is missing
    cfg_no_rating_col = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="review_body",
        ratings_are_labels=True,
        rating_column=None,
    )
    assert not cfg_no_rating_col.has_required_columns()
    rep_no_rating = inspect_dataset(cfg_no_rating_col)
    assert not rep_no_rating.configured_correctly
    assert any("rating_column" in issue for issue in rep_no_rating.issues)

    # Case E: ratings_are_labels=True with invalid target sentiment in rating_to_sentiment
    cfg_bad_map = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="review_body",
        ratings_are_labels=True,
        rating_column="stars",
        rating_to_sentiment={1: "negative", 5: "super_happy"},
    )
    assert not cfg_bad_map.has_required_columns()
    rep_bad_map = inspect_dataset(cfg_bad_map)
    assert not rep_bad_map.configured_correctly
    assert any("super_happy" in issue for issue in rep_bad_map.issues)

    # Case F: ratings_are_labels=True with unmapped rating value (99) in CSV
    cfg_unmapped_val = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="review_body",
        ratings_are_labels=True,
        rating_column="stars",
    )
    rep_unmapped = inspect_dataset(cfg_unmapped_val)
    assert not rep_unmapped.configured_correctly
    assert any("no mapping in rating_to_sentiment" in issue for issue in rep_unmapped.issues)


# ---------------------------------------------------------------------------
# 9. Missing CSV
# ---------------------------------------------------------------------------

def test_missing_csv_handling(tmp_path: Path):
    """Missing CSV returns exists=False, configured_correctly=False, and 503 via API."""
    missing_path = tmp_path / "does_not_exist.csv"
    cfg = DatasetConfig(
        csv_path=str(missing_path),
        review_text_column="review",
        label_column="sentiment",
    )
    report = inspect_dataset(cfg)
    assert report.exists is False
    assert report.configured_correctly is False
    assert "File does not exist." in report.issues
    # Must be JSON serializable
    serialized = json.dumps(report.to_dict())
    assert '"exists": false' in serialized

    with patch.object(main_module, "get_dataset_config", return_value=cfg):
        client = TestClient(main_module.app)
        resp = client.get("/dataset/inspect")
        assert resp.status_code == 503
        body = resp.json()
        assert body["data"]["exists"] is False


# ---------------------------------------------------------------------------
# 10. Valid dataset inspection & JSON serializability
# ---------------------------------------------------------------------------

def test_valid_dataset_inspection_and_json_serializable(tmp_path: Path):
    """Valid dataset inspection reports accurate metrics, configured_correctly=True, and serializes cleanly."""
    csv_file = tmp_path / "valid_reviews.csv"
    pd.DataFrame({
        "review_text": [
            "This product is good",
            "This product is not good",
            "I don't recommend this product",
            "I really like this product",
            "The package arrived on Tuesday.",
        ],
        "sentiment": [
            "positive",
            "negative",
            "negative",
            "positive",
            "neutral",
        ],
        "product": ["Earbuds", "Earbuds", "Watch", "Lamp", "Cable"],
    }).to_csv(csv_file, index=False)

    cfg = DatasetConfig(
        csv_path=str(csv_file),
        review_text_column="review_text",
        label_column="sentiment",
        product_column="product",
    )
    report = inspect_dataset(cfg)
    assert report.exists is True
    assert report.configured_correctly is True
    assert report.total_rows == 5
    assert report.total_columns == 3
    assert report.missing_text == 0
    assert report.missing_labels == 0
    assert report.duplicate_reviews == 0
    assert report.conflicting_duplicate_reviews == 0
    assert report.label_classes == {"positive": 2, "negative": 2, "neutral": 1}
    assert report.issues == []

    # Verify strict JSON serializability with standard json.dumps
    payload_str = json.dumps(report.to_dict())
    parsed = json.loads(payload_str)
    assert parsed["configured_correctly"] is True
    assert parsed["total_rows"] == 5

    # Verify FastAPI endpoint returns 200 for valid dataset
    with patch.object(main_module, "get_dataset_config", return_value=cfg):
        client = TestClient(main_module.app)
        resp = client.get("/dataset/inspect")
        assert resp.status_code == 200
        assert resp.json()["data"]["configured_correctly"] is True

    # Ensure the real project data directory was NOT polluted with any test CSV
    real_default_csv = config_module.DATA_DIR / "reviews.csv"
    assert not real_default_csv.exists()
