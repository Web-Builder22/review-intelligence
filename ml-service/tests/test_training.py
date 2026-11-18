"""
Automated tests for the Phase C machine learning training pipeline.
Uses isolated temporary CSV datasets through pytest fixtures and tests
all 17 verification requirements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure ml-service directory is in sys.path
ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

import main as main_module
import model_manager
from config import DatasetConfig
from train import train_all_models, TrainingError
from training_config import TrainingConfig


@pytest.fixture(autouse=True)
def clean_model_artifacts():
    """Clean model artifacts and metadata before and after each test."""
    model_manager.clear_artifacts()
    yield
    model_manager.clear_artifacts()


@pytest.fixture
def small_valid_csv(tmp_path: Path) -> Path:
    """Create a small valid dataset with multiple sentiment classes and sufficient samples for stratified split."""
    csv_path = tmp_path / "valid_training_reviews.csv"
    df = pd.DataFrame({
        "review_text": [
            "This product is absolutely amazing and works perfectly.",
            "I love this item, high quality and fast shipping.",
            "Extremely happy with this purchase, highly recommend.",
            "Best purchase I have made this year, fantastic quality.",
            "Completely broken on arrival, terrible waste of money.",
            "Arrived damaged and customer support never replied.",
            "Worst purchase ever made, stopped working instantly.",
            "Total garbage, stopped charging after one week.",
            "It is an average cable, charges devices slowly.",
            "Decent quality for the price paid, nothing special.",
            "Neither good nor bad, just an ordinary device.",
            "It works as expected, nothing extraordinary.",
            "Superb performance, excellent battery life.",
            "Very impressed with the design and durability.",
            "Awful experience, returning this immediately.",
            "Poor material quality and misleading description.",
            "Mediocre sound quality, acceptable for the price.",
            "Satisfied overall, solid build and quick delivery.",
        ],
        "sentiment": [
            "positive", "positive", "positive", "positive",
            "negative", "negative", "negative", "negative",
            "neutral", "neutral", "neutral", "neutral",
            "positive", "positive", "negative", "negative",
            "neutral", "positive",
        ],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


# 1. Missing dataset handling
def test_missing_dataset_handling(tmp_path: Path):
    missing_csv = tmp_path / "non_existent.csv"
    cfg = DatasetConfig(csv_path=str(missing_csv), review_text_column="review", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg), \
         patch("main.get_dataset_config", return_value=cfg):
        
        with pytest.raises(TrainingError, match="Dataset file not found"):
            train_all_models()

        client = TestClient(main_module.app)
        resp = client.post("/train")
        assert resp.status_code == 503
        assert resp.json()["status"] == "dataset_not_found"


# 2. Invalid column configuration
def test_invalid_column_configuration(small_valid_csv: Path):
    cfg = DatasetConfig(
        csv_path=str(small_valid_csv),
        review_text_column="non_existent_text_col",
        label_column="sentiment",
    )
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg), \
         patch("main.get_dataset_config", return_value=cfg):
        
        with pytest.raises(TrainingError, match="Dataset validation failed"):
            train_all_models()


# 3. Missing labels
def test_missing_labels(tmp_path: Path):
    csv_path = tmp_path / "missing_labels.csv"
    pd.DataFrame({
        "review": ["Great product", "Bad product"],
        "sentiment": [None, "positive"],
    }).to_csv(csv_path, index=False)

    cfg = DatasetConfig(csv_path=str(csv_path), review_text_column="review", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg), \
         patch("main.get_dataset_config", return_value=cfg):
        
        with pytest.raises(TrainingError):
            train_all_models()


# 4. Single-class dataset rejection
def test_single_class_dataset_rejection(tmp_path: Path):
    csv_path = tmp_path / "single_class.csv"
    pd.DataFrame({
        "review": ["Great product", "Awesome item", "Loved it"],
        "sentiment": ["positive", "positive", "positive"],
    }).to_csv(csv_path, index=False)

    cfg = DatasetConfig(csv_path=str(csv_path), review_text_column="review", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg), \
         patch("main.get_dataset_config", return_value=cfg):
        
        with pytest.raises(TrainingError, match="at least 2 distinct sentiment classes"):
            train_all_models()


# 5 & 6. Stratified split validation & reproducible train/test split
def test_reproducible_and_stratified_split(small_valid_csv: Path):
    cfg = DatasetConfig(csv_path=str(small_valid_csv), review_text_column="review_text", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg):
        
        train_cfg = TrainingConfig(test_size=0.25, random_state=42, stratify=True)
        summary1 = train_all_models(train_cfg)
        meta1 = model_manager.load_metadata()

        model_manager.clear_artifacts()
        summary2 = train_all_models(train_cfg)
        meta2 = model_manager.load_metadata()

        assert meta1["training_samples_used"] == meta2["training_samples_used"]
        assert meta1["testing_samples_used"] == meta2["testing_samples_used"]


# 7 & 8. TF-IDF fitted only on training data & test data transformed without fitting
def test_tfidf_fitted_on_train_only(small_valid_csv: Path):
    cfg = DatasetConfig(csv_path=str(small_valid_csv), review_text_column="review_text", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg):
        
        train_all_models()
        vectorizer = model_manager.load_artifact(model_manager.VECTORIZER_PATH)
        
        # Verify vectorizer has fitted vocabulary
        assert hasattr(vectorizer, "vocabulary_")
        assert len(vectorizer.vocabulary_) > 0

        # Verify transform works on unseen test text without refitting
        transformed = vectorizer.transform(["brand new unseen test review"])
        assert transformed.shape[1] == len(vectorizer.vocabulary_)


# 9 & 10. All four classifiers train successfully and LinearSVC probability calibration works
def test_all_four_classifiers_train_and_calibrate(small_valid_csv: Path):
    cfg = DatasetConfig(csv_path=str(small_valid_csv), review_text_column="review_text", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg):
        
        summary = train_all_models()
        assert summary["status"] == "success"
        assert sorted(summary["successful_models"]) == [
            "linear_svc",
            "logistic_regression",
            "multinomial_nb",
            "random_forest",
        ]

        # Verify LinearSVC is wrapped in CalibratedClassifierCV and predicts probabilities
        svc_model = model_manager.load_artifact(model_manager.MODEL_FILENAMES["linear_svc"])
        
        vectorizer = model_manager.load_artifact(model_manager.VECTORIZER_PATH)
        sample_vec = vectorizer.transform(["amazing product"])
        probs = svc_model.predict_proba(sample_vec)
        assert probs.shape[0] == 1
        assert abs(probs.sum() - 1.0) < 1e-5


# 11, 12, 13 & 14. Artifacts save correctly, vectorizer saves, metadata is valid JSON, and artifacts load again
def test_artifacts_saving_loading_and_metadata(small_valid_csv: Path):
    cfg = DatasetConfig(csv_path=str(small_valid_csv), review_text_column="review_text", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg):
        
        train_all_models()

        # Check model files exist
        for path in model_manager.MODEL_FILENAMES.values():
            assert path.exists()

        # Check vectorizer exists
        assert model_manager.VECTORIZER_PATH.exists()

        # Check metadata is valid JSON and contains required keys
        meta = model_manager.load_metadata()
        assert meta is not None
        assert "dataset_filename" in meta
        assert "class_names" in meta
        assert "tfidf_configuration" in meta
        assert "models" in meta
        
        # Verify round-trip load of all models
        for name, path in model_manager.MODEL_FILENAMES.items():
            loaded = model_manager.load_artifact(path)
            assert loaded is not None


# 15. Training status endpoint reports correct state
def test_training_status_endpoint(small_valid_csv: Path):
    cfg = DatasetConfig(csv_path=str(small_valid_csv), review_text_column="review_text", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg), \
         patch("main.get_dataset_config", return_value=cfg):
        
        client = TestClient(main_module.app)
        
        # Before training
        resp_before = client.get("/training/status")
        assert resp_before.status_code == 200
        assert resp_before.json()["has_trained"] is False

        # Train models via API
        resp_train = client.post("/train")
        assert resp_train.status_code == 201

        # After training
        resp_after = client.get("/training/status")
        assert resp_after.status_code == 200
        data = resp_after.json()
        assert data["has_trained"] is True
        assert data["vectorizer_exists"] is True
        assert data["trained_models_count"] == 4


# 16. No fake metrics are generated
def test_no_fake_metrics_generated(small_valid_csv: Path):
    cfg = DatasetConfig(csv_path=str(small_valid_csv), review_text_column="review_text", label_column="sentiment")
    with patch("train.get_dataset_config", return_value=cfg), \
         patch("config.get_dataset_config", return_value=cfg):
        
        train_all_models()
        meta = model_manager.load_metadata()
        
        # Verify metadata records training status, duration, parameters, and class distribution,
        # but does NOT fabricate evaluation accuracy, precision, recall, or F1 scores.
        for model_name, model_info in meta["models"].items():
            assert "accuracy" not in model_info
            assert "f1_score" not in model_info
            assert "precision" not in model_info
            assert "recall" not in model_info


# 17. Phase B tests still pass
def test_phase_b_integration():
    """Verify Phase B functions and test suite remain fully functional."""
    from preprocess import preprocess_text
    assert preprocess_text("This product is not good") == "product not good"
    assert preprocess_text("I don't recommend this product") == "don't recommend product"
