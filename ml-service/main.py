"""
FastAPI service exposing dataset inspection, preprocessing, and Phase C training pipeline.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import API_HOST, API_PORT, MAX_TEXT_LENGTH, get_dataset_config
from dataset import DatasetError, inspect_dataset
from model_manager import get_training_status
from preprocess import check_nltk_resources, preprocess_text
from train import TrainingError, train_all_models
from training_config import get_training_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ml-service")

app = FastAPI(
    title="E-Commerce Review Intelligence — ML Service",
    version="0.3.0-phase-c",
    description="Dataset inspection, text preprocessing, and machine learning model training pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lock and flag to prevent concurrent simultaneous training requests
_TRAINING_LOCK = threading.Lock()
_TRAINING_IN_PROGRESS = False


@app.exception_handler(DatasetError)
def dataset_error_handler(_req, exc: DatasetError):
    return JSONResponse(status_code=503, content={"error": str(exc), "detail": "dataset_unavailable"})


@app.get("/health")
def health():
    status_info = get_training_status()
    return {
        "status": "ok",
        "service": "ml-service",
        "phase": "C",
        "ml_trained": status_info["has_trained"],
        "nltk_resources": check_nltk_resources(),
    }


@app.get("/models/status")
def models_status():
    """Report trained model artifacts status."""
    status_info = get_training_status()
    return {
        "models_trained": status_info["has_trained"],
        "trained_models": status_info["successful_models"] if "successful_models" in status_info else [],
        "selected_model": None,
        "details": status_info,
    }


@app.get("/dataset/inspect")
def dataset_inspect():
    """Factual inspection of the configured CSV. Returns 503 when unavailable or invalid."""
    config = get_dataset_config()
    report = inspect_dataset(config)
    status_code = 200 if report.exists else 503
    return JSONResponse(status_code=status_code, content=jsonable_encoder({"data": report.to_dict()}))


@app.get("/dataset/config")
def dataset_config():
    """Show active dataset configuration and validation status."""
    config = get_dataset_config()
    validation_issues = config.validate()
    return {
        "csv_path": str(config.resolve_csv_path()),
        "review_text_column": config.review_text_column,
        "label_column": config.label_column,
        "product_column": config.product_column,
        "rating_column": config.rating_column,
        "ratings_are_labels": config.ratings_are_labels,
        "expected_classes": list(config.expected_classes),
        "configured_correctly": config.has_required_columns(),
        "validation_issues": validation_issues,
    }


@app.post("/preprocess")
def preprocess_endpoint(payload: dict):
    """Utility endpoint applying the shared preprocessing pipeline to text."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' (non-empty string) is required.")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters.")
    return {"data": {"original_length": len(text), "processed": preprocess_text(text)}}


# --- Phase C Training Endpoints ---


@app.post("/training/validate")
def training_validate():
    """Validate dataset readiness, sample counts, class distributions, and split feasibility without training."""
    config = get_dataset_config()
    report = inspect_dataset(config)

    if not report.exists:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "dataset_not_found",
                "error": f"Dataset CSV file not found at '{report.file_path}'.",
                "ready_for_training": False,
            },
        )

    if not report.configured_correctly or report.issues:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "invalid_dataset",
                "error": "Dataset validation failed.",
                "issues": report.issues,
                "ready_for_training": False,
            },
        )

    classes_valid = len(report.label_classes) >= 2
    min_count = min(report.label_classes.values()) if report.label_classes else 0
    stratifiable = classes_valid and min_count >= 2

    return {
        "status": "ready" if (classes_valid and stratifiable) else "warning",
        "ready_for_training": classes_valid,
        "total_rows": report.total_rows,
        "classes": report.label_classes,
        "class_percentages": report.label_percentages,
        "stratifiable": stratifiable,
        "issues": report.issues,
        "notes": report.notes,
    }


@app.post("/train")
def train_models():
    """Run the complete Phase C machine learning training pipeline across all four classifiers."""
    global _TRAINING_IN_PROGRESS

    if not _TRAINING_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is already in progress. Please wait for completion.",
        )

    try:
        _TRAINING_IN_PROGRESS = True
        config = get_dataset_config()
        report = inspect_dataset(config)

        if not report.exists:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "dataset_not_found",
                    "error": f"Dataset CSV file not found at '{report.file_path}'. Training cannot proceed.",
                },
            )

        summary = train_all_models()
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=jsonable_encoder(summary))

    except TrainingError as exc:
        logger.error("Training failed: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "failed", "error": str(exc)},
        )
    except Exception as exc:
        logger.exception("Unexpected training error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(exc)},
        )
    finally:
        _TRAINING_IN_PROGRESS = False
        _TRAINING_LOCK.release()


@app.get("/training/status")
def training_status():
    """Report whether training has occurred, artifact status, and compatibility."""
    status_info = get_training_status()
    config = get_dataset_config()
    dataset_exists = config.resolve_csv_path().exists()

    return {
        "training_in_progress": _TRAINING_IN_PROGRESS,
        "dataset_available": dataset_exists,
        "dataset_path": str(config.resolve_csv_path()),
        **status_info,
    }


@app.get("/training/config")
def training_config():
    """Return active training configuration and hyperparameter settings."""
    train_cfg = get_training_config()
    dataset_cfg = get_dataset_config()
    return {
        "training_parameters": train_cfg.to_dict(),
        "dataset_configuration": {
            "csv_path": str(dataset_cfg.resolve_csv_path()),
            "review_text_column": dataset_cfg.review_text_column,
            "label_column": dataset_cfg.label_column,
            "expected_classes": list(dataset_cfg.expected_classes),
            "ratings_are_labels": dataset_cfg.ratings_are_labels,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
