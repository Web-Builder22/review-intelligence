"""
Model artifact management: saving, loading, and inspecting trained models,
vectorizers, and training metadata under ml-service/models/ and ml-service/artifacts/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import sklearn

from config import MODELS_DIR, ARTIFACTS_DIR

logger = logging.getLogger("ml-service.model_manager")

METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

MODEL_FILENAMES = {
    "linear_svc": MODELS_DIR / "linear_svc.joblib",
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "multinomial_nb": MODELS_DIR / "multinomial_nb.joblib",
}


def save_artifact(obj: Any, path: Path) -> None:
    """Save a Python object (model or vectorizer) using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    """Load a joblib artifact from disk. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at '{path}'.")
    return joblib.load(path)


def save_metadata(metadata: dict[str, Any]) -> None:
    """Save training metadata as a JSON file."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata() -> dict[str, Any] | None:
    """Load training metadata JSON if it exists, else return None."""
    if not METADATA_PATH.exists():
        return None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not parse training metadata JSON: %s", exc)
        return None


def get_training_status() -> dict[str, Any]:
    """Report honest status of training artifacts, vectorizer, and compatibility."""
    metadata = load_metadata()
    vectorizer_exists = VECTORIZER_PATH.exists()

    model_status: dict[str, bool] = {}
    for name, path in MODEL_FILENAMES.items():
        model_status[name] = path.exists()

    trained_models_count = sum(1 for exists in model_status.values() if exists)
    has_trained = vectorizer_exists and trained_models_count > 0

    return {
        "has_trained": has_trained,
        "vectorizer_exists": vectorizer_exists,
        "models_exist": model_status,
        "trained_models_count": trained_models_count,
        "total_models": len(MODEL_FILENAMES),
        "metadata_exists": metadata is not None,
        "metadata": metadata,
        "sklearn_version": sklearn.__version__,
    }


def clear_artifacts() -> None:
    """Remove existing model and vectorizer artifacts (used in testing)."""
    for path in MODEL_FILENAMES.values():
        if path.exists():
            path.unlink()
    if VECTORIZER_PATH.exists():
        VECTORIZER_PATH.unlink()
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()
