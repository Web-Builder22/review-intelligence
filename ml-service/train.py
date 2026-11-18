"""
Genuine machine learning training pipeline for Phase C.

Loads, cleans, splits, vectorizes with TF-IDF, trains four Scikit-learn
classifiers (LinearSVC with calibration, LogisticRegression, RandomForest,
and MultinomialNB), and saves model artifacts and training metadata.
"""

from __future__ import annotations

import logging
import math
import platform
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from config import get_dataset_config, MODELS_DIR
from dataset import DatasetError, inspect_dataset, load_raw_dataframe
from model_manager import (
    METADATA_PATH,
    VECTORIZER_PATH,
    MODEL_FILENAMES,
    save_artifact,
    save_metadata,
)
from preprocess import clean_dataframe
from training_config import get_training_config, TrainingConfig

logger = logging.getLogger("ml-service.train")


class TrainingError(Exception):
    """Raised when training fails due to data, configuration, or model errors."""


def train_all_models(train_config: TrainingConfig | None = None) -> dict[str, Any]:
    """Execute the full Phase C training pipeline.

    Returns an honest summary of trained models, training durations, and dataset details.
    """
    start_time = time.time()
    utc_timestamp = datetime.now(timezone.utc).isoformat()

    if train_config is None:
        train_config = get_training_config()

    dataset_config = get_dataset_config()

    # 1. Validate dataset configuration
    config_issues = dataset_config.validate()
    if config_issues:
        raise TrainingError(f"Dataset configuration invalid: {'; '.join(config_issues)}")

    # 2. Check dataset file existence
    csv_path = dataset_config.resolve_csv_path()
    if not csv_path.exists():
        raise TrainingError(
            f"Dataset file not found at '{csv_path}'. "
            "Place the CSV in ml-service/data/ or set DATASET_CSV_PATH in .env."
        )

    # 3. Inspect dataset
    report = inspect_dataset(dataset_config)
    if not report.exists:
        raise TrainingError(f"Dataset file does not exist at '{csv_path}'.")
    if not report.configured_correctly or report.issues:
        issues_str = "; ".join(report.issues)
        raise TrainingError(f"Dataset validation failed: {issues_str}")

    # 4. Load raw dataframe and clean dataset
    try:
        raw_df = load_raw_dataframe(dataset_config)
    except Exception as exc:
        raise TrainingError(f"Could not load dataset CSV: {exc}") from exc

    text_col = dataset_config.review_text_column
    label_col = dataset_config.label_column
    if not text_col:
        raise TrainingError("Review text column is not configured.")

    if dataset_config.ratings_are_labels:
        # If ratings are labels, map them into a temporary label column for cleaning/training
        from dataset import map_labels
        cleaned_labels = map_labels(raw_df, dataset_config)
        raw_df = raw_df.copy()
        raw_df["_resolved_label"] = cleaned_labels
        effective_label_col = "_resolved_label"
    else:
        effective_label_col = label_col

    if not effective_label_col:
        raise TrainingError("Sentiment label column is not configured.")

    clean_df, clean_stats = clean_dataframe(
        raw_df,
        text_column=text_col,
        label_column=effective_label_col,
        apply_text_pipeline=False,  # We will vectorize clean review texts directly
        config=dataset_config,
    )

    if len(clean_df) == 0:
        raise TrainingError("Dataset contains 0 valid rows after cleaning (all rows missing text or labels).")

    X_texts = clean_df[text_col].astype(str).tolist()
    y_labels = clean_df[effective_label_col].astype(str).tolist()

    # 5. Validate label classes and counts
    classes = sorted(list(set(y_labels)))
    n_classes = len(classes)
    if n_classes < 2:
        raise TrainingError(
            f"Dataset must contain at least 2 distinct sentiment classes for classification, "
            f"but found only {n_classes} class(es): {classes}."
        )

    class_counts = pd.Series(y_labels).value_counts().to_dict()
    min_class_count = min(class_counts.values())

    n_samples = len(X_texts)
    test_count = int(round(n_samples * train_config.test_size))
    train_count = n_samples - test_count

    # Check stratification feasibility
    stratify_labels = None
    if train_config.stratify:
        if min_class_count < 2:
            raise TrainingError(
                f"Cannot use stratified split: minority class has only {min_class_count} sample(s). "
                f"Class distribution: {class_counts}."
            )
        if test_count < n_classes or train_count < n_classes:
            raise TrainingError(
                f"Dataset size ({n_samples}) with test_size={train_config.test_size} is too small "
                f"for stratified split with {n_classes} classes (requires at least {n_classes} "
                f"train samples and {n_classes} test samples)."
            )
        stratify_labels = y_labels

    if n_samples < 5:
        raise TrainingError(f"Dataset is too small for training: only {n_samples} valid samples found.")

    # 6. Reproducible train/test split (split BEFORE fitting TF-IDF)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_texts,
            y_labels,
            test_size=train_config.test_size,
            random_state=train_config.random_state,
            stratify=stratify_labels,
        )
    except Exception as exc:
        raise TrainingError(f"Train/test split failed (insufficient samples per class): {exc}") from exc

    # 7. Fit TF-IDF Vectorizer strictly on training data only
    from preprocess import preprocess_text

    # Preprocess training and test texts using the shared pipeline
    X_train_processed = [preprocess_text(t) for t in X_train]
    X_test_processed = [preprocess_text(t) for t in X_test]

    # Filter out samples that became empty after preprocessing
    train_valid_pairs = [(txt, lbl) for txt, lbl in zip(X_train_processed, y_train) if txt.strip()]
    test_valid_pairs = [(txt, lbl) for txt, lbl in zip(X_test_processed, y_test) if txt.strip()]

    if not train_valid_pairs:
        raise TrainingError("All training reviews became empty after preprocessing.")

    X_train_clean, y_train_clean = zip(*train_valid_pairs)
    X_test_clean, y_test_clean = zip(*test_valid_pairs) if test_valid_pairs else ([], [])

    vectorizer = TfidfVectorizer(
        max_features=train_config.max_features,
        ngram_range=train_config.ngram_range,
        min_df=min(train_config.min_df, max(1, len(X_train_clean) // 2)),
        max_df=train_config.max_df,
        sublinear_tf=train_config.sublinear_tf,
    )

    try:
        X_train_tfidf = vectorizer.fit_transform(X_train_clean)
        X_test_tfidf = vectorizer.transform(X_test_clean) if X_test_clean else None
    except Exception as exc:
        raise TrainingError(f"TF-IDF vectorization failed: {exc}") from exc

    if X_train_tfidf.shape[1] == 0:
        raise TrainingError("TF-IDF vocabulary is empty after preprocessing and vectorization.")

    # 8. Train four classifiers
    model_results: dict[str, dict[str, Any]] = {}

    # --- Model 1: Linear Support Vector Machine (with CalibratedClassifierCV) ---
    svm_start = time.time()
    try:
        base_svc = LinearSVC(
            C=train_config.svm_c,
            class_weight=train_config.svm_class_weight,
            random_state=train_config.random_state,
            max_iter=2000,
        )
        # Wrap in CalibratedClassifierCV to obtain reliable probability estimates
        calibrated_svc = CalibratedClassifierCV(estimator=base_svc, method="sigmoid")
        calibrated_svc.fit(X_train_tfidf, y_train_clean)
        svm_duration = time.time() - svm_start

        save_artifact(calibrated_svc, MODEL_FILENAMES["linear_svc"])
        model_results["linear_svc"] = {
            "status": "success",
            "duration_seconds": round(svm_duration, 4),
            "parameters": {
                "estimator": "LinearSVC",
                "C": train_config.svm_c,
                "class_weight": train_config.svm_class_weight,
                "calibrated": True,
            },
        }
    except Exception as exc:
        logger.exception("LinearSVC training failed")
        model_results["linear_svc"] = {
            "status": "failed",
            "error": str(exc),
            "duration_seconds": round(time.time() - svm_start, 4),
        }

    # --- Model 2: Logistic Regression ---
    lr_start = time.time()
    try:
        lr_model = LogisticRegression(
            max_iter=train_config.lr_max_iter,
            class_weight=train_config.lr_class_weight,
            random_state=train_config.random_state,
        )
        lr_model.fit(X_train_tfidf, y_train_clean)
        lr_duration = time.time() - lr_start

        save_artifact(lr_model, MODEL_FILENAMES["logistic_regression"])
        model_results["logistic_regression"] = {
            "status": "success",
            "duration_seconds": round(lr_duration, 4),
            "parameters": {
                "max_iter": train_config.lr_max_iter,
                "class_weight": train_config.lr_class_weight,
                "random_state": train_config.random_state,
            },
        }
    except Exception as exc:
        logger.exception("LogisticRegression training failed")
        model_results["logistic_regression"] = {
            "status": "failed",
            "error": str(exc),
            "duration_seconds": round(time.time() - lr_start, 4),
        }

    # --- Model 3: Random Forest ---
    rf_start = time.time()
    try:
        rf_model = RandomForestClassifier(
            n_estimators=train_config.rf_n_estimators,
            max_depth=train_config.rf_max_depth,
            class_weight=train_config.rf_class_weight,
            random_state=train_config.random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train_tfidf, y_train_clean)
        rf_duration = time.time() - rf_start

        save_artifact(rf_model, MODEL_FILENAMES["random_forest"])
        model_results["random_forest"] = {
            "status": "success",
            "duration_seconds": round(rf_duration, 4),
            "parameters": {
                "n_estimators": train_config.rf_n_estimators,
                "max_depth": train_config.rf_max_depth,
                "class_weight": train_config.rf_class_weight,
                "random_state": train_config.random_state,
            },
        }
    except Exception as exc:
        logger.exception("RandomForest training failed")
        model_results["random_forest"] = {
            "status": "failed",
            "error": str(exc),
            "duration_seconds": round(time.time() - rf_start, 4),
        }

    # --- Model 4: Multinomial Naive Bayes ---
    nb_start = time.time()
    try:
        nb_model = MultinomialNB(alpha=train_config.nb_alpha)
        nb_model.fit(X_train_tfidf, y_train_clean)
        nb_duration = time.time() - nb_start

        save_artifact(nb_model, MODEL_FILENAMES["multinomial_nb"])
        model_results["multinomial_nb"] = {
            "status": "success",
            "duration_seconds": round(nb_duration, 4),
            "parameters": {
                "alpha": train_config.nb_alpha,
            },
        }
    except Exception as exc:
        logger.exception("MultinomialNB training failed")
        model_results["multinomial_nb"] = {
            "status": "failed",
            "error": str(exc),
            "duration_seconds": round(time.time() - nb_start, 4),
        }

    # Save fitted TF-IDF vectorizer artifact
    save_artifact(vectorizer, VECTORIZER_PATH)

    # 9. Build and save training metadata
    total_duration = time.time() - start_time
    successful_models = [name for name, res in model_results.items() if res["status"] == "success"]
    failed_models = [name for name, res in model_results.items() if res["status"] == "failed"]

    metadata = {
        "dataset_filename": csv_path.name,
        "dataset_size_rows": report.total_rows,
        "training_samples_used": len(X_train_clean),
        "testing_samples_used": len(X_test_clean),
        "review_text_column": text_col,
        "sentiment_label_column": label_col,
        "ratings_are_labels": dataset_config.ratings_are_labels,
        "class_names": classes,
        "class_distribution": class_counts,
        "train_test_split": {
            "test_size": train_config.test_size,
            "random_state": train_config.random_state,
            "stratify": train_config.stratify,
        },
        "tfidf_configuration": {
            "max_features": train_config.max_features,
            "ngram_range": list(train_config.ngram_range),
            "min_df": train_config.min_df,
            "max_df": train_config.max_df,
            "sublinear_tf": train_config.sublinear_tf,
            "vocabulary_size": int(X_train_tfidf.shape[1]),
        },
        "models": model_results,
        "successful_models": successful_models,
        "failed_models": failed_models,
        "training_timestamp_utc": utc_timestamp,
        "training_duration_seconds": round(total_duration, 2),
        "artifacts": {
            "vectorizer": str(VECTORIZER_PATH.name),
            "models": {name: str(path.name) for name, path in MODEL_FILENAMES.items()},
        },
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }

    save_metadata(metadata)

    return {
        "status": "success" if not failed_models else "partial_success",
        "successful_models": successful_models,
        "failed_models": failed_models,
        "training_duration_seconds": round(total_duration, 2),
        "dataset": {
            "filename": csv_path.name,
            "total_rows": report.total_rows,
            "training_rows": len(X_train_clean),
            "testing_rows": len(X_test_clean),
            "classes": classes,
            "class_distribution": class_counts,
        },
        "artifacts_location": str(MODELS_DIR),
        "metadata": metadata,
    }
