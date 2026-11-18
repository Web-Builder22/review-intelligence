"""
Training configuration settings and hyperparameter defaults for Phase C.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Hyperparameters and settings for the Phase C training pipeline."""

    test_size: float = float(os.getenv("TRAIN_TEST_SIZE", "0.20"))
    random_state: int = int(os.getenv("TRAIN_RANDOM_STATE", "42"))
    stratify: bool = os.getenv("TRAIN_STRATIFY", "true").lower() == "true"

    # TF-IDF Vectorizer settings
    max_features: int = int(os.getenv("TFIDF_MAX_FEATURES", "10000"))
    ngram_range: tuple[int, int] = (
        int(os.getenv("TFIDF_NGRAM_MIN", "1")),
        int(os.getenv("TFIDF_NGRAM_MAX", "2")),
    )
    min_df: int = int(os.getenv("TFIDF_MIN_DF", "2"))
    max_df: float = float(os.getenv("TFIDF_MAX_DF", "0.95"))
    sublinear_tf: bool = os.getenv("TFIDF_SUBLINEAR_TF", "true").lower() == "true"

    # Model 1: Linear SVM (with CalibratedClassifierCV for probability estimates)
    svm_c: float = float(os.getenv("SVM_C", "1.0"))
    svm_class_weight: str | None = os.getenv("SVM_CLASS_WEIGHT", "balanced") or None

    # Model 2: Logistic Regression
    lr_max_iter: int = int(os.getenv("LR_MAX_ITER", "2000"))
    lr_class_weight: str | None = os.getenv("LR_CLASS_WEIGHT", "balanced") or None

    # Model 3: Random Forest
    rf_n_estimators: int = int(os.getenv("RF_N_ESTIMATORS", "200"))
    rf_max_depth: int | None = (
        int(os.getenv("RF_MAX_DEPTH")) if os.getenv("RF_MAX_DEPTH") else None
    )
    rf_class_weight: str | None = os.getenv("RF_CLASS_WEIGHT", "balanced") or None

    # Model 4: Multinomial Naive Bayes
    nb_alpha: float = float(os.getenv("NB_ALPHA", "1.0"))

    def to_dict(self) -> dict[str, object]:
        return {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
            "tfidf": {
                "max_features": self.max_features,
                "ngram_range": list(self.ngram_range),
                "min_df": self.min_df,
                "max_df": self.max_df,
                "sublinear_tf": self.sublinear_tf,
            },
            "models": {
                "linear_svc": {
                    "C": self.svm_c,
                    "class_weight": self.svm_class_weight,
                    "calibrated": True,
                },
                "logistic_regression": {
                    "max_iter": self.lr_max_iter,
                    "class_weight": self.lr_class_weight,
                },
                "random_forest": {
                    "n_estimators": self.rf_n_estimators,
                    "max_depth": self.rf_max_depth,
                    "class_weight": self.rf_class_weight,
                },
                "multinomial_nb": {
                    "alpha": self.nb_alpha,
                },
            },
        }


def get_training_config() -> TrainingConfig:
    return TrainingConfig()
