"""
ModelTrainer — trains, evaluates, and registers gradient-boosted ensemble
models for the Claim Denial Prediction system.

Design references:
  - Requirements 9.1–9.9
  - Design §Model_Trainer

MLflow is wrapped in a no-op shim (same pattern as dataset_builder.py) so
this module runs without a live MLflow server.  Set
``CLAIM_DENIAL_MLFLOW_ENABLED=1`` + ``MLFLOW_TRACKING_URI`` to enable real
tracking.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from claim_denial.registry.model_registry import ModelArtifact, ModelRegistry
from claim_denial.training.dataset_builder import (
    ClassWeightConfig,
    TrainingDataset,
    TrainingDatasetBuilder,
)
from claim_denial.models import ModelStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLflow shim — identical pattern to dataset_builder
# ---------------------------------------------------------------------------


def _build_mlflow_shim():
    """Return a no-op MLflow shim or the real mlflow module."""
    if os.environ.get("CLAIM_DENIAL_MLFLOW_ENABLED", "0") == "1":
        try:
            import mlflow as _real_mlflow  # type: ignore
            return _real_mlflow
        except ImportError:
            pass

    class _NoOpRun:  # noqa: D101
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class _NoOpMLflow:  # noqa: D101
        @staticmethod
        def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
            pass

        @staticmethod
        def log_metrics(metrics: Dict[str, float]) -> None:
            pass

        @staticmethod
        def log_param(key: str, value: Any) -> None:
            pass

        @staticmethod
        def log_params(params: Dict[str, Any]) -> None:
            pass

        @staticmethod
        def set_tag(key: str, value: str) -> None:
            pass

        @staticmethod
        def set_tags(tags: Dict[str, str]) -> None:
            pass

        @staticmethod
        def start_run(run_name: Optional[str] = None, nested: bool = False):
            return _NoOpRun()

        @staticmethod
        def log_dict(d: dict, artifact_file: str) -> None:
            pass

        @staticmethod
        def sklearn():
            class _NoOpSklearn:
                @staticmethod
                def log_model(model: Any, artifact_path: str) -> None:
                    pass
            return _NoOpSklearn()

    return _NoOpMLflow()


mlflow = _build_mlflow_shim()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRAIN_FRACTION: float = 0.70
_VAL_FRACTION: float = 0.15
_TEST_FRACTION: float = 0.15
_MIN_OPTUNA_TRIALS: int = 50
_ECE_N_BINS: int = 10
_SHAP_TOP_N: int = 20

_SUPPORTED_ALGORITHMS = ("XGBoost", "CatBoost", "LightGBM")
_DEFAULT_ALGORITHM = "XGBoost"

# Precision / ECE thresholds for Staging registration (mirrors ModelRegistry)
_PRECISION_THRESHOLD: float = 0.90
_ECE_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# TrainingResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """
    Result of a completed training run.

    Attributes
    ----------
    model_artifact:
        The registered :class:`~claim_denial.registry.model_registry.ModelArtifact`.
    precision:
        Held-out test-set precision.
    recall:
        Held-out test-set recall.
    f1:
        Held-out test-set F1.
    auc_roc:
        Held-out test-set AUC-ROC.
    auc_pr:
        Held-out test-set AUC-PR (area under precision-recall curve).
    ece:
        Expected Calibration Error (10-bin, equal-width).
    shap_top20:
        Top-20 features by mean absolute SHAP value.  Each element is a
        dict with keys ``"feature"`` and ``"mean_abs_shap"``.
    train_record_count:
        Number of records in the training split.
    validation_record_count:
        Number of records in the validation split.
    test_record_count:
        Number of records in the held-out test split.
    best_hyperparameters:
        Hyperparameter dict selected by Bayesian optimization.
    """

    model_artifact: ModelArtifact
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    ece: float
    shap_top20: List[Dict[str, Any]]
    train_record_count: int
    validation_record_count: int
    test_record_count: int
    best_hyperparameters: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal split helper
# ---------------------------------------------------------------------------


def _temporal_three_way_split(
    dataset: TrainingDataset,
) -> tuple[TrainingDataset, TrainingDataset, TrainingDataset]:
    """
    Split *dataset* into train (70%), validation (15%), and test (15%)
    subsets using a temporal sort on submission_date.

    Boundary rule (same as dataset_builder.py 80/20 rule):
    - All records whose date equals the *first* boundary date go to the
      later (validation) split.
    - All records whose date equals the *second* boundary date go to the
      test split.

    Parameters
    ----------
    dataset:
        Full training dataset.

    Returns
    -------
    tuple[TrainingDataset, TrainingDataset, TrainingDataset]
        ``(train, validation, test)``
    """
    n = len(dataset)
    if n == 0:
        empty = TrainingDataset()
        return empty, empty, empty

    # Sort all indices by submission_date ascending
    indexed = sorted(range(n), key=lambda i: dataset.submission_dates[i])

    # ── First boundary: train/validation ─────────────────────────────────
    target_val_start = math.floor(n * _TRAIN_FRACTION)  # first idx of val
    # Boundary date is the date at target_val_start
    val_boundary_date = dataset.submission_dates[indexed[target_val_start]]
    # Push back to include ALL records on that date in the later split
    first_val_idx = target_val_start
    while (
        first_val_idx > 0
        and dataset.submission_dates[indexed[first_val_idx - 1]] == val_boundary_date
    ):
        first_val_idx -= 1

    # ── Second boundary: validation/test ─────────────────────────────────
    target_test_start = math.floor(n * (_TRAIN_FRACTION + _VAL_FRACTION))
    test_boundary_date = dataset.submission_dates[indexed[target_test_start]]
    first_test_idx = target_test_start
    while (
        first_test_idx > first_val_idx
        and dataset.submission_dates[indexed[first_test_idx - 1]] == test_boundary_date
    ):
        first_test_idx -= 1

    train_idx = [indexed[i] for i in range(first_val_idx)]
    val_idx = [indexed[i] for i in range(first_val_idx, first_test_idx)]
    test_idx = [indexed[i] for i in range(first_test_idx, n)]

    def _subset(indices):
        return TrainingDataset(
            records=[dataset.records[i] for i in indices],
            labels=[dataset.labels[i] for i in indices],
            submission_dates=[dataset.submission_dates[i] for i in indices],
        )

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)


# ---------------------------------------------------------------------------
# Feature-matrix helpers
# ---------------------------------------------------------------------------


def _records_to_matrix(
    records: List[dict],
) -> tuple[np.ndarray, List[str]]:
    """
    Convert a list of feature-store dicts to a 2-D float32 numpy matrix.

    Only numeric (int / float) and boolean fields are included; string
    and list fields are skipped (NLP embeddings, specialty strings, etc.).

    Returns
    -------
    tuple[np.ndarray, List[str]]
        ``(X, feature_names)`` where X has shape (n_samples, n_features).
    """
    if not records:
        return np.empty((0, 0), dtype=np.float32), []

    # Determine feature columns from the first record
    sample = records[0]
    feature_names: List[str] = [
        k
        for k, v in sample.items()
        if isinstance(v, (int, float, bool)) and not isinstance(v, str)
        # Skip the label-like columns that come from adjudication
        and k not in ("adjudication_outcome",)
    ]

    n = len(records)
    m = len(feature_names)
    X = np.zeros((n, m), dtype=np.float32)
    for row_idx, rec in enumerate(records):
        for col_idx, fname in enumerate(feature_names):
            val = rec.get(fname, 0)
            if val is None:
                val = 0
            X[row_idx, col_idx] = float(val)

    return X, feature_names


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model(
    algorithm: str,
    params: Dict[str, Any],
    class_weight_config: ClassWeightConfig,
):
    """
    Instantiate a sklearn-compatible gradient-boosted tree model.

    Falls back to a graceful error if the requested library is not installed.

    Parameters
    ----------
    algorithm:
        One of "XGBoost", "CatBoost", or "LightGBM".
    params:
        Hyperparameter dict for the chosen algorithm.
    class_weight_config:
        Class-weight configuration from TrainingDatasetBuilder.

    Returns
    -------
    sklearn-compatible estimator
    """
    if algorithm == "XGBoost":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as exc:
            raise ImportError("xgboost is required for algorithm='XGBoost'") from exc
        model_params = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "min_child_weight": params.get("min_child_weight", 1),
            "reg_alpha": params.get("reg_alpha", 0.0),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "scale_pos_weight": class_weight_config.scale_pos_weight,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
            "random_state": 42,
        }
        # Remove unsupported param (xgboost >= 2.x dropped use_label_encoder)
        try:
            return XGBClassifier(**model_params)
        except TypeError:
            model_params.pop("use_label_encoder", None)
            return XGBClassifier(**model_params)

    elif algorithm == "CatBoost":
        try:
            from catboost import CatBoostClassifier  # type: ignore
        except ImportError as exc:
            raise ImportError("catboost is required for algorithm='CatBoost'") from exc
        cw = class_weight_config.class_weight
        return CatBoostClassifier(
            iterations=params.get("n_estimators", 200),
            depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            class_weights=[cw.get(0, 1.0), cw.get(1, 1.0)],
            verbose=0,
            random_state=42,
        )

    elif algorithm == "LightGBM":
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            raise ImportError("lightgbm is required for algorithm='LightGBM'") from exc
        cw = class_weight_config.class_weight
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            class_weight=cw,
            verbose=-1,
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Must be one of: {_SUPPORTED_ALGORITHMS}"
        )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1, AUC-ROC, AUC-PR, and ECE.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_pred_proba:
        Predicted probabilities for the positive class.
    threshold:
        Decision threshold for Precision/Recall/F1 computation.

    Returns
    -------
    Dict[str, float]
        Keys: precision, recall, f1, auc_roc, auc_pr, ece
    """
    from sklearn.metrics import (  # type: ignore
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    precision = float(
        precision_score(y_true, y_pred_binary, zero_division=0)
    )
    recall = float(
        recall_score(y_true, y_pred_binary, zero_division=0)
    )
    f1 = float(
        f1_score(y_true, y_pred_binary, zero_division=0)
    )

    # AUC-ROC and AUC-PR require at least two distinct labels
    if len(np.unique(y_true)) < 2:
        auc_roc = 0.0
        auc_pr = 0.0
    else:
        auc_roc = float(roc_auc_score(y_true, y_pred_proba))
        auc_pr = float(average_precision_score(y_true, y_pred_proba))

    ece = _compute_ece(y_true, y_pred_proba)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "ece": ece,
    }


def _compute_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = _ECE_N_BINS,
) -> float:
    """
    Compute Expected Calibration Error using *n_bins* equal-width bins.

    Formula (per design doc):
      ECE = sum_b [ (|bin_b| / N) * |mean_pred_b − frac_pos_b| ]

    Bins are [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    Empty bins contribute 0 to the sum.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred_proba:
        Predicted probabilities in [0, 1].
    n_bins:
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        ECE in [0.0, 1.0].
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for b in range(n_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        # Last bin is inclusive on the right
        if b == n_bins - 1:
            mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)
        else:
            mask = (y_pred_proba >= lo) & (y_pred_proba < hi)

        bin_size = int(np.sum(mask))
        if bin_size == 0:
            continue

        mean_pred = float(np.mean(y_pred_proba[mask]))
        frac_pos = float(np.mean(y_true[mask]))
        ece += (bin_size / n) * abs(mean_pred - frac_pos)

    return float(ece)


# ---------------------------------------------------------------------------
# SHAP helper
# ---------------------------------------------------------------------------


def _compute_shap_top20(
    model: Any,
    X_test: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Compute SHAP values on *X_test* and return the top-20 features by mean
    absolute SHAP value.

    Falls back gracefully if SHAP is not installed or if the model does not
    support SHAP tree-explainer.

    Parameters
    ----------
    model:
        Trained sklearn-compatible model.
    X_test:
        Test feature matrix, shape (n_samples, n_features).
    feature_names:
        List of feature column names aligned with X_test columns.

    Returns
    -------
    List[Dict[str, Any]]
        Up to 20 dicts, each with keys ``"feature"`` (str) and
        ``"mean_abs_shap"`` (float), sorted descending.
    """
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification shap may return a list [neg, pos]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_matrix = shap_values[1]
        else:
            shap_matrix = np.array(shap_values)

        # Ensure 2D
        if shap_matrix.ndim == 1:
            shap_matrix = shap_matrix.reshape(1, -1)

        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        n_features = min(len(feature_names), len(mean_abs))

        ranked = sorted(
            range(n_features),
            key=lambda i: float(mean_abs[i]),
            reverse=True,
        )[:_SHAP_TOP_N]

        return [
            {
                "feature": feature_names[i],
                "mean_abs_shap": float(mean_abs[i]),
            }
            for i in ranked
        ]

    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP computation failed: %s. Returning empty list.", exc)
        return []


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------


class ModelTrainer:
    """
    Trains, evaluates, and registers a gradient-boosted ensemble model for
    claim denial prediction.

    Requirements: 9.1–9.9

    Parameters
    ----------
    algorithm:
        One of "XGBoost" (default), "CatBoost", or "LightGBM".
    n_trials:
        Number of Bayesian optimization trials (minimum 50).
    model_registry:
        :class:`~claim_denial.registry.model_registry.ModelRegistry` instance.
        A new in-memory registry is created if not provided.
    experiment_name:
        MLflow experiment name used for run logging.
    """

    def __init__(
        self,
        algorithm: str = _DEFAULT_ALGORITHM,
        n_trials: int = _MIN_OPTUNA_TRIALS,
        model_registry: Optional[ModelRegistry] = None,
        experiment_name: str = "claim-denial-prediction",
    ) -> None:
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {_SUPPORTED_ALGORITHMS}, got '{algorithm}'"
            )
        if n_trials < _MIN_OPTUNA_TRIALS:
            raise ValueError(
                f"n_trials must be ≥ {_MIN_OPTUNA_TRIALS}, got {n_trials}"
            )
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.registry = model_registry if model_registry is not None else ModelRegistry()
        self.experiment_name = experiment_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_training(
        self,
        training_dataset: TrainingDataset,
        trigger_reason: Optional[str] = None,
    ) -> TrainingResult:
        """
        Execute the full training pipeline.

        Steps
        -----
        1. Temporal 70/15/15 split.
        2. Class-weight detection from the training split.
        3. Bayesian hyperparameter optimization (≥50 trials) on validation
           precision via Optuna.
        4. Final model training on the train split with best hyperparameters.
        5. Evaluation on the held-out test split (Precision, Recall, F1,
           AUC-ROC, AUC-PR, ECE).
        6. SHAP feature importance (top-20).
        7. MLflow logging of all hyperparameters, metrics, and model artifact.
        8. Model registration via ModelRegistry (Staging or Failed).

        Parameters
        ----------
        training_dataset:
            Labeled dataset produced by
            :class:`~claim_denial.training.dataset_builder.TrainingDatasetBuilder`.
        trigger_reason:
            When not ``None``, logged as an MLflow tag/param along with
            ``rolling_precision_at_trigger`` and ``run_start_timestamp``.

        Returns
        -------
        TrainingResult

        Raises
        ------
        ValueError
            If training_dataset is empty.
        """
        run_start = datetime.now(tz=timezone.utc)

        if len(training_dataset) == 0:
            raise ValueError("training_dataset is empty; cannot train.")

        logger.info(
            "ModelTrainer.run_training started | algorithm=%s | n_trials=%d | "
            "dataset_size=%d",
            self.algorithm,
            self.n_trials,
            len(training_dataset),
        )

        # ── 1. Temporal 70/15/15 split ────────────────────────────────────
        train_ds, val_ds, test_ds = _temporal_three_way_split(training_dataset)
        logger.info(
            "Temporal 70/15/15 split: train=%d | val=%d | test=%d",
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )

        # ── 2. Feature matrices ───────────────────────────────────────────
        X_train, feature_names = _records_to_matrix(train_ds.records)
        y_train = np.array(train_ds.labels, dtype=np.int32)

        X_val, _ = _records_to_matrix(val_ds.records)
        y_val = np.array(val_ds.labels, dtype=np.int32)

        X_test, _ = _records_to_matrix(test_ds.records)
        y_test = np.array(test_ds.labels, dtype=np.int32)

        # ── 3. Class weights ──────────────────────────────────────────────
        builder = TrainingDatasetBuilder()
        class_weight_config = builder.check_class_imbalance(train_ds)

        # ── 4. Bayesian hyperparameter optimisation (Optuna) ──────────────
        best_params = self._optimize_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            class_weight_config=class_weight_config,
        )

        # ── 5. Final model training ───────────────────────────────────────
        logger.info("Training final model with best hyperparameters: %s", best_params)
        final_model = _build_model(self.algorithm, best_params, class_weight_config)

        try:
            # Try Spark-distributed training first (XGBoost SparkXGBClassifier)
            final_model = self._try_spark_train(
                final_model, X_train, y_train, best_params, class_weight_config
            )
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "Spark distributed training not available (%s). "
                "Falling back to local sklearn-compatible training.",
                exc,
            )
            final_model.fit(X_train, y_train)

        # ── 6. Evaluation on held-out test set ────────────────────────────
        y_pred_proba = self._predict_proba(final_model, X_test)
        metrics = _compute_metrics(y_test, y_pred_proba)

        logger.info(
            "Test-set metrics | precision=%.4f | recall=%.4f | f1=%.4f | "
            "auc_roc=%.4f | auc_pr=%.4f | ece=%.4f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["auc_roc"],
            metrics["auc_pr"],
            metrics["ece"],
        )

        # ── 7. SHAP feature importance ────────────────────────────────────
        shap_top20 = _compute_shap_top20(final_model, X_test, feature_names)

        # ── 8. Algorithm version ──────────────────────────────────────────
        algo_version = self._get_algorithm_version()

        # ── 9. MLflow logging + model registration ────────────────────────
        version_id = f"v-{uuid.uuid4().hex[:8]}"

        try:
            with mlflow.start_run(run_name=f"training-{version_id}"):
                self._log_to_mlflow(
                    version_id=version_id,
                    best_params=best_params,
                    metrics=metrics,
                    shap_top20=shap_top20,
                    model=final_model,
                    trigger_reason=trigger_reason,
                    run_start=run_start,
                    train_count=len(train_ds),
                    val_count=len(val_ds),
                    test_count=len(test_ds),
                    class_weight_config=class_weight_config,
                    training_dataset=training_dataset,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging failed: %s", exc)

        # ── 10. Build and register ModelArtifact ──────────────────────────
        denied_count = sum(1 for lbl in train_ds.labels if lbl == 1)
        paid_count = len(train_ds) - denied_count
        denied_pct = denied_count / len(train_ds) if len(train_ds) > 0 else 0.0

        artifact = ModelArtifact(
            version_id=version_id,
            stage=ModelStage.FAILED,          # will be overridden by registry
            training_date=run_start.isoformat(),
            algorithm=self.algorithm,
            algorithm_version=algo_version,
            hyperparameters=best_params,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            auc_roc=metrics["auc_roc"],
            auc_pr=metrics["auc_pr"],
            ece=metrics["ece"],
            shap_top20=shap_top20,
            train_record_count=len(train_ds),
            test_record_count=len(test_ds),
            denied_class_pct=denied_pct,
            trigger_reason=trigger_reason,
            model_object=final_model,
        )

        registered = self.registry.register_model(artifact)

        logger.info(
            "Model %s registered as %s.",
            registered.version_id,
            registered.stage.value,
        )

        return TrainingResult(
            model_artifact=registered,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            auc_roc=metrics["auc_roc"],
            auc_pr=metrics["auc_pr"],
            ece=metrics["ece"],
            shap_top20=shap_top20,
            train_record_count=len(train_ds),
            validation_record_count=len(val_ds),
            test_record_count=len(test_ds),
            best_hyperparameters=best_params,
        )

    # ------------------------------------------------------------------
    # Bayesian hyperparameter optimisation
    # ------------------------------------------------------------------

    def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight_config: ClassWeightConfig,
    ) -> Dict[str, Any]:
        """
        Run Optuna Bayesian search to maximise Precision on the validation set.

        Each trial is logged as a nested MLflow run.

        Parameters
        ----------
        X_train, y_train:
            Training feature matrix and labels.
        X_val, y_val:
            Validation feature matrix and labels.
        class_weight_config:
            Class-weight parameters.

        Returns
        -------
        Dict[str, Any]
            Best hyperparameter configuration found.
        """
        try:
            import optuna  # type: ignore
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("optuna is required for hyperparameter optimization") from exc

        from sklearn.metrics import precision_score  # type: ignore

        algorithm = self.algorithm

        def objective(trial: "optuna.Trial") -> float:  # type: ignore
            params = self._suggest_params(trial)

            try:
                model = _build_model(algorithm, params, class_weight_config)
                model.fit(X_train, y_train)
                y_pred_proba = self._predict_proba(model, X_val)
                y_pred_binary = (y_pred_proba >= 0.5).astype(int)
                prec = float(precision_score(y_val, y_pred_binary, zero_division=0))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Optuna trial %d failed: %s", trial.number, exc)
                return 0.0

            # Log each trial to MLflow as a nested run
            try:
                with mlflow.start_run(
                    run_name=f"trial-{trial.number}", nested=True
                ):
                    mlflow.log_params({f"trial_{k}": v for k, v in params.items()})
                    mlflow.log_metric("trial_val_precision", prec)
            except Exception:  # noqa: BLE001
                pass

            return prec

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = self._suggest_params_from_trial(study.best_trial)
        logger.info(
            "Optuna best trial: precision=%.4f | params=%s",
            study.best_value,
            best_params,
        )
        return best_params

    def _suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the chosen algorithm using the Optuna trial
        object.

        Parameters
        ----------
        trial:
            An ``optuna.Trial`` object.

        Returns
        -------
        Dict[str, Any]
            Suggested hyperparameter dict.
        """
        if self.algorithm == "XGBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        elif self.algorithm == "CatBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
        else:  # LightGBM
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }

    def _suggest_params_from_trial(self, trial: Any) -> Dict[str, Any]:
        """
        Reconstruct the best hyperparameter dict from a completed Optuna trial.

        Parameters
        ----------
        trial:
            A completed ``optuna.FrozenTrial`` (from ``study.best_trial``).

        Returns
        -------
        Dict[str, Any]
            Best hyperparameter dict with proper types.
        """
        p = trial.params

        if self.algorithm == "XGBoost":
            return {
                "n_estimators": int(p.get("n_estimators", 200)),
                "max_depth": int(p.get("max_depth", 6)),
                "learning_rate": float(p.get("learning_rate", 0.05)),
                "subsample": float(p.get("subsample", 0.8)),
                "colsample_bytree": float(p.get("colsample_bytree", 0.8)),
                "min_child_weight": int(p.get("min_child_weight", 1)),
                "reg_alpha": float(p.get("reg_alpha", 0.0)),
                "reg_lambda": float(p.get("reg_lambda", 1.0)),
            }
        elif self.algorithm == "CatBoost":
            return {
                "n_estimators": int(p.get("n_estimators", 200)),
                "max_depth": int(p.get("max_depth", 6)),
                "learning_rate": float(p.get("learning_rate", 0.05)),
                "subsample": float(p.get("subsample", 0.8)),
            }
        else:  # LightGBM
            return {
                "n_estimators": int(p.get("n_estimators", 200)),
                "max_depth": int(p.get("max_depth", 6)),
                "learning_rate": float(p.get("learning_rate", 0.05)),
                "subsample": float(p.get("subsample", 0.8)),
                "colsample_bytree": float(p.get("colsample_bytree", 0.8)),
            }

    # ------------------------------------------------------------------
    # Spark distribution (best-effort)
    # ------------------------------------------------------------------

    def _try_spark_train(
        self,
        sklearn_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Dict[str, Any],
        class_weight_config: ClassWeightConfig,
    ) -> Any:
        """
        Attempt Spark-distributed training.

        For XGBoost this uses ``xgboost.spark.SparkXGBClassifier`` when
        available.  For CatBoost / LightGBM falls back immediately to the
        sklearn-compatible API (those libraries provide their own
        multi-threaded training).

        Raises an exception if Spark is unavailable so the caller can
        fall back to local training.
        """
        if self.algorithm == "XGBoost":
            from xgboost.spark import SparkXGBClassifier  # type: ignore  # may raise
            from pyspark.sql import SparkSession  # type: ignore

            spark = SparkSession.builder.appName("claim-denial-xgb-training").getOrCreate()

            import pandas as pd
            df_pd = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
            df_pd["label"] = y_train.tolist()
            df_spark = spark.createDataFrame(df_pd)

            feature_cols = [c for c in df_pd.columns if c != "label"]
            from pyspark.ml.feature import VectorAssembler  # type: ignore
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            df_assembled = assembler.transform(df_spark)

            spark_model = SparkXGBClassifier(
                features_col="features",
                label_col="label",
                num_workers=2,
                **{k: v for k, v in params.items()
                   if k in ("n_estimators", "max_depth", "learning_rate",
                             "subsample", "colsample_bytree")},
            )
            spark_model.fit(df_assembled)
            # Return the underlying sklearn model extracted from the Spark wrapper
            return spark_model.get_booster() if hasattr(spark_model, "get_booster") \
                else sklearn_model

        # For CatBoost / LightGBM, fall back immediately
        raise NotImplementedError(
            f"Spark training not implemented for {self.algorithm}; "
            "falling back to sklearn interface."
        )

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
        """
        Return positive-class predicted probabilities.

        Handles models that return a 1D array (single probability) or a
        2D array (neg/pos columns).

        Parameters
        ----------
        model:
            Trained sklearn-compatible estimator.
        X:
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            1-D array of positive-class probabilities, shape (n_samples,).
        """
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1].astype(np.float64)
        return proba.astype(np.float64)

    # ------------------------------------------------------------------
    # MLflow logging helper
    # ------------------------------------------------------------------

    def _log_to_mlflow(
        self,
        version_id: str,
        best_params: Dict[str, Any],
        metrics: Dict[str, float],
        shap_top20: List[Dict[str, Any]],
        model: Any,
        trigger_reason: Optional[str],
        run_start: datetime,
        train_count: int,
        val_count: int,
        test_count: int,
        class_weight_config: ClassWeightConfig,
        training_dataset: TrainingDataset,
    ) -> None:
        """Log all hyperparameters, metrics, and artifacts to MLflow."""
        # Params
        try:
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("algorithm_version", self._get_algorithm_version())
            mlflow.log_param("version_id", version_id)
            mlflow.log_param("train_record_count", train_count)
            mlflow.log_param("validation_record_count", val_count)
            mlflow.log_param("test_record_count", test_count)
            mlflow.log_param("n_optuna_trials", self.n_trials)

            # Best hyperparameters
            for k, v in best_params.items():
                mlflow.log_param(f"best_{k}", v)

            # Class weight info
            mlflow.log_param("scale_pos_weight", class_weight_config.scale_pos_weight)

            # Metrics
            mlflow.log_metrics(metrics)

            # SHAP top-20 as artifact dict
            mlflow.log_dict(
                {"shap_top20": shap_top20},
                "shap_top20_features.json",
            )

            # Model artifact
            try:
                mlflow.sklearn().log_model(model, artifact_path="model")
            except Exception:  # noqa: BLE001
                pass

            # Trigger-reason tags (Requirement 9 – trigger logging)
            if trigger_reason is not None:
                mlflow.set_tag("trigger_reason", trigger_reason)
                mlflow.log_param("trigger_reason", trigger_reason)
                mlflow.log_param("run_start_timestamp", run_start.isoformat())

                # Compute rolling precision at trigger from dataset labels
                all_labels = training_dataset.labels
                n_denied = sum(1 for lbl in all_labels if lbl == 1)
                rolling_precision = (
                    n_denied / len(all_labels) if len(all_labels) > 0 else 0.0
                )
                mlflow.log_param(
                    "rolling_precision_at_trigger", rolling_precision
                )

        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow param/metric logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Algorithm version helper
    # ------------------------------------------------------------------

    def _get_algorithm_version(self) -> str:
        """Return the installed library version string for the chosen algorithm."""
        try:
            if self.algorithm == "XGBoost":
                import xgboost  # type: ignore
                return xgboost.__version__
            elif self.algorithm == "CatBoost":
                import catboost  # type: ignore
                return catboost.__version__
            else:
                import lightgbm  # type: ignore
                return lightgbm.__version__
        except Exception:  # noqa: BLE001
            return "unknown"
