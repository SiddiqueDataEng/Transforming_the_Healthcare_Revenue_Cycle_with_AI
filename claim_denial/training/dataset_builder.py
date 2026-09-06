"""
TrainingDatasetBuilder — constructs labeled training datasets for the
Claim Denial Prediction model.

Design references: Requirements 8.1–8.6, Design §Training Dataset Construction.

MLflow is patched with no-op implementations so this module can run
without a live MLflow server.  Set the environment variable
``CLAIM_DENIAL_MLFLOW_ENABLED=1`` (and configure ``MLFLOW_TRACKING_URI``)
to enable real MLflow tracking.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# MLflow shim — no-op unless real tracking is explicitly enabled
# ---------------------------------------------------------------------------

def _build_mlflow_shim():
    """Return a no-op MLflow shim or the real mlflow module."""
    if os.environ.get("CLAIM_DENIAL_MLFLOW_ENABLED", "0") == "1":
        try:
            import mlflow as _real_mlflow  # type: ignore
            return _real_mlflow
        except ImportError:
            pass

    # No-op shim — satisfies the interface without a server
    class _NoOpMLflow:  # noqa: D101  (no public docstring)
        @staticmethod
        def log_metric(key: str, value: float, step: Optional[int] = None) -> None:  # noqa: D102
            pass

        @staticmethod
        def log_param(key: str, value) -> None:  # noqa: D102
            pass

        @staticmethod
        def set_tag(key: str, value: str) -> None:  # noqa: D102
            pass

    return _NoOpMLflow()


mlflow = _build_mlflow_shim()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_INITIAL_DAYS: int = 90
_WINDOW_EXPANDED_DAYS: int = 180
_MIN_LABELED_RECORDS: int = 1_000
_TEST_FRACTION: float = 0.20
_IMBALANCE_THRESHOLD: float = 0.05   # 5 %

_OUTCOME_DENIED: str = "denied"
_OUTCOME_PAID: str = "paid"

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingDataset:
    """
    A labeled dataset ready for model training or evaluation.

    Attributes
    ----------
    records:
        Feature dicts for each claim.  Each record is a plain ``dict``
        mirroring the Feature Store "Claim Point of View" schema.
    labels:
        Parallel list of binary labels: 1 = denied, 0 = paid.
    submission_dates:
        Parallel list of claim submission dates (used for temporal
        splitting and MLflow logging).
    """

    records: List[dict] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    submission_dates: List[date] = field(default_factory=list)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.labels)

    def __post_init__(self) -> None:
        if not (len(self.records) == len(self.labels) == len(self.submission_dates)):
            raise ValueError(
                "records, labels, and submission_dates must all have the same length; "
                f"got {len(self.records)}, {len(self.labels)}, {len(self.submission_dates)}"
            )


@dataclass
class ClassWeightConfig:
    """
    Class-weight parameters for handling denied-class imbalance.

    These are passed directly to the gradient-boosted tree trainer.

    Attributes
    ----------
    scale_pos_weight:
        XGBoost ``scale_pos_weight`` parameter (ratio of negative to
        positive examples, i.e. paid_count / denied_count).
    class_weight:
        CatBoost / LightGBM ``class_weight`` dict mapping label → weight.
        ``{0: 1.0, 1: scale_pos_weight}``
    """

    scale_pos_weight: float = 1.0
    class_weight: Dict[int, float] = field(default_factory=lambda: {0: 1.0, 1: 1.0})


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class InsufficientDataError(RuntimeError):
    """
    Raised when the labeled dataset contains fewer than 1,000 records
    after exclusions, as required by Requirement 8.6.
    """


# ---------------------------------------------------------------------------
# TrainingDatasetBuilder
# ---------------------------------------------------------------------------


class TrainingDatasetBuilder:
    """
    Constructs a labeled :class:`TrainingDataset` from Feature Store records
    and provides helpers for temporal splitting and class-imbalance detection.

    Usage::

        builder = TrainingDatasetBuilder()

        # Build from feature-store records
        dataset = builder.build(
            construction_date=date.today(),
            feature_store_records=records,
        )

        # Temporal 80/20 split
        train_ds, test_ds = builder.temporal_split(dataset)

        # Class-weight config for the training set
        weights = builder.check_class_imbalance(train_ds)
    """

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(
        self,
        construction_date: date,
        feature_store_records: List[dict],
    ) -> TrainingDataset:
        """
        Construct a :class:`TrainingDataset` from raw Feature Store records.

        The method follows the window-expansion logic in Requirement 8.1:

        1. Filter records whose ``submission_date`` falls within the 90-day
           window ending on *construction_date*.
        2. If the filtered window yields fewer than 1,000 labeled records,
           expand to 180 days and re-filter.
        3. Label records (Requirement 8.2):
           - ``adjudication_outcome == "denied"`` → label 1
           - ``adjudication_outcome == "paid"`` → label 0
           - All other outcomes are excluded; the excluded count is logged
             to MLflow.
        4. If labeled records < 1,000 after exclusions, halt and raise
           :class:`InsufficientDataError` (Requirement 8.6).

        Parameters
        ----------
        construction_date:
            The anchor date for the training window.  Typically today.
        feature_store_records:
            Raw Feature Store rows.  Each dict must contain at minimum:

            - ``claim_id`` (str)
            - ``submission_date`` (``date`` or ISO-8601 ``str``)
            - ``adjudication_outcome`` (str: ``"denied"`` / ``"paid"`` / other)

        Returns
        -------
        TrainingDataset

        Raises
        ------
        InsufficientDataError
            When fewer than 1,000 labeled records remain after exclusions,
            even after expanding the window to 180 days.
        """
        # ── Step 1: 90-day window ─────────────────────────────────────────
        window_days = _WINDOW_INITIAL_DAYS
        window_records = self._filter_window(
            feature_store_records, construction_date, window_days
        )
        labeled_in_window = self._count_labeled(window_records)

        # ── Step 2: expand to 180 days if needed (Requirement 8.1) ───────
        if labeled_in_window < _MIN_LABELED_RECORDS:
            logger.info(
                "90-day window yielded %d labeled records (< %d); "
                "expanding to %d-day window.",
                labeled_in_window,
                _MIN_LABELED_RECORDS,
                _WINDOW_EXPANDED_DAYS,
            )
            window_days = _WINDOW_EXPANDED_DAYS
            window_records = self._filter_window(
                feature_store_records, construction_date, window_days
            )

        # ── Step 3: label & exclude (Requirement 8.2) ─────────────────────
        records: List[dict] = []
        labels: List[int] = []
        dates: List[date] = []
        excluded_count: int = 0

        for rec in window_records:
            outcome = rec.get("adjudication_outcome", "")
            if outcome == _OUTCOME_DENIED:
                label = 1
            elif outcome == _OUTCOME_PAID:
                label = 0
            else:
                excluded_count += 1
                continue

            records.append(rec)
            labels.append(label)
            dates.append(self._parse_date(rec["submission_date"]))

        # Log excluded count to MLflow (Requirement 8.2)
        mlflow.log_metric("excluded_claim_count", excluded_count)
        mlflow.log_param("training_window_days", window_days)

        if excluded_count:
            logger.info(
                "Excluded %d claims with non-paid/non-denied adjudication status.",
                excluded_count,
            )

        # ── Step 4: halt if insufficient data (Requirement 8.6) ──────────
        total_labeled = len(labels)
        if total_labeled < _MIN_LABELED_RECORDS:
            msg = (
                f"Insufficient labeled data: {total_labeled} records found "
                f"(minimum required: {_MIN_LABELED_RECORDS}). "
                f"Training window was {window_days} days."
            )
            logger.error(msg)
            mlflow.log_param("dataset_construction_error", msg)
            mlflow.log_metric("labeled_record_count", total_labeled)
            mlflow.log_metric("min_required_record_count", _MIN_LABELED_RECORDS)
            raise InsufficientDataError(msg)

        mlflow.log_metric("labeled_record_count", total_labeled)
        logger.info("Built training dataset with %d labeled records.", total_labeled)

        return TrainingDataset(
            records=records,
            labels=labels,
            submission_dates=dates,
        )

    # ------------------------------------------------------------------
    # temporal_split
    # ------------------------------------------------------------------

    def temporal_split(
        self,
        dataset: TrainingDataset,
    ) -> Tuple[TrainingDataset, TrainingDataset]:
        """
        Split *dataset* into training (80%) and test (20%) subsets using a
        temporal boundary on submission date.

        Rules (Requirement 8.3):
        - The test set contains the most recent 20% of claims sorted by
          submission date.
        - When claims share the boundary date and straddle the 80/20
          boundary, **all** claims on that boundary date are assigned to
          the test set.

        After splitting, logs to MLflow (Requirement 8.4):
        - ``total_record_count``
        - ``denied_count``
        - ``paid_count``
        - ``denied_pct``
        - ``cutoff_date``

        Parameters
        ----------
        dataset:
            The full labeled dataset produced by :meth:`build`.

        Returns
        -------
        Tuple[TrainingDataset, TrainingDataset]
            ``(train_dataset, test_dataset)``
        """
        n = len(dataset)
        if n == 0:
            return TrainingDataset(), TrainingDataset()

        # ── Sort indices by submission_date ascending ─────────────────────
        indexed = sorted(
            range(n),
            key=lambda i: dataset.submission_dates[i],
        )

        # ── Identify the cutoff index (80 / 20 boundary) ─────────────────
        # The test set should be the most recent 20%.
        # We round *up* so the test set is never smaller than 20 %.
        target_test_size = math.ceil(n * _TEST_FRACTION)
        # The first index (in the sorted order) that belongs to the test set
        boundary_idx_in_sorted = n - target_test_size  # 0-based

        # The boundary date is the submission_date of the first record
        # assigned to the test set (before tie-expansion).
        boundary_date = dataset.submission_dates[indexed[boundary_idx_in_sorted]]

        # Expand boundary to include ALL records on boundary_date in test set
        # (Requirement 8.3: "all claims on boundary date go to the test set").
        # Walk backward until we find the first record with a date < boundary_date.
        first_test_idx_in_sorted = boundary_idx_in_sorted
        while (
            first_test_idx_in_sorted > 0
            and dataset.submission_dates[indexed[first_test_idx_in_sorted - 1]]
            == boundary_date
        ):
            first_test_idx_in_sorted -= 1

        train_indices = [indexed[i] for i in range(first_test_idx_in_sorted)]
        test_indices = [indexed[i] for i in range(first_test_idx_in_sorted, n)]

        train_ds = TrainingDataset(
            records=[dataset.records[i] for i in train_indices],
            labels=[dataset.labels[i] for i in train_indices],
            submission_dates=[dataset.submission_dates[i] for i in train_indices],
        )
        test_ds = TrainingDataset(
            records=[dataset.records[i] for i in test_indices],
            labels=[dataset.labels[i] for i in test_indices],
            submission_dates=[dataset.submission_dates[i] for i in test_indices],
        )

        # ── MLflow logging (Requirement 8.4) ──────────────────────────────
        total_count = n
        denied_count = sum(1 for lbl in dataset.labels if lbl == 1)
        paid_count = sum(1 for lbl in dataset.labels if lbl == 0)
        denied_pct = denied_count / total_count if total_count > 0 else 0.0

        mlflow.log_metric("split_total_record_count", total_count)
        mlflow.log_metric("split_denied_count", denied_count)
        mlflow.log_metric("split_paid_count", paid_count)
        mlflow.log_metric("split_denied_pct", denied_pct)
        mlflow.log_param("split_cutoff_date", boundary_date.isoformat())

        logger.info(
            "Temporal split: %d train / %d test | "
            "denied=%d (%.1f%%) | paid=%d | cutoff=%s",
            len(train_ds),
            len(test_ds),
            denied_count,
            denied_pct * 100,
            paid_count,
            boundary_date.isoformat(),
        )

        return train_ds, test_ds

    # ------------------------------------------------------------------
    # check_class_imbalance
    # ------------------------------------------------------------------

    def check_class_imbalance(
        self,
        train_dataset: TrainingDataset,
    ) -> ClassWeightConfig:
        """
        Detect whether the denied class is below 5 % in *train_dataset*
        and compute class-weight balancing parameters (Requirement 8.5).

        The method **never modifies** the record count; it only returns
        weight parameters to be passed to the ML trainer.

        Parameters
        ----------
        train_dataset:
            The training split produced by :meth:`temporal_split`.

        Returns
        -------
        ClassWeightConfig
            If the denied class is ≥ 5 %, returns balanced weights of
            ``scale_pos_weight=1.0`` and ``class_weight={0: 1.0, 1: 1.0}``.

            If the denied class is < 5 %, returns:

            - ``scale_pos_weight = paid_count / denied_count`` (XGBoost param)
            - ``class_weight = {0: 1.0, 1: paid_count / denied_count}``
              (CatBoost / LightGBM param)
        """
        n = len(train_dataset)
        if n == 0:
            return ClassWeightConfig()

        denied_count = sum(1 for lbl in train_dataset.labels if lbl == 1)
        paid_count = n - denied_count
        denied_pct = denied_count / n if n > 0 else 0.0

        if denied_pct < _IMBALANCE_THRESHOLD and denied_count > 0:
            # Compute scale_pos_weight = #negative / #positive (XGBoost convention)
            scale_pos_weight = paid_count / denied_count
            class_weight = {0: 1.0, 1: scale_pos_weight}
            logger.info(
                "Class imbalance detected: denied=%.2f%% "
                "(threshold %.0f%%). scale_pos_weight=%.4f",
                denied_pct * 100,
                _IMBALANCE_THRESHOLD * 100,
                scale_pos_weight,
            )
        elif denied_count == 0:
            # Edge case: no denied records at all — weight 1 for both classes
            scale_pos_weight = 1.0
            class_weight = {0: 1.0, 1: 1.0}
            logger.warning(
                "Training set contains zero denied records; "
                "class weights set to 1.0."
            )
        else:
            # No imbalance — equal weights
            scale_pos_weight = 1.0
            class_weight = {0: 1.0, 1: 1.0}

        return ClassWeightConfig(
            scale_pos_weight=scale_pos_weight,
            class_weight=class_weight,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value) -> date:
        """Coerce a ``date`` object or ISO-8601 string to ``date``."""
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    @staticmethod
    def _filter_window(
        records: List[dict],
        construction_date: date,
        window_days: int,
    ) -> List[dict]:
        """
        Return records whose ``submission_date`` falls within
        [construction_date − window_days, construction_date] (inclusive).
        """
        start = construction_date - timedelta(days=window_days)
        result: List[dict] = []
        for rec in records:
            try:
                sub_date = TrainingDatasetBuilder._parse_date(rec["submission_date"])
            except (KeyError, ValueError, TypeError):
                continue
            if start <= sub_date <= construction_date:
                result.append(rec)
        return result

    @staticmethod
    def _count_labeled(records: List[dict]) -> int:
        """Count records whose outcome is "denied" or "paid"."""
        return sum(
            1 for r in records
            if r.get("adjudication_outcome") in (_OUTCOME_DENIED, _OUTCOME_PAID)
        )
