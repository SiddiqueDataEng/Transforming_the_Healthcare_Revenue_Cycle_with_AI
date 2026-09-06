"""
Nightly Batch Scoring Pipeline — BatchScorer

Applies the current Production model to all Active claims in the Feature Store
and writes denial scores back, as part of the Airflow-orchestrated nightly
batch pipeline.

Design references
-----------------
- Batch_Scorer component (design.md §Components and Interfaces)
- Requirements: 10.6, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7

Public API
----------
``BatchScorer.load_production_model()``
    Load the Production model artifact; halt on NoProductionModelError.

``BatchScorer.fetch_active_claims()``
    Query the Feature Store for all claims with ``claim_status = "Active"``.

``BatchScorer.score_claims(model, claims)``
    Distributed inference via Spark (or pure-Python fallback).

``BatchScorer.write_scores(results)``
    Upsert scored results to Feature Store; retry ×3 with exponential back-off.

``BatchScorer.emit_completion_event(stats)``
    Emit structured pipeline-completion event.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from claim_denial.models import ClaimFeatureRecord
from claim_denial.registry.model_registry import ModelArtifact, NoProductionModelError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of write retries before giving up on a single result.
WRITE_MAX_RETRIES: int = 3

#: Base back-off delay in seconds; doubles on each retry (1s, 2s, 4s).
WRITE_BACKOFF_BASE_SECONDS: float = 1.0

#: Score quality flag value used when null features are present after imputation.
SCORE_QUALITY_DEGRADED: str = "degraded"

#: Numeric feature fields on ClaimFeatureRecord that are checked for nullness.
NUMERIC_FEATURE_FIELDS: tuple[str, ...] = (
    "claim_submission_date_doy",
    "claim_submission_date_month",
    "patient_age",
    "gender",
    "ins_type_medicare",
    "ins_type_medicaid",
    "ins_type_commercial",
    "ins_type_tricare",
    "ins_type_champva",
    "ins_type_other",
    "claim_type_professional",
    "claim_type_institutional",
    "adm_type_emergency",
    "adm_type_elective",
    "adm_type_urgent",
    "adm_type_trauma",
    "adm_src_physician_referral",
    "adm_src_transfer_hospital",
    "adm_src_transfer_snf",
    "adm_src_er",
    "adm_src_court_law",
    "adm_src_not_available",
    "billing_provider_id_encoded",
    "provider_historical_denial_rate",
    "provider_claim_volume",
    "facility_bed_size",
    "principal_dx_ccs_category",
    "secondary_dx_count",
    "length_of_stay",
    "comorbidity_risk_score",
    "diagnosis_procedure_mismatch",
    "payer_historical_denial_rate",
    "payer_historical_denial_rate_by_type",
    "days_since_last_payment",
    "payer_contract_stop_loss",
    "documents_medical_necessity",
    "mentions_lack_of_pre_auth",
)


# ---------------------------------------------------------------------------
# ScoringResult
# ---------------------------------------------------------------------------


@dataclass
class ScoringResult:
    """
    Output record produced by the Batch_Scorer for a single claim.

    Attributes
    ----------
    claim_id:
        Claim identifier (Feature Store row key).
    predicted_denial_score:
        Model output — float32 in [0.0, 1.0], or ``None`` on failure.
    model_version_id:
        Version identifier of the model that produced this score.
    scoring_timestamp_utc:
        ISO-8601 UTC timestamp when scoring occurred.
    score_quality_flag:
        ``"degraded"`` when one or more numeric features were null after
        imputation; ``None`` otherwise.
    error:
        Exception message when scoring failed for this claim; ``None``
        on success.
    """

    claim_id: str
    predicted_denial_score: Optional[float]
    model_version_id: str
    scoring_timestamp_utc: str
    score_quality_flag: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------


@dataclass
class PipelineStats:
    """
    Aggregate statistics emitted in the pipeline completion event.

    Attributes
    ----------
    total_active_claims:
        Total number of Active claims retrieved from the Feature Store.
    total_scored:
        Claims for which a score was successfully produced.
    degraded_count:
        Claims scored with ``score_quality_flag = "degraded"``.
    failed_count:
        Claims that failed scoring entirely.
    wall_clock_seconds:
        Pipeline wall-clock duration in seconds.
    """

    total_active_claims: int = 0
    total_scored: int = 0
    degraded_count: int = 0
    failed_count: int = 0
    wall_clock_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PipelineBlockedError(RuntimeError):
    """
    Raised when the nightly pipeline cannot proceed because no Production model
    is available in the Model Registry.

    Requirement 10.6
    """


# ---------------------------------------------------------------------------
# Feature Store protocol — thin interface for testability
# ---------------------------------------------------------------------------


class FeatureStoreProtocol(Protocol):
    """Minimal interface required by BatchScorer for Feature Store access."""

    def fetch_active_claims(self) -> List[ClaimFeatureRecord]:
        """Return all claims with claim_status = 'Active'."""
        ...

    def upsert(self, claim_id: str, columns: Dict[str, Any]) -> None:
        """Upsert *columns* into the row keyed by *claim_id*."""
        ...


class InMemoryFeatureStore:
    """
    In-memory Feature Store for tests and local development.

    Stores :class:`~claim_denial.models.ClaimFeatureRecord` instances and
    supports the :class:`FeatureStoreProtocol` interface.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, ClaimFeatureRecord] = {}

    def add_claim(self, record: ClaimFeatureRecord) -> None:
        """Insert or replace a :class:`ClaimFeatureRecord`."""
        self._rows[record.claim_id] = record

    def fetch_active_claims(self) -> List[ClaimFeatureRecord]:
        """Return all stored claims with ``claim_status == 'Active'``."""
        return [r for r in self._rows.values() if r.claim_status == "Active"]

    def upsert(self, claim_id: str, columns: Dict[str, Any]) -> None:
        """Merge *columns* into the row keyed by *claim_id*.

        If the row does not exist it is silently ignored (only updates are
        relevant in the scoring context — claims must have been fetched first).
        """
        if claim_id in self._rows:
            rec = self._rows[claim_id]
            for key, value in columns.items():
                if hasattr(rec, key):
                    object.__setattr__(rec, key, value)
        else:
            logger.warning(
                "BatchScorer.upsert: claim_id '%s' not found in store; skipping.",
                claim_id,
            )

    def get(self, claim_id: str) -> Optional[ClaimFeatureRecord]:
        """Return the stored record for *claim_id*, or ``None``."""
        return self._rows.get(claim_id)


# ---------------------------------------------------------------------------
# ModelRegistry protocol — thin interface so BatchScorer does not hard-depend
# on the concrete ModelRegistry (allows injection of mocks in tests).
# ---------------------------------------------------------------------------


class ModelRegistryProtocol(Protocol):
    """Minimal interface required by BatchScorer."""

    def get_production_model(self) -> ModelArtifact:
        """Return the current Production model artifact."""
        ...


# ---------------------------------------------------------------------------
# BatchScorer
# ---------------------------------------------------------------------------


class BatchScorer:
    """
    Nightly batch scoring pipeline component.

    Parameters
    ----------
    model_registry:
        Object implementing :class:`ModelRegistryProtocol`.  Defaults to a new
        ``ModelRegistry`` instance when not provided — inject a mock for tests.
    feature_store:
        Object implementing :class:`FeatureStoreProtocol`.  Defaults to a new
        :class:`InMemoryFeatureStore` when not provided.
    notification_channel:
        Callable that accepts a string message.  Used to emit alerts
        (pipeline-blocked, SLA breach).  Defaults to a logger-based emitter.

    Usage
    -----
    The typical call order mirrors the Airflow task chain::

        scorer = BatchScorer(registry, feature_store)
        model   = scorer.load_production_model()     # halts on failure
        claims  = scorer.fetch_active_claims()
        results = scorer.score_claims(model, claims)
        scorer.write_scores(results)
        scorer.emit_completion_event(stats)
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistryProtocol] = None,
        feature_store: Optional[FeatureStoreProtocol] = None,
        notification_channel: Optional[Any] = None,
    ) -> None:
        # Lazy import to avoid a circular dependency at module load time.
        if model_registry is None:
            from claim_denial.registry.model_registry import ModelRegistry
            model_registry = ModelRegistry()
        if feature_store is None:
            feature_store = InMemoryFeatureStore()

        self._registry: ModelRegistryProtocol = model_registry
        self._store: FeatureStoreProtocol = feature_store
        self._notify = notification_channel or self._default_notify

    # ------------------------------------------------------------------
    # 1.  Load production model  (Requirement 10.6)
    # ------------------------------------------------------------------

    def load_production_model(self) -> ModelArtifact:
        """
        Query the Model Registry for the Production model artifact.

        If no Production model exists, emits a pipeline-blocked alert (as a
        structured log event and via the notification channel) and raises
        :class:`PipelineBlockedError` to halt the pipeline.

        Returns
        -------
        ModelArtifact
            The current Production model artifact.

        Raises
        ------
        PipelineBlockedError
            When :class:`NoProductionModelError` is raised by the registry.
        """
        try:
            model = self._registry.get_production_model()
            logger.info(
                "BatchScorer: loaded Production model version_id='%s'.",
                model.version_id,
            )
            return model
        except NoProductionModelError as exc:
            alert_message = (
                "PIPELINE_BLOCKED: No Production model exists in the Model Registry. "
                "Nightly batch scoring pipeline cannot proceed. "
                f"Original error: {exc}"
            )
            # Structured log — parseable by log aggregation systems.
            logger.error(
                '{"event": "pipeline_blocked", "reason": "no_production_model", '
                '"timestamp_utc": "%s"}',
                datetime.utcnow().isoformat() + "Z",
            )
            self._notify(alert_message)
            raise PipelineBlockedError(alert_message) from exc

    # ------------------------------------------------------------------
    # 2.  Fetch active claims  (Requirement 11.2)
    # ------------------------------------------------------------------

    def fetch_active_claims(self) -> List[ClaimFeatureRecord]:
        """
        Query the Feature Store for all claims with ``claim_status = "Active"``.

        Returns
        -------
        List[ClaimFeatureRecord]
            All Active claim feature records staged for scoring.
        """
        claims = self._store.fetch_active_claims()
        logger.info(
            "BatchScorer: fetched %d Active claims from Feature Store.",
            len(claims),
        )
        return claims

    # ------------------------------------------------------------------
    # 3.  Score claims  (Requirements 11.3, 11.6)
    # ------------------------------------------------------------------

    def score_claims(
        self,
        model: ModelArtifact,
        claims: List[ClaimFeatureRecord],
    ) -> List[ScoringResult]:
        """
        Apply *model* to each claim feature vector and produce scoring results.

        Distributed inference is performed via PySpark when a SparkSession is
        active; otherwise each claim is scored in-process (pure-Python fallback
        for unit tests and environments without a Spark cluster).

        Requirements: 11.3, 11.6

        Parameters
        ----------
        model:
            Production :class:`ModelArtifact` loaded by
            :meth:`load_production_model`.
        claims:
            List of :class:`ClaimFeatureRecord` instances to score.

        Returns
        -------
        List[ScoringResult]
            One :class:`ScoringResult` per claim.  Failed claims have
            ``predicted_denial_score = None`` and a non-None ``error``.
        """
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
        except ImportError:
            spark = None

        if spark is not None:
            return self._score_claims_spark(spark, model, claims)
        else:
            return self._score_claims_local(model, claims)

    # ------------------------------------------------------------------
    # 4.  Write scores  (Requirements 11.4, 11.5; error handling design)
    # ------------------------------------------------------------------

    def write_scores(self, results: List[ScoringResult]) -> None:
        """
        Upsert scored results back to the Feature Store.

        For each result, the following columns are written:
        - ``predicted_denial_score``
        - ``model_version_id``
        - ``scoring_timestamp_utc``
        - ``score_quality_flag``

        Write failures are retried up to :data:`WRITE_MAX_RETRIES` times with
        exponential back-off (1 s, 2 s, 4 s).  After exhausting retries the
        failure is logged and processing continues with the next result.

        Requirements: 11.4, 11.5

        Parameters
        ----------
        results:
            Scoring results from :meth:`score_claims`.
        """
        for result in results:
            if result.predicted_denial_score is None:
                # Skip claims that failed scoring — nothing useful to write.
                logger.warning(
                    "BatchScorer.write_scores: skipping failed claim_id='%s' (error=%s).",
                    result.claim_id,
                    result.error,
                )
                continue

            columns: Dict[str, Any] = {
                "predicted_denial_score": result.predicted_denial_score,
                "model_version_id": result.model_version_id,
                "scoring_timestamp_utc": result.scoring_timestamp_utc,
                "score_quality_flag": result.score_quality_flag,
            }

            self._write_with_retry(result.claim_id, columns)

    # ------------------------------------------------------------------
    # 5.  Emit completion event  (Requirement 11.7)
    # ------------------------------------------------------------------

    def emit_completion_event(self, stats: PipelineStats) -> None:
        """
        Emit a structured pipeline completion event.

        The event is emitted via the logger at INFO level (structured JSON)
        and through the notification channel.

        Requirement 11.7

        Parameters
        ----------
        stats:
            Aggregate statistics from the completed pipeline run.
        """
        event = {
            "event": "batch_scoring_complete",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "total_active_claims": stats.total_active_claims,
            "total_scored": stats.total_scored,
            "degraded_count": stats.degraded_count,
            "failed_count": stats.failed_count,
            "wall_clock_seconds": stats.wall_clock_seconds,
        }
        import json as _json
        message = _json.dumps(event)
        logger.info("BatchScorer completion event: %s", message)
        self._notify(message)

    # ------------------------------------------------------------------
    # Private — Spark scoring path
    # ------------------------------------------------------------------

    def _score_claims_spark(
        self,
        spark: Any,
        model: ModelArtifact,
        claims: List[ClaimFeatureRecord],
    ) -> List[ScoringResult]:
        """
        Score *claims* using Spark distributed inference via mapPartitions.

        Each partition receives a batch of serialized feature dicts and the
        model artifact; the model's ``predict_proba`` method (or equivalent)
        is applied per record.
        """
        import pandas as pd
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            FloatType,
            StringType,
            StructField,
            StructType,
        )

        timestamp_utc = datetime.utcnow().isoformat() + "Z"
        version_id = model.version_id
        model_object = model.model_object

        # Convert claims to a list of dicts for Spark parallelisation.
        claim_dicts = [_claim_to_feature_dict(c) for c in claims]
        claim_ids = [c.claim_id for c in claims]
        quality_flags = [_detect_quality_flag(c) for c in claims]

        # Broadcast the model object to all executors.
        model_bc = spark.sparkContext.broadcast(model_object)

        # Build an RDD of (claim_id, feature_dict, quality_flag) tuples.
        rdd_data = list(zip(claim_ids, claim_dicts, quality_flags))
        rdd = spark.sparkContext.parallelize(rdd_data)

        def score_partition(records: Any) -> Any:
            """Score a single partition — runs on Spark executors."""
            model_local = model_bc.value
            for claim_id, feat_dict, quality_flag in records:
                try:
                    score = _invoke_model(model_local, feat_dict)
                    # Clamp to [0.0, 1.0] and cast to float32.
                    score_f32 = float(max(0.0, min(1.0, float(score))))
                    yield (claim_id, score_f32, quality_flag, None)
                except Exception as exc:  # noqa: BLE001
                    yield (claim_id, None, quality_flag, str(exc))

        raw_results = rdd.mapPartitions(score_partition).collect()

        return [
            ScoringResult(
                claim_id=claim_id,
                predicted_denial_score=score,
                model_version_id=version_id,
                scoring_timestamp_utc=timestamp_utc,
                score_quality_flag=quality_flag,
                error=error,
            )
            for claim_id, score, quality_flag, error in raw_results
        ]

    # ------------------------------------------------------------------
    # Private — local (non-Spark) scoring path
    # ------------------------------------------------------------------

    def _score_claims_local(
        self,
        model: ModelArtifact,
        claims: List[ClaimFeatureRecord],
    ) -> List[ScoringResult]:
        """
        Score *claims* in the current process (single-threaded).

        Used when no active SparkSession is available — primarily for unit
        tests and local development environments without Spark.
        """
        timestamp_utc = datetime.utcnow().isoformat() + "Z"
        version_id = model.version_id
        model_object = model.model_object

        results: List[ScoringResult] = []

        for claim in claims:
            quality_flag = _detect_quality_flag(claim)
            try:
                feat_dict = _claim_to_feature_dict(claim)
                score = _invoke_model(model_object, feat_dict)
                # Clamp to [0.0, 1.0] and cast to float32.
                score_f32 = float(max(0.0, min(1.0, float(score))))
                results.append(
                    ScoringResult(
                        claim_id=claim.claim_id,
                        predicted_denial_score=score_f32,
                        model_version_id=version_id,
                        scoring_timestamp_utc=timestamp_utc,
                        score_quality_flag=quality_flag,
                        error=None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "BatchScorer: scoring failed for claim_id='%s': %s",
                    claim.claim_id,
                    exc,
                )
                results.append(
                    ScoringResult(
                        claim_id=claim.claim_id,
                        predicted_denial_score=None,
                        model_version_id=version_id,
                        scoring_timestamp_utc=timestamp_utc,
                        score_quality_flag=quality_flag,
                        error=str(exc),
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Private — write with retry / exponential back-off
    # ------------------------------------------------------------------

    def _write_with_retry(
        self,
        claim_id: str,
        columns: Dict[str, Any],
    ) -> None:
        """
        Attempt to upsert *columns* to the Feature Store up to
        :data:`WRITE_MAX_RETRIES` times.

        Back-off schedule: 1 s, 2 s, 4 s (exponential, base
        :data:`WRITE_BACKOFF_BASE_SECONDS`).
        """
        delay = WRITE_BACKOFF_BASE_SECONDS
        for attempt in range(1, WRITE_MAX_RETRIES + 1):
            try:
                self._store.upsert(claim_id, columns)
                if attempt > 1:
                    logger.info(
                        "BatchScorer: write succeeded for claim_id='%s' on attempt %d.",
                        claim_id,
                        attempt,
                    )
                return
            except Exception as exc:  # noqa: BLE001
                if attempt < WRITE_MAX_RETRIES:
                    logger.warning(
                        "BatchScorer: write attempt %d/%d failed for claim_id='%s' "
                        "(retrying in %.1fs): %s",
                        attempt,
                        WRITE_MAX_RETRIES,
                        claim_id,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        "BatchScorer: all %d write attempts failed for claim_id='%s': %s",
                        WRITE_MAX_RETRIES,
                        claim_id,
                        exc,
                    )

    # ------------------------------------------------------------------
    # Private — default notification channel
    # ------------------------------------------------------------------

    @staticmethod
    def _default_notify(message: str) -> None:
        """Log *message* at WARNING level as the default notification channel."""
        logger.warning("BatchScorer notification: %s", message)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _detect_quality_flag(claim: ClaimFeatureRecord) -> Optional[str]:
    """
    Return ``"degraded"`` when any numeric feature field on *claim* is
    ``None`` after imputation; ``None`` otherwise.

    Requirement 11.6
    """
    for field_name in NUMERIC_FEATURE_FIELDS:
        value = getattr(claim, field_name, None)
        if value is None:
            return SCORE_QUALITY_DEGRADED
    return None


def _claim_to_feature_dict(claim: ClaimFeatureRecord) -> Dict[str, Any]:
    """
    Extract the numeric feature columns from *claim* into a flat dict.

    ``None`` values are replaced with 0 (imputation sentinel) so that
    downstream model inference does not encounter Python ``None``s; the
    quality flag is set separately via :func:`_detect_quality_flag`.
    """
    result: Dict[str, Any] = {}
    for field_name in NUMERIC_FEATURE_FIELDS:
        value = getattr(claim, field_name, None)
        result[field_name] = 0 if value is None else value

    # Include the procedure vector (list of ints).
    proc_vec = claim.procedure_category_vector
    result["procedure_category_vector"] = proc_vec if proc_vec is not None else []

    # Include the NLP embedding (list of floats).
    nlp_emb = claim.tx_plan_complexity_embedding
    result["tx_plan_complexity_embedding"] = nlp_emb if nlp_emb is not None else []

    return result


def _invoke_model(model_object: Any, feat_dict: Dict[str, Any]) -> float:
    """
    Invoke the in-memory model object and return a single probability score.

    Supports several common ML library interfaces:

    - Objects with a ``predict_proba`` method (scikit-learn, XGBoost
      sklearn-API, LightGBM sklearn-API, CatBoost sklearn-API):
      calls ``predict_proba([feature_vector])[0][1]`` (probability of class 1).
    - Objects with a ``predict`` method that returns probabilities directly
      (e.g. XGBoost Booster, LightGBM Booster).
    - Callables (for test stubs): called directly with *feat_dict*.

    Raises
    ------
    TypeError
        When *model_object* does not expose a recognised inference interface.
    """
    if model_object is None:
        raise TypeError("Model artifact has no model_object; cannot score.")

    # Build an ordered feature vector from the feature dict.
    feature_vector = [feat_dict.get(f, 0) for f in NUMERIC_FEATURE_FIELDS]

    # Try scikit-learn / sklearn-API: predict_proba returns [[p0, p1], ...]
    if hasattr(model_object, "predict_proba"):
        proba = model_object.predict_proba([feature_vector])
        # proba shape: (1, n_classes).  Class-1 probability is the last column.
        return float(proba[0][-1])

    # Try XGBoost native Booster / LightGBM native Booster: predict(DMatrix)
    if hasattr(model_object, "predict"):
        result = model_object.predict([feature_vector])
        # Returns an array-like of shape (1,) or a scalar.
        if hasattr(result, "__len__"):
            return float(result[0])
        return float(result)

    # Callable stub (used in unit tests).
    if callable(model_object):
        return float(model_object(feat_dict))

    raise TypeError(
        f"Unsupported model_object type: {type(model_object)!r}.  "
        "Expected an object with predict_proba(), predict(), or a callable."
    )
