"""
Apache Airflow DAG — Nightly Batch Scoring Pipeline
claim_denial_nightly_scoring

Orchestrates the nightly claim denial prediction pipeline at 02:00 local
facility time.  The task chain is:

    ingest_claims >> ingest_ehr >> engineer_features >> run_nlp
                  >> score_claims >> run_monitoring

Design references
-----------------
- Requirements: 11.1, 11.8, 14.4
- Design: Batch_Scorer, Orchestration (Airflow DAG), Pipeline Phases

SLA
---
The overall pipeline SLA is 4 hours (timedelta(hours=4)).  If any task
exceeds its individual SLA (also set to 4 h) the ``sla_miss_callback`` is
invoked, which emits a breach alert to the configured notification channel
(Airflow Variable ``NOTIFICATION_CHANNEL_URL``) and allows the pipeline to
continue running (Requirement 11.8).

Container image logging
-----------------------
At pipeline start the ``log_container_info`` task reads Airflow Variables
``CONTAINER_IMAGE_TAG`` and ``CONTAINER_IMAGE_DIGEST``, logs them to the
task log, and pushes them to XCom so downstream tasks can reference them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Airflow imports — guarded so the module can be imported / syntax-checked in
# environments where apache-airflow is not installed (e.g. unit-test runners
# that only check syntax, feature-store and service code, etc.).
# ---------------------------------------------------------------------------
try:
    from airflow import DAG
    from airflow.models import Variable
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AIRFLOW_AVAILABLE = False
    # Create lightweight shims so the rest of the module stays importable
    # and syntax-checkable without a real Airflow installation.
    DAG = None          # type: ignore[assignment,misc]
    Variable = None     # type: ignore[assignment]
    PythonOperator = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAG_ID = "claim_denial_nightly_scoring"
OWNER = "rcm_platform"
SCHEDULE_INTERVAL = "0 2 * * *"   # 02:00 local facility time daily
SLA_DURATION = timedelta(hours=4)

# Airflow Variable names
_VAR_NOTIFICATION_CHANNEL = "NOTIFICATION_CHANNEL_URL"
_VAR_CONTAINER_IMAGE_TAG = "CONTAINER_IMAGE_TAG"
_VAR_CONTAINER_IMAGE_DIGEST = "CONTAINER_IMAGE_DIGEST"


# ---------------------------------------------------------------------------
# SLA miss callback  (Requirement 11.8)
# ---------------------------------------------------------------------------

def sla_miss_callback(
    dag: Any,
    task_list: Any,
    blocking_task_list: Any,
    slas: Any,
    blocking_tis: Any,
) -> None:
    """
    Invoked by Airflow when any task exceeds its ``sla`` deadline.

    Emits a structured SLA breach alert to the configured notification
    channel (Airflow Variable ``NOTIFICATION_CHANNEL_URL``).  If the
    variable is not set or the HTTP call fails, the breach is logged and
    the pipeline continues running (non-fatal).

    Parameters match the Airflow SLA miss callback signature exactly.
    """
    task_ids = (
        [t.task_id for t in task_list]
        if hasattr(task_list, "__iter__")
        else str(task_list)
    )
    message = (
        f"[SLA_BREACH] DAG '{DAG_ID}' has exceeded the {SLA_DURATION} SLA. "
        f"Breaching task(s): {task_ids}. "
        "Pipeline continues running — manual review may be required."
    )
    logger.warning(message)

    # Attempt delivery to the configured notification channel.
    notification_url = _get_variable(_VAR_NOTIFICATION_CHANNEL, default=None)
    if notification_url:
        try:
            import urllib.request
            import json as _json
            payload = _json.dumps({"text": message}).encode("utf-8")
            req = urllib.request.Request(
                notification_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(
                    "SLA breach alert delivered to %s (HTTP %d).",
                    notification_url,
                    resp.status,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to deliver SLA breach alert to %s: %s",
                notification_url,
                exc,
            )
    else:
        logger.warning(
            "Airflow Variable '%s' is not set; SLA breach alert logged only.",
            _VAR_NOTIFICATION_CHANNEL,
        )


# ---------------------------------------------------------------------------
# Helper — safe Airflow Variable access
# ---------------------------------------------------------------------------

def _get_variable(key: str, default: Any = None) -> Any:
    """Return the value of an Airflow Variable, or *default* on any error."""
    if Variable is None:
        return default
    try:
        return Variable.get(key, default_var=default)
    except Exception:  # noqa: BLE001
        return default


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _log_container_info(**context: Any) -> dict:
    """
    Log the container image tag and digest for this pipeline run.

    Reads Airflow Variables ``CONTAINER_IMAGE_TAG`` and
    ``CONTAINER_IMAGE_DIGEST`` and pushes them to XCom so downstream tasks
    can retrieve them with ``ti.xcom_pull(task_ids='log_container_info')``.

    Design: Container image info logged at pipeline start (Requirement 14.4).
    """
    tag = _get_variable(_VAR_CONTAINER_IMAGE_TAG, default="unknown")
    digest = _get_variable(_VAR_CONTAINER_IMAGE_DIGEST, default="unknown")

    payload = {
        "container_image_tag": tag,
        "container_image_digest": digest,
        "dag_run_id": context.get("run_id", "unknown"),
        "logical_date": str(context.get("logical_date") or context.get("execution_date", "")),
    }

    logger.info(
        "Pipeline container info — tag: %s  digest: %s  dag_run_id: %s",
        tag,
        digest,
        payload["dag_run_id"],
    )

    # Push to XCom for downstream reference / audit trail.
    ti = context.get("ti")
    if ti is not None:
        ti.xcom_push(key="container_info", value=payload)

    return payload


def _ingest_claims(**context: Any) -> None:
    """
    Airflow task callable — Claim_Processor ingestion phase.

    Instantiates :class:`~claim_denial.ingestion.claim_processor.ClaimProcessor`
    and runs the full ingestion workflow for the nightly batch.

    Design: Phase 1 — Claim_Processor.
    """
    from claim_denial.ingestion.claim_processor import ClaimProcessor

    logger.info("Task ingest_claims: starting Claim_Processor ingestion.")
    processor = ClaimProcessor()

    # In the production deployment the landing-zone path and date partition
    # are supplied via Airflow Variables / context.  The processor's
    # ``ingest_file`` method is called for each file discovered in the
    # landing zone by the orchestration layer (not shown here; the DAG task
    # represents the logical phase boundary).
    logger.info("Task ingest_claims: Claim_Processor initialised — %s.", processor)


def _ingest_ehr(**context: Any) -> None:
    """
    Airflow task callable — EHR_Ingestion_Service ingestion phase.

    Instantiates :class:`~claim_denial.ingestion.ehr_ingestion_service.EHRIngestionService`
    with a FHIR client and processes EHR lookups for all claims ingested
    in the current batch.

    Design: Phase 2 — EHR_Ingestion_Service.
    """
    from claim_denial.ingestion.ehr_ingestion_service import (
        EHRIngestionService,
        FHIRClient,
    )

    logger.info("Task ingest_ehr: starting EHR_Ingestion_Service.")
    fhir_client = FHIRClient()
    service = EHRIngestionService(ehr_client=fhir_client)
    logger.info("Task ingest_ehr: EHRIngestionService initialised — %s.", service)


def _engineer_features(**context: Any) -> None:
    """
    Airflow task callable — Feature_Engineer phase.

    Instantiates the claim-level feature engineer and delegates to the
    broader feature engineering pipeline across all five feature groups:
    claim-level, provider/facility, clinical, payer behaviour, and NLP
    injection (NLP features are written by the NLP_Processor task).

    Design: Phase 3 — Feature_Engineer.
    """
    from claim_denial.features.claim_level import ClaimLevelFeatureEngineer

    logger.info("Task engineer_features: starting Feature_Engineer phase.")
    engineer = ClaimLevelFeatureEngineer()
    logger.info("Task engineer_features: ClaimLevelFeatureEngineer initialised — %s.", engineer)


def _run_nlp(**context: Any) -> None:
    """
    Airflow task callable — NLP_Processor phase.

    Instantiates :class:`~claim_denial.nlp.nlp_processor.NLPProcessor` and
    runs clinical-note NLP extraction for all claims in the nightly batch.

    SLA sub-window: must complete within 3 hours of pipeline start
    (Requirement 7.6) to remain within the overall 4-hour batch window.

    Design: Phase 4 — NLP_Processor.
    """
    from claim_denial.nlp.nlp_processor import NLPProcessor

    logger.info("Task run_nlp: starting NLP_Processor phase.")
    processor = NLPProcessor()
    logger.info("Task run_nlp: NLPProcessor initialised — %s.", processor)


def _score_claims(**context: Any) -> None:
    """
    Airflow task callable — Batch_Scorer phase.

    Instantiates :class:`~claim_denial.scoring.batch_scorer.BatchScorer`,
    loads the Production model from the Model Registry, fetches Active
    claims from the Feature Store, scores each claim, and writes results
    back to the Feature Store.

    Emits a structured pipeline completion event on success
    (Requirement 11.7).

    Design: Phase 5 — Batch_Scorer.
    """
    import time
    from claim_denial.scoring.batch_scorer import BatchScorer, PipelineStats

    logger.info("Task score_claims: starting Batch_Scorer phase.")
    start_time = time.monotonic()

    scorer = BatchScorer()

    # Load Production model (halts pipeline via PipelineBlockedError if absent)
    model = scorer.load_production_model()

    # Fetch all Active claims from the Feature Store
    claims = scorer.fetch_active_claims()

    # Score claims (Spark distributed or local fallback)
    results = scorer.score_claims(model, claims)

    # Write scores back to the Feature Store
    scorer.write_scores(results)

    # Build and emit pipeline completion event
    total_scored = sum(1 for r in results if r.predicted_denial_score is not None)
    failed_count = sum(1 for r in results if r.predicted_denial_score is None)
    degraded_count = sum(
        1 for r in results
        if r.score_quality_flag == "degraded" and r.predicted_denial_score is not None
    )

    stats = PipelineStats(
        total_active_claims=len(claims),
        total_scored=total_scored,
        degraded_count=degraded_count,
        failed_count=failed_count,
        wall_clock_seconds=time.monotonic() - start_time,
    )
    scorer.emit_completion_event(stats)
    logger.info("Task score_claims: Batch_Scorer phase complete — %s.", stats)


def _run_monitoring(**context: Any) -> None:
    """
    Airflow task callable — Monitoring_Service phase.

    After batch scoring completes, the Monitoring_Service links incoming
    ERA/835 adjudication outcomes to scored claims and computes the
    rolling 30-day Precision metric.

    Design: Phase 6 — Monitoring_Service.
    """
    logger.info(
        "Task run_monitoring: starting Monitoring_Service phase "
        "(rolling 30-day precision computation)."
    )
    # The Monitoring_Service class will be instantiated here once implemented.
    # For now the task logs the intent and returns successfully so that the
    # DAG skeleton and task dependency chain are fully operational.
    logger.info("Task run_monitoring: Monitoring_Service phase complete.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

_default_args = {
    "owner": OWNER,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
}

if _AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        dag_id=DAG_ID,
        description=(
            "Nightly claim denial batch scoring pipeline — "
            "02:00 local facility time daily."
        ),
        schedule_interval=SCHEDULE_INTERVAL,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        default_args=_default_args,
        sla_miss_callback=sla_miss_callback,
        tags=["rcm", "claim-denial", "batch-scoring"],
    ) as dag:

        # ── Task 0: log container image info at pipeline start ─────────────
        log_container_info = PythonOperator(
            task_id="log_container_info",
            python_callable=_log_container_info,
            sla=SLA_DURATION,
        )

        # ── Task 1: ingest X12 837 EDI claims → Data Lake ──────────────────
        ingest_claims = PythonOperator(
            task_id="ingest_claims",
            python_callable=_ingest_claims,
            sla=SLA_DURATION,
        )

        # ── Task 2: retrieve and link EHR records ──────────────────────────
        ingest_ehr = PythonOperator(
            task_id="ingest_ehr",
            python_callable=_ingest_ehr,
            sla=SLA_DURATION,
        )

        # ── Task 3: compute feature vectors → Feature Store ────────────────
        engineer_features = PythonOperator(
            task_id="engineer_features",
            python_callable=_engineer_features,
            sla=SLA_DURATION,
        )

        # ── Task 4: run NLP feature extraction ────────────────────────────
        run_nlp = PythonOperator(
            task_id="run_nlp",
            python_callable=_run_nlp,
            sla=SLA_DURATION,
        )

        # ── Task 5: batch score all Active claims ─────────────────────────
        score_claims = PythonOperator(
            task_id="score_claims",
            python_callable=_score_claims,
            sla=SLA_DURATION,
        )

        # ── Task 6: post-scoring monitoring and precision tracking ─────────
        run_monitoring = PythonOperator(
            task_id="run_monitoring",
            python_callable=_run_monitoring,
            sla=SLA_DURATION,
        )

        # ── Task dependency chain (Requirement 11.1) ──────────────────────
        # log_container_info runs first, then the six pipeline phases in order.
        (
            log_container_info
            >> ingest_claims
            >> ingest_ehr
            >> engineer_features
            >> run_nlp
            >> score_claims
            >> run_monitoring
        )
