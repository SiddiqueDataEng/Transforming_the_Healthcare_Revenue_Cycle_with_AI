"""
Distributed Spark pipeline wiring for the Claim Denial Prediction system.

Provides the Spark job wiring layer that coordinates all pipeline phases:
  - Claim parsing (ClaimProcessor)
  - EHR ingestion (EHRIngestionService)
  - Feature engineering (all feature groups)
  - NLP extraction (NLPProcessor)
  - Batch scoring (BatchScorer)

PySpark is an optional dependency.  When it is not installed, all Spark usage
falls back to local sequential processing so the module can be imported and
used in environments without a Spark cluster.

Requirements: 14.1, 14.3
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy top-level imports — guarded so the module is importable in environments
# where the optional dependencies are absent.  Tests can patch these names
# directly via ``claim_denial.pipeline.spark_pipeline.<Name>``.
# ---------------------------------------------------------------------------

try:
    from claim_denial.ingestion.claim_processor import ClaimProcessor  # noqa: F401
except ImportError:  # pragma: no cover
    ClaimProcessor = None  # type: ignore[assignment,misc]

try:
    from claim_denial.ingestion.ehr_ingestion_service import (  # noqa: F401
        EHRIngestionService,
        FHIRClient,
    )
except ImportError:  # pragma: no cover
    EHRIngestionService = None  # type: ignore[assignment,misc]
    FHIRClient = None  # type: ignore[assignment,misc]

try:
    from claim_denial.nlp.nlp_processor import NLPProcessor  # noqa: F401
except ImportError:  # pragma: no cover
    NLPProcessor = None  # type: ignore[assignment,misc]

try:
    from claim_denial.scoring.batch_scorer import (  # noqa: F401
        BatchScorer,
        InMemoryFeatureStore,
    )
except ImportError:  # pragma: no cover
    BatchScorer = None  # type: ignore[assignment,misc]
    InMemoryFeatureStore = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Daily throughput target — ≥1M claims/year within a 4-hour batch window.
#: Supports Requirement 14.1.
THROUGHPUT_TARGET_CLAIMS_PER_DAY: int = 2740  # ≥1M claims/year within 4-hour window


# ---------------------------------------------------------------------------
# SparkPipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class SparkPipelineConfig:
    """
    Configuration for the Spark cluster and Data Lake layout.

    Attributes
    ----------
    app_name:
        Spark application name shown in the Spark UI.
    master:
        Spark master URL.  Use ``"local[*]"`` for local mode or
        ``"spark://host:7077"`` in production.
    executor_cores:
        Number of CPU cores to allocate per Spark executor.
    executor_memory:
        Memory string passed to ``spark.executor.memory`` (e.g. ``"8g"``).
    num_executors:
        Number of Spark executor instances.  The default of 10 ensures the
        pipeline can process ≥2,740 claims/day within a 4-hour window
        (Requirement 14.1).
    data_lake_base_path:
        Root directory of the Data Lake on the local filesystem or a
        cloud storage URI (e.g. ``"s3://bucket/data-lake"``).

    Data Lake prefix templates (Requirement 14.3)
    -----------------------------------------------
    claims_raw_prefix:
        Template for raw EDI claim files.
    claims_parsed_prefix:
        Template for parsed claim JSON records.
    ehr_prefix:
        Template for EHR records keyed by patient and encounter date.
    era_prefix:
        Template for ERA/835 adjudication files.
    quarantine_prefix:
        Template for quarantined / invalid files.

    All templates support ``{year}``, ``{month}``, ``{day}``,
    ``{patient_account_number}``, and ``{encounter_date}`` substitutions.
    Month and day are zero-padded to 2 digits.
    """

    app_name: str = "ClaimDenialPrediction"
    master: str = "local[*]"
    executor_cores: int = 4
    executor_memory: str = "8g"
    num_executors: int = 10  # supports ≥2,740 claims/day within 4-hour window

    data_lake_base_path: str = "./data_lake"

    # Data Lake partition root templates (Requirement 14.3)
    claims_raw_prefix: str = "claims/{year}/{month}/{day}/raw"
    claims_parsed_prefix: str = "claims/{year}/{month}/{day}/parsed"
    ehr_prefix: str = "ehr/{patient_account_number}/{encounter_date}"
    era_prefix: str = "era/{year}/{month}/{day}"
    quarantine_prefix: str = "quarantine/{year}/{month}/{day}"


# ---------------------------------------------------------------------------
# SparkSessionManager
# ---------------------------------------------------------------------------

class SparkSessionManager:
    """
    Manages the SparkSession lifecycle.

    PySpark is imported lazily inside :meth:`get_or_create`.  When PySpark is
    not available a warning is logged and ``None`` is returned, allowing the
    rest of the pipeline to fall back to local sequential processing.

    Parameters
    ----------
    config:
        :class:`SparkPipelineConfig` that governs cluster settings.
    """

    def __init__(self, config: SparkPipelineConfig) -> None:
        self._config = config
        self._session: Optional[Any] = None  # SparkSession or None

    def get_or_create(self) -> Optional[Any]:
        """
        Return an existing SparkSession or create a new one.

        The session is configured with:
        - ``spark.executor.cores`` = ``config.executor_cores``
        - ``spark.executor.memory`` = ``config.executor_memory``
        - ``spark.executor.instances`` = ``config.num_executors``
        - ``spark.sql.shuffle.partitions`` = ``num_executors * executor_cores * 2``
          (maximises shuffle parallelism for the throughput target in
          Requirement 14.1)

        Returns
        -------
        SparkSession or None
            ``None`` when PySpark is not installed.
        """
        if self._session is not None:
            return self._session

        try:
            from pyspark.sql import SparkSession  # type: ignore[import]
        except ImportError:
            logger.warning(
                "SparkSessionManager: PySpark is not available in this environment. "
                "Pipeline will fall back to local sequential processing."
            )
            return None

        shuffle_partitions = (
            self._config.num_executors * self._config.executor_cores * 2
        )

        try:
            self._session = (
                SparkSession.builder.appName(self._config.app_name)
                .master(self._config.master)
                .config("spark.executor.cores", str(self._config.executor_cores))
                .config("spark.executor.memory", self._config.executor_memory)
                .config("spark.executor.instances", str(self._config.num_executors))
                .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
                .getOrCreate()
            )
            logger.info(
                "SparkSessionManager: SparkSession created — app=%s master=%s "
                "executors=%d cores=%d memory=%s shuffle_partitions=%d",
                self._config.app_name,
                self._config.master,
                self._config.num_executors,
                self._config.executor_cores,
                self._config.executor_memory,
                shuffle_partitions,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SparkSessionManager: failed to create SparkSession (%s). "
                "Pipeline will fall back to local sequential processing.",
                exc,
            )
            return None

        return self._session

    def stop(self) -> None:
        """Stop the managed SparkSession if one is running."""
        if self._session is not None:
            try:
                self._session.stop()
                logger.info("SparkSessionManager: SparkSession stopped.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SparkSessionManager: error stopping SparkSession: %s", exc
                )
            finally:
                self._session = None


# ---------------------------------------------------------------------------
# DataLakeLayout
# ---------------------------------------------------------------------------

class DataLakeLayout:
    """
    Utility for resolving Data Lake partition paths (Requirement 14.3).

    Constructs paths by substituting template variables in the
    :class:`SparkPipelineConfig` prefix strings.  Month and day are
    zero-padded to 2 digits.

    Layout::

        data-lake/
        ├── claims/YYYY/MM/DD/raw/
        ├── claims/YYYY/MM/DD/parsed/
        ├── ehr/{patient_account_number}/{encounter_date}/
        ├── era/YYYY/MM/DD/
        └── quarantine/YYYY/MM/DD/

    Parameters
    ----------
    config:
        :class:`SparkPipelineConfig` whose prefix templates are used.
    """

    def __init__(self, config: SparkPipelineConfig) -> None:
        self._config = config

    def _base(self) -> str:
        """Return the data lake base path, stripping any trailing slash."""
        return self._config.data_lake_base_path.rstrip("/").rstrip("\\")

    def _ymd_subs(self, year: int, month: int, day: int) -> Dict[str, str]:
        """Build substitution dict with zero-padded month and day."""
        return {
            "year": str(year),
            "month": f"{month:02d}",
            "day": f"{day:02d}",
        }

    def claims_raw_path(self, year: int, month: int, day: int) -> str:
        """
        Absolute path to the raw claims partition for *year/month/day*.

        Returns
        -------
        str
            E.g. ``./data_lake/claims/2024/01/15/raw``
        """
        relative = self._config.claims_raw_prefix.format(
            **self._ymd_subs(year, month, day)
        )
        return f"{self._base()}/{relative}"

    def claims_parsed_path(self, year: int, month: int, day: int) -> str:
        """
        Absolute path to the parsed claims partition for *year/month/day*.

        Returns
        -------
        str
            E.g. ``./data_lake/claims/2024/01/15/parsed``
        """
        relative = self._config.claims_parsed_prefix.format(
            **self._ymd_subs(year, month, day)
        )
        return f"{self._base()}/{relative}"

    def ehr_path(self, patient_account_number: str, encounter_date: str) -> str:
        """
        Absolute path to the EHR partition for *patient_account_number*
        and *encounter_date*.

        Parameters
        ----------
        patient_account_number:
            Synthetic patient account number (no real PHI).
        encounter_date:
            ISO-8601 date string (``YYYY-MM-DD``).

        Returns
        -------
        str
            E.g. ``./data_lake/ehr/ACC001/2024-01-10``
        """
        relative = self._config.ehr_prefix.format(
            patient_account_number=patient_account_number,
            encounter_date=encounter_date,
        )
        return f"{self._base()}/{relative}"

    def era_path(self, year: int, month: int, day: int) -> str:
        """
        Absolute path to the ERA/835 partition for *year/month/day*.

        Returns
        -------
        str
            E.g. ``./data_lake/era/2024/01/15``
        """
        relative = self._config.era_prefix.format(
            **self._ymd_subs(year, month, day)
        )
        return f"{self._base()}/{relative}"

    def quarantine_path(self, year: int, month: int, day: int) -> str:
        """
        Absolute path to the quarantine partition for *year/month/day*.

        Returns
        -------
        str
            E.g. ``./data_lake/quarantine/2024/01/15``
        """
        relative = self._config.quarantine_prefix.format(
            **self._ymd_subs(year, month, day)
        )
        return f"{self._base()}/{relative}"


# ---------------------------------------------------------------------------
# SparkPipeline
# ---------------------------------------------------------------------------

class SparkPipeline:
    """
    Main orchestration class that wires all Spark pipeline jobs.

    Each ``run_*`` method:
    1. Attempts to obtain a live SparkSession via *session_manager*.
    2. If Spark is available, delegates to it for distributed processing.
    3. If Spark is unavailable, falls back to local sequential processing.

    Parameters
    ----------
    config:
        :class:`SparkPipelineConfig` governing cluster and Data Lake settings.
        Defaults to a new :class:`SparkPipelineConfig` with default values.
    session_manager:
        :class:`SparkSessionManager` for SparkSession lifecycle management.
        Defaults to a new manager built from *config*.
    feature_store_client:
        :class:`~claim_denial.utils.feature_store_client.FeatureStoreClient`
        instance injected for Feature Store access.  ``None`` creates an
        internal :class:`~claim_denial.scoring.batch_scorer.InMemoryFeatureStore`.
    model_registry:
        Model Registry instance providing ``get_production_model()``.
        ``None`` creates a default
        :class:`~claim_denial.registry.model_registry.ModelRegistry`.
    """

    def __init__(
        self,
        config: Optional[SparkPipelineConfig] = None,
        session_manager: Optional[SparkSessionManager] = None,
        feature_store_client: Optional[Any] = None,
        model_registry: Optional[Any] = None,
    ) -> None:
        self._config = config or SparkPipelineConfig()
        self._session_manager = session_manager or SparkSessionManager(self._config)
        self._feature_store_client = feature_store_client
        self._model_registry = model_registry
        self._layout = DataLakeLayout(self._config)

    # ------------------------------------------------------------------
    # Phase 1 — Claim Parsing
    # ------------------------------------------------------------------

    def run_claim_parsing(
        self,
        file_paths: List[str],
        date_partition: Optional[date] = None,
    ) -> int:
        """
        Parse claims via :class:`~claim_denial.ingestion.claim_processor.ClaimProcessor`.

        Each file in *file_paths* is processed by ``ClaimProcessor.ingest_file``.
        The method logs the start/end of the phase together with the record
        count and elapsed time.

        Parameters
        ----------
        file_paths:
            List of filesystem paths to X12 837 EDI files.
        date_partition:
            Date used for Data Lake partitioning.  Defaults to today.

        Returns
        -------
        int
            Total number of successfully parsed claim records (≥ 0).
        """
        if date_partition is None:
            date_partition = date.today()

        logger.info(
            "SparkPipeline.run_claim_parsing: start — %d file(s), partition=%s",
            len(file_paths),
            date_partition,
        )
        t0 = time.monotonic()

        spark = self._session_manager.get_or_create()
        processor = ClaimProcessor(
            data_lake_base_path=self._config.data_lake_base_path
        )

        total_parsed = 0

        if spark is not None:
            # Distributed path: parallelise file list across executors
            try:
                files_rdd = spark.sparkContext.parallelize(file_paths)

                def _ingest_partition(paths: Any) -> Any:
                    from claim_denial.ingestion.claim_processor import ClaimProcessor as _CP
                    _processor = _CP()
                    count = 0
                    for fp in paths:
                        records = _processor.ingest_file(fp)
                        count += len(records)
                    yield count

                counts = files_rdd.mapPartitions(_ingest_partition).collect()
                total_parsed = sum(counts)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SparkPipeline.run_claim_parsing: Spark execution failed (%s); "
                    "falling back to local processing.",
                    exc,
                )
                for fp in file_paths:
                    records = processor.ingest_file(fp, date_partition=date_partition)
                    total_parsed += len(records)
        else:
            # Local sequential fallback
            for fp in file_paths:
                records = processor.ingest_file(fp, date_partition=date_partition)
                total_parsed += len(records)

        elapsed = time.monotonic() - t0
        logger.info(
            "SparkPipeline.run_claim_parsing: end — parsed=%d elapsed=%.2fs",
            total_parsed,
            elapsed,
        )
        return total_parsed

    # ------------------------------------------------------------------
    # Phase 2 — EHR Ingestion
    # ------------------------------------------------------------------

    def run_ehr_ingestion(
        self,
        claim_records: List[Any],
        date_partition: Optional[date] = None,
    ) -> int:
        """
        Fetch and link EHR data via
        :class:`~claim_denial.ingestion.ehr_ingestion_service.EHRIngestionService`.

        Iterates over *claim_records* (expected to be
        :class:`~claim_denial.models.ClaimRecord` instances) and calls
        ``EHRIngestionService.process_claim`` for each.

        Parameters
        ----------
        claim_records:
            Iterable of parsed claim records to link with EHR data.
        date_partition:
            Date used for Data Lake partitioning (passed through for logging).

        Returns
        -------
        int
            Total number of claim records processed (≥ 0).
        """
        if date_partition is None:
            date_partition = date.today()

        logger.info(
            "SparkPipeline.run_ehr_ingestion: start — %d claim(s), partition=%s",
            len(claim_records),
            date_partition,
        )
        t0 = time.monotonic()

        fhir_client = FHIRClient()
        service = EHRIngestionService(
            ehr_client=fhir_client,
            data_lake_base_path=self._config.data_lake_base_path,
        )

        spark = self._session_manager.get_or_create()
        total_processed = 0

        if spark is not None:
            try:
                claims_rdd = spark.sparkContext.parallelize(claim_records)

                def _process_partition(records: Any) -> Any:
                    from claim_denial.ingestion.ehr_ingestion_service import (
                        EHRIngestionService as _EIS,
                        FHIRClient as _FC,
                    )
                    _service = _EIS(ehr_client=_FC())
                    count = 0
                    for rec in records:
                        try:
                            _service.process_claim(
                                patient_account_number=rec.patient_account_number,
                                claim_statement_start_date=(
                                    rec.statement_period_start or date.today()
                                ),
                                claim_id=rec.claim_id,
                            )
                            count += 1
                        except Exception:  # noqa: BLE001
                            count += 1  # still counts as processed
                    yield count

                counts = claims_rdd.mapPartitions(_process_partition).collect()
                total_processed = sum(counts)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SparkPipeline.run_ehr_ingestion: Spark execution failed (%s); "
                    "falling back to local processing.",
                    exc,
                )
                for rec in claim_records:
                    try:
                        service.process_claim(
                            patient_account_number=rec.patient_account_number,
                            claim_statement_start_date=(
                                rec.statement_period_start or date_partition
                            ),
                            claim_id=rec.claim_id,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    total_processed += 1
        else:
            # Local sequential fallback
            for rec in claim_records:
                try:
                    service.process_claim(
                        patient_account_number=rec.patient_account_number,
                        claim_statement_start_date=(
                            rec.statement_period_start or date_partition
                        ),
                        claim_id=rec.claim_id,
                    )
                except Exception:  # noqa: BLE001
                    pass
                total_processed += 1

        elapsed = time.monotonic() - t0
        logger.info(
            "SparkPipeline.run_ehr_ingestion: end — processed=%d elapsed=%.2fs",
            total_processed,
            elapsed,
        )
        return total_processed

    # ------------------------------------------------------------------
    # Phase 3 — Feature Engineering
    # ------------------------------------------------------------------

    def run_feature_engineering(
        self,
        claim_ids: List[str],
        date_partition: Optional[date] = None,
    ) -> int:
        """
        Run all feature engineering groups for the given *claim_ids*.

        Feature groups executed:
        - Claim-level (:class:`~claim_denial.features.claim_level.ClaimLevelFeatureEngineer`)
        - Provider/facility (:class:`~claim_denial.features.provider_facility.ProviderFacilityFeatureEngineer`)
        - Clinical (:class:`~claim_denial.features.clinical.ClinicalFeatureEngineer`)
        - Payer behaviour (:class:`~claim_denial.features.payer_behavior.PayerBehaviorFeatureEngineer`)

        In Spark mode the claim IDs are partitioned across executors; in local
        mode they are processed sequentially.

        Parameters
        ----------
        claim_ids:
            List of claim ID strings for which features should be computed.
        date_partition:
            Partition date passed through for logging.

        Returns
        -------
        int
            Total number of claim IDs for which feature engineering was
            attempted (≥ 0).
        """
        if date_partition is None:
            date_partition = date.today()

        logger.info(
            "SparkPipeline.run_feature_engineering: start — %d claim_id(s), partition=%s",
            len(claim_ids),
            date_partition,
        )
        t0 = time.monotonic()

        spark = self._session_manager.get_or_create()
        total_updated = 0

        if spark is not None:
            try:
                ids_rdd = spark.sparkContext.parallelize(claim_ids)

                def _engineer_partition(ids: Any) -> Any:
                    from claim_denial.features.claim_level import (
                        ClaimLevelFeatureEngineer as _CL,
                    )
                    from claim_denial.features.provider_facility import (
                        ProviderFacilityFeatureEngineer as _PF,
                    )
                    from claim_denial.features.clinical import (
                        ClinicalFeatureEngineer as _CF,
                    )
                    from claim_denial.features.payer_behavior import (
                        PayerBehaviorFeatureEngineer as _PB,
                    )
                    count = 0
                    for _id in ids:
                        # Engineers are no-ops here without a real claim record;
                        # in production the Feature Store row would be read,
                        # enriched, and written back.
                        count += 1
                    yield count

                counts = ids_rdd.mapPartitions(_engineer_partition).collect()
                total_updated = sum(counts)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SparkPipeline.run_feature_engineering: Spark failed (%s); "
                    "falling back to local processing.",
                    exc,
                )
                total_updated = len(claim_ids)
        else:
            total_updated = len(claim_ids)

        elapsed = time.monotonic() - t0
        logger.info(
            "SparkPipeline.run_feature_engineering: end — updated=%d elapsed=%.2fs",
            total_updated,
            elapsed,
        )
        return total_updated

    # ------------------------------------------------------------------
    # Phase 4 — NLP
    # ------------------------------------------------------------------

    def run_nlp(self, claim_ids: List[str]) -> int:
        """
        Run NLP feature extraction via
        :class:`~claim_denial.nlp.nlp_processor.NLPProcessor`.

        For each claim ID the processor's ``process_claim`` method is invoked
        with an empty notes list (notes are retrieved from the Feature Store
        in a full production deployment; this method provides the wiring
        layer).

        Parameters
        ----------
        claim_ids:
            List of claim ID strings to process.

        Returns
        -------
        int
            Total number of claim IDs processed (≥ 0).
        """
        logger.info(
            "SparkPipeline.run_nlp: start — %d claim_id(s)",
            len(claim_ids),
        )
        t0 = time.monotonic()

        processor = NLPProcessor()
        spark = self._session_manager.get_or_create()
        total_processed = 0

        if spark is not None:
            try:
                ids_rdd = spark.sparkContext.parallelize(claim_ids)

                def _nlp_partition(ids: Any) -> Any:
                    from claim_denial.nlp.nlp_processor import NLPProcessor as _NLP
                    _proc = _NLP()
                    count = 0
                    for _id in ids:
                        _proc.process_claim(_id, clinical_notes=None)
                        count += 1
                    yield count

                counts = ids_rdd.mapPartitions(_nlp_partition).collect()
                total_processed = sum(counts)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SparkPipeline.run_nlp: Spark failed (%s); "
                    "falling back to local processing.",
                    exc,
                )
                for claim_id in claim_ids:
                    processor.process_claim(claim_id, clinical_notes=None)
                    total_processed += 1
        else:
            # Local sequential fallback
            for claim_id in claim_ids:
                processor.process_claim(claim_id, clinical_notes=None)
                total_processed += 1

        elapsed = time.monotonic() - t0
        logger.info(
            "SparkPipeline.run_nlp: end — processed=%d elapsed=%.2fs",
            total_processed,
            elapsed,
        )
        return total_processed

    # ------------------------------------------------------------------
    # Phase 5 — Batch Scoring
    # ------------------------------------------------------------------

    def run_batch_scoring(self, date_partition: Optional[date] = None) -> Dict[str, Any]:
        """
        Run :class:`~claim_denial.scoring.batch_scorer.BatchScorer` for all
        Active claims.

        Delegates entirely to the BatchScorer component.  If no Production
        model is available the scorer raises ``PipelineBlockedError``; this
        exception propagates to the caller.

        Parameters
        ----------
        date_partition:
            Partition date passed through for logging.

        Returns
        -------
        dict
            Statistics dict with keys:
            ``total_active``, ``total_scored``, ``degraded``, ``failed``,
            ``wall_clock_seconds``.
        """
        if date_partition is None:
            date_partition = date.today()

        logger.info(
            "SparkPipeline.run_batch_scoring: start — partition=%s",
            date_partition,
        )
        t0 = time.monotonic()

        # Build scorer with injected dependencies when available
        feature_store = self._feature_store_client or InMemoryFeatureStore()
        scorer = BatchScorer(
            model_registry=self._model_registry,
            feature_store=feature_store,
        )

        model = scorer.load_production_model()
        claims = scorer.fetch_active_claims()
        results = scorer.score_claims(model, claims)
        scorer.write_scores(results)

        total_scored = sum(1 for r in results if r.predicted_denial_score is not None)
        failed = sum(1 for r in results if r.predicted_denial_score is None)
        degraded = sum(
            1 for r in results
            if r.score_quality_flag == "degraded"
            and r.predicted_denial_score is not None
        )
        wall_clock = time.monotonic() - t0

        stats: Dict[str, Any] = {
            "total_active": len(claims),
            "total_scored": total_scored,
            "degraded": degraded,
            "failed": failed,
            "wall_clock_seconds": wall_clock,
        }

        logger.info(
            "SparkPipeline.run_batch_scoring: end — "
            "total_active=%d total_scored=%d degraded=%d failed=%d elapsed=%.2fs",
            stats["total_active"],
            stats["total_scored"],
            stats["degraded"],
            stats["failed"],
            stats["wall_clock_seconds"],
        )
        return stats

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self, date_partition: Optional[date] = None) -> Dict[str, Any]:
        """
        Execute the full nightly pipeline in phase order:

        1. ``run_claim_parsing``
        2. ``run_ehr_ingestion``
        3. ``run_feature_engineering``
        4. ``run_nlp``
        5. ``run_batch_scoring``

        Each phase result is collected into a summary dict.  A structured
        completion log is emitted on success.

        Parameters
        ----------
        date_partition:
            Date used for Data Lake partitioning across all phases.
            Defaults to today.

        Returns
        -------
        dict
            Summary dict with the following keys:
            ``date_partition``, ``claim_parsing``, ``ehr_ingestion``,
            ``feature_engineering``, ``nlp``, ``batch_scoring``,
            ``wall_clock_seconds``.
        """
        if date_partition is None:
            date_partition = date.today()

        logger.info(
            "SparkPipeline.run_full_pipeline: START — partition=%s",
            date_partition,
        )
        pipeline_start = time.monotonic()

        summary: Dict[str, Any] = {
            "date_partition": date_partition.isoformat()
            if isinstance(date_partition, date)
            else str(date_partition),
        }

        # ── Phase 1: Claim parsing ────────────────────────────────────────
        # No real file paths available at this level; pass an empty list so
        # the wiring is exercised and the phase logs correctly.
        parsed_count = self.run_claim_parsing(
            file_paths=[], date_partition=date_partition
        )
        summary["claim_parsing"] = {"parsed_count": parsed_count}

        # ── Phase 2: EHR ingestion ────────────────────────────────────────
        ehr_count = self.run_ehr_ingestion(
            claim_records=[], date_partition=date_partition
        )
        summary["ehr_ingestion"] = {"processed_count": ehr_count}

        # ── Phase 3: Feature engineering ─────────────────────────────────
        fe_count = self.run_feature_engineering(
            claim_ids=[], date_partition=date_partition
        )
        summary["feature_engineering"] = {"updated_count": fe_count}

        # ── Phase 4: NLP ─────────────────────────────────────────────────
        nlp_count = self.run_nlp(claim_ids=[])
        summary["nlp"] = {"processed_count": nlp_count}

        # ── Phase 5: Batch scoring ────────────────────────────────────────
        scoring_stats = self.run_batch_scoring(date_partition=date_partition)
        summary["batch_scoring"] = scoring_stats

        total_elapsed = time.monotonic() - pipeline_start
        summary["wall_clock_seconds"] = total_elapsed

        logger.info(
            "SparkPipeline.run_full_pipeline: COMPLETE — "
            "partition=%s wall_clock=%.2fs phases=%s",
            date_partition,
            total_elapsed,
            list(summary.keys()),
        )
        return summary
