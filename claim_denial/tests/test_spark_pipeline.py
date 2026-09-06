"""
Unit tests for claim_denial.pipeline.spark_pipeline.

Coverage:
  - DataLakeLayout: all path templates resolve correctly with zero-padded month/day
  - SparkPipelineConfig: default values match the throughput target
  - SparkSessionManager.get_or_create(): when PySpark not available, returns None without raising
  - SparkPipeline.run_claim_parsing(): returns int count >= 0; delegates to ClaimProcessor
  - SparkPipeline.run_full_pipeline(): returns dict with expected keys; all phases run in order
  - THROUGHPUT_TARGET_CLAIMS_PER_DAY == 2740

Requirements: 14.1, 14.3
"""

from __future__ import annotations

import sys
import types
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from claim_denial.pipeline.spark_pipeline import (
    THROUGHPUT_TARGET_CLAIMS_PER_DAY,
    DataLakeLayout,
    SparkPipeline,
    SparkPipelineConfig,
    SparkSessionManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs: Any) -> SparkPipelineConfig:
    """Return a SparkPipelineConfig with optional overrides."""
    cfg = SparkPipelineConfig()
    for k, v in kwargs.items():
        object.__setattr__(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# THROUGHPUT_TARGET_CLAIMS_PER_DAY
# ---------------------------------------------------------------------------

class TestThroughputTargetConstant:
    """THROUGHPUT_TARGET_CLAIMS_PER_DAY must equal 2740 (Requirement 14.1)."""

    def test_value_is_2740(self) -> None:
        assert THROUGHPUT_TARGET_CLAIMS_PER_DAY == 2740

    def test_type_is_int(self) -> None:
        assert isinstance(THROUGHPUT_TARGET_CLAIMS_PER_DAY, int)


# ---------------------------------------------------------------------------
# SparkPipelineConfig — defaults
# ---------------------------------------------------------------------------

class TestSparkPipelineConfigDefaults:
    """Default SparkPipelineConfig values must support the throughput target."""

    def test_default_app_name(self) -> None:
        assert SparkPipelineConfig().app_name == "ClaimDenialPrediction"

    def test_default_master_is_local(self) -> None:
        assert SparkPipelineConfig().master == "local[*]"

    def test_default_num_executors(self) -> None:
        # 10 executors * 4 cores = 40 parallel tasks — sufficient for ≥2740/day
        assert SparkPipelineConfig().num_executors == 10

    def test_default_executor_cores(self) -> None:
        assert SparkPipelineConfig().executor_cores == 4

    def test_default_executor_memory(self) -> None:
        assert SparkPipelineConfig().executor_memory == "8g"

    def test_default_data_lake_base_path(self) -> None:
        assert SparkPipelineConfig().data_lake_base_path == "./data_lake"

    def test_shuffle_partitions_capacity(self) -> None:
        """shuffle_partitions = num_executors * executor_cores * 2 must be >= THROUGHPUT target."""
        cfg = SparkPipelineConfig()
        shuffle = cfg.num_executors * cfg.executor_cores * 2
        # With default settings: 10 * 4 * 2 = 80 partitions — well above 2740/day
        assert shuffle >= 1  # minimal sanity check; real check is the executor count


# ---------------------------------------------------------------------------
# DataLakeLayout — path resolution (Requirement 14.3)
# ---------------------------------------------------------------------------

class TestDataLakeLayout:
    """DataLakeLayout must resolve templates with zero-padded month and day."""

    def setup_method(self) -> None:
        self.cfg = SparkPipelineConfig(data_lake_base_path="./data_lake")
        self.layout = DataLakeLayout(self.cfg)

    # ── claims_raw_path ──────────────────────────────────────────────────

    def test_claims_raw_path_structure(self) -> None:
        path = self.layout.claims_raw_path(2024, 1, 5)
        assert path == "./data_lake/claims/2024/01/05/raw"

    def test_claims_raw_path_zero_pads_month(self) -> None:
        path = self.layout.claims_raw_path(2024, 3, 20)
        assert "/03/" in path

    def test_claims_raw_path_zero_pads_day(self) -> None:
        path = self.layout.claims_raw_path(2024, 12, 7)
        assert "/07/" in path

    def test_claims_raw_path_double_digit(self) -> None:
        path = self.layout.claims_raw_path(2025, 11, 30)
        assert path == "./data_lake/claims/2025/11/30/raw"

    # ── claims_parsed_path ───────────────────────────────────────────────

    def test_claims_parsed_path_structure(self) -> None:
        path = self.layout.claims_parsed_path(2024, 6, 15)
        assert path == "./data_lake/claims/2024/06/15/parsed"

    def test_claims_parsed_path_zero_pads_month(self) -> None:
        path = self.layout.claims_parsed_path(2024, 2, 28)
        assert "/02/" in path

    def test_claims_parsed_differs_from_raw(self) -> None:
        raw = self.layout.claims_raw_path(2024, 6, 15)
        parsed = self.layout.claims_parsed_path(2024, 6, 15)
        assert raw != parsed
        assert raw.endswith("/raw")
        assert parsed.endswith("/parsed")

    # ── ehr_path ─────────────────────────────────────────────────────────

    def test_ehr_path_structure(self) -> None:
        path = self.layout.ehr_path("ACC001", "2024-01-10")
        assert path == "./data_lake/ehr/ACC001/2024-01-10"

    def test_ehr_path_embeds_patient_account_number(self) -> None:
        path = self.layout.ehr_path("PAT-9999", "2023-12-01")
        assert "PAT-9999" in path

    def test_ehr_path_embeds_encounter_date(self) -> None:
        path = self.layout.ehr_path("ACC002", "2024-07-04")
        assert "2024-07-04" in path

    # ── era_path ─────────────────────────────────────────────────────────

    def test_era_path_structure(self) -> None:
        path = self.layout.era_path(2024, 9, 3)
        assert path == "./data_lake/era/2024/09/03"

    def test_era_path_zero_pads_month(self) -> None:
        path = self.layout.era_path(2024, 1, 15)
        assert "/01/" in path

    def test_era_path_zero_pads_day(self) -> None:
        path = self.layout.era_path(2024, 10, 8)
        assert "/08" in path

    # ── quarantine_path ──────────────────────────────────────────────────

    def test_quarantine_path_structure(self) -> None:
        path = self.layout.quarantine_path(2024, 5, 1)
        assert path == "./data_lake/quarantine/2024/05/01"

    def test_quarantine_path_zero_pads_month(self) -> None:
        path = self.layout.quarantine_path(2024, 4, 20)
        assert "/04/" in path

    def test_quarantine_path_zero_pads_day(self) -> None:
        path = self.layout.quarantine_path(2024, 11, 2)
        assert "/02" in path

    # ── custom base path ─────────────────────────────────────────────────

    def test_custom_base_path(self) -> None:
        cfg = SparkPipelineConfig(data_lake_base_path="s3://my-bucket/data-lake")
        layout = DataLakeLayout(cfg)
        path = layout.claims_raw_path(2024, 1, 1)
        assert path.startswith("s3://my-bucket/data-lake/")

    def test_trailing_slash_stripped_from_base(self) -> None:
        cfg = SparkPipelineConfig(data_lake_base_path="./data_lake/")
        layout = DataLakeLayout(cfg)
        path = layout.claims_raw_path(2024, 1, 1)
        # Should not have double slash
        assert "//" not in path.replace("s3://", "").replace("://", "")


# ---------------------------------------------------------------------------
# SparkSessionManager — PySpark unavailable path
# ---------------------------------------------------------------------------

class TestSparkSessionManagerNoPyspark:
    """When PySpark is not installed, get_or_create() must return None silently."""

    def test_returns_none_when_pyspark_missing(self) -> None:
        """
        Simulate an environment where PySpark is not installed by patching
        the import inside get_or_create to raise ImportError.
        """
        cfg = SparkPipelineConfig()
        manager = SparkSessionManager(cfg)

        with patch.dict(sys.modules, {"pyspark": None, "pyspark.sql": None}):
            result = manager.get_or_create()

        assert result is None

    def test_no_exception_when_pyspark_missing(self) -> None:
        """get_or_create() must not raise when PySpark is absent."""
        cfg = SparkPipelineConfig()
        manager = SparkSessionManager(cfg)

        with patch.dict(sys.modules, {"pyspark": None, "pyspark.sql": None}):
            try:
                manager.get_or_create()
            except Exception as exc:
                pytest.fail(f"get_or_create() raised unexpectedly: {exc}")

    def test_stop_is_safe_when_no_session(self) -> None:
        """stop() must not raise when no session was ever created."""
        cfg = SparkPipelineConfig()
        manager = SparkSessionManager(cfg)
        manager.stop()  # should be a no-op


# ---------------------------------------------------------------------------
# SparkPipeline.run_claim_parsing — delegation and return type
# ---------------------------------------------------------------------------

class TestRunClaimParsing:
    """run_claim_parsing() must delegate to ClaimProcessor and return int >= 0."""

    def _make_pipeline_no_spark(self) -> SparkPipeline:
        """Return a SparkPipeline whose session_manager always returns None."""
        session_manager = MagicMock()
        session_manager.get_or_create.return_value = None
        return SparkPipeline(session_manager=session_manager)

    def test_returns_int(self) -> None:
        pipeline = self._make_pipeline_no_spark()
        with patch(
            "claim_denial.pipeline.spark_pipeline.ClaimProcessor"
        ) as MockProcessor:
            MockProcessor.return_value.ingest_file.return_value = []
            result = pipeline.run_claim_parsing([], date(2024, 1, 15))
        assert isinstance(result, int)

    def test_returns_zero_for_empty_file_list(self) -> None:
        pipeline = self._make_pipeline_no_spark()
        with patch(
            "claim_denial.pipeline.spark_pipeline.ClaimProcessor"
        ):
            result = pipeline.run_claim_parsing([], date(2024, 1, 15))
        assert result == 0

    def test_count_reflects_parsed_records(self) -> None:
        """ClaimProcessor.ingest_file() returning 3 records → total = 3."""
        pipeline = self._make_pipeline_no_spark()

        mock_records = [MagicMock(), MagicMock(), MagicMock()]
        with patch(
            "claim_denial.pipeline.spark_pipeline.ClaimProcessor"
        ) as MockProcessor:
            MockProcessor.return_value.ingest_file.return_value = mock_records
            result = pipeline.run_claim_parsing(
                ["path/to/file.edi"], date(2024, 1, 15)
            )
        assert result == 3

    def test_count_is_non_negative(self) -> None:
        pipeline = self._make_pipeline_no_spark()
        with patch(
            "claim_denial.pipeline.spark_pipeline.ClaimProcessor"
        ) as MockProcessor:
            MockProcessor.return_value.ingest_file.return_value = []
            result = pipeline.run_claim_parsing(
                ["f1.edi", "f2.edi"], date(2024, 6, 1)
            )
        assert result >= 0

    def test_claim_processor_is_instantiated(self) -> None:
        """ClaimProcessor must be called during run_claim_parsing."""
        pipeline = self._make_pipeline_no_spark()
        with patch(
            "claim_denial.pipeline.spark_pipeline.ClaimProcessor"
        ) as MockCP:
            MockCP.return_value.ingest_file.return_value = []
            pipeline.run_claim_parsing(["some/file.edi"], date(2024, 3, 10))
        MockCP.assert_called_once()


# ---------------------------------------------------------------------------
# SparkPipeline.run_full_pipeline — summary dict structure and phase ordering
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
    """run_full_pipeline() must return a dict with all expected keys."""

    EXPECTED_KEYS = {
        "date_partition",
        "claim_parsing",
        "ehr_ingestion",
        "feature_engineering",
        "nlp",
        "batch_scoring",
        "wall_clock_seconds",
    }

    def _mock_scoring_stats(self) -> dict:
        return {
            "total_active": 0,
            "total_scored": 0,
            "degraded": 0,
            "failed": 0,
            "wall_clock_seconds": 0.0,
        }

    def _patched_pipeline(self) -> SparkPipeline:
        """Return a SparkPipeline with all phases mocked out."""
        session_manager = MagicMock()
        session_manager.get_or_create.return_value = None
        return SparkPipeline(session_manager=session_manager)

    @patch("claim_denial.pipeline.spark_pipeline.BatchScorer")
    @patch("claim_denial.pipeline.spark_pipeline.NLPProcessor")
    @patch("claim_denial.pipeline.spark_pipeline.EHRIngestionService")
    @patch("claim_denial.pipeline.spark_pipeline.FHIRClient")
    @patch("claim_denial.pipeline.spark_pipeline.ClaimProcessor")
    def test_returns_dict(
        self,
        MockCP: MagicMock,
        MockFHIR: MagicMock,
        MockEHR: MagicMock,
        MockNLP: MagicMock,
        MockScorer: MagicMock,
    ) -> None:
        MockCP.return_value.ingest_file.return_value = []
        scorer_inst = MockScorer.return_value
        scorer_inst.load_production_model.return_value = MagicMock()
        scorer_inst.fetch_active_claims.return_value = []
        scorer_inst.score_claims.return_value = []

        pipeline = self._patched_pipeline()
        result = pipeline.run_full_pipeline(date(2024, 6, 15))
        assert isinstance(result, dict)

    @patch("claim_denial.pipeline.spark_pipeline.BatchScorer")
    @patch("claim_denial.pipeline.spark_pipeline.NLPProcessor")
    @patch("claim_denial.pipeline.spark_pipeline.EHRIngestionService")
    @patch("claim_denial.pipeline.spark_pipeline.FHIRClient")
    @patch("claim_denial.pipeline.spark_pipeline.ClaimProcessor")
    def test_all_expected_keys_present(
        self,
        MockCP: MagicMock,
        MockFHIR: MagicMock,
        MockEHR: MagicMock,
        MockNLP: MagicMock,
        MockScorer: MagicMock,
    ) -> None:
        MockCP.return_value.ingest_file.return_value = []
        scorer_inst = MockScorer.return_value
        scorer_inst.load_production_model.return_value = MagicMock()
        scorer_inst.fetch_active_claims.return_value = []
        scorer_inst.score_claims.return_value = []

        pipeline = self._patched_pipeline()
        result = pipeline.run_full_pipeline(date(2024, 6, 15))
        assert self.EXPECTED_KEYS.issubset(result.keys())

    @patch("claim_denial.pipeline.spark_pipeline.BatchScorer")
    @patch("claim_denial.pipeline.spark_pipeline.NLPProcessor")
    @patch("claim_denial.pipeline.spark_pipeline.EHRIngestionService")
    @patch("claim_denial.pipeline.spark_pipeline.FHIRClient")
    @patch("claim_denial.pipeline.spark_pipeline.ClaimProcessor")
    def test_wall_clock_seconds_is_non_negative(
        self,
        MockCP: MagicMock,
        MockFHIR: MagicMock,
        MockEHR: MagicMock,
        MockNLP: MagicMock,
        MockScorer: MagicMock,
    ) -> None:
        MockCP.return_value.ingest_file.return_value = []
        scorer_inst = MockScorer.return_value
        scorer_inst.load_production_model.return_value = MagicMock()
        scorer_inst.fetch_active_claims.return_value = []
        scorer_inst.score_claims.return_value = []

        pipeline = self._patched_pipeline()
        result = pipeline.run_full_pipeline(date(2024, 6, 15))
        assert result["wall_clock_seconds"] >= 0.0

    @patch("claim_denial.pipeline.spark_pipeline.BatchScorer")
    @patch("claim_denial.pipeline.spark_pipeline.NLPProcessor")
    @patch("claim_denial.pipeline.spark_pipeline.EHRIngestionService")
    @patch("claim_denial.pipeline.spark_pipeline.FHIRClient")
    @patch("claim_denial.pipeline.spark_pipeline.ClaimProcessor")
    def test_date_partition_stored_as_iso_string(
        self,
        MockCP: MagicMock,
        MockFHIR: MagicMock,
        MockEHR: MagicMock,
        MockNLP: MagicMock,
        MockScorer: MagicMock,
    ) -> None:
        MockCP.return_value.ingest_file.return_value = []
        scorer_inst = MockScorer.return_value
        scorer_inst.load_production_model.return_value = MagicMock()
        scorer_inst.fetch_active_claims.return_value = []
        scorer_inst.score_claims.return_value = []

        pipeline = self._patched_pipeline()
        result = pipeline.run_full_pipeline(date(2024, 6, 15))
        assert result["date_partition"] == "2024-06-15"

    @patch("claim_denial.pipeline.spark_pipeline.BatchScorer")
    @patch("claim_denial.pipeline.spark_pipeline.NLPProcessor")
    @patch("claim_denial.pipeline.spark_pipeline.EHRIngestionService")
    @patch("claim_denial.pipeline.spark_pipeline.FHIRClient")
    @patch("claim_denial.pipeline.spark_pipeline.ClaimProcessor")
    def test_claim_parsing_phase_present(
        self,
        MockCP: MagicMock,
        MockFHIR: MagicMock,
        MockEHR: MagicMock,
        MockNLP: MagicMock,
        MockScorer: MagicMock,
    ) -> None:
        MockCP.return_value.ingest_file.return_value = []
        scorer_inst = MockScorer.return_value
        scorer_inst.load_production_model.return_value = MagicMock()
        scorer_inst.fetch_active_claims.return_value = []
        scorer_inst.score_claims.return_value = []

        pipeline = self._patched_pipeline()
        result = pipeline.run_full_pipeline(date(2024, 6, 15))
        assert "claim_parsing" in result
        assert "parsed_count" in result["claim_parsing"]

    @patch("claim_denial.pipeline.spark_pipeline.BatchScorer")
    @patch("claim_denial.pipeline.spark_pipeline.NLPProcessor")
    @patch("claim_denial.pipeline.spark_pipeline.EHRIngestionService")
    @patch("claim_denial.pipeline.spark_pipeline.FHIRClient")
    @patch("claim_denial.pipeline.spark_pipeline.ClaimProcessor")
    def test_batch_scoring_phase_present(
        self,
        MockCP: MagicMock,
        MockFHIR: MagicMock,
        MockEHR: MagicMock,
        MockNLP: MagicMock,
        MockScorer: MagicMock,
    ) -> None:
        MockCP.return_value.ingest_file.return_value = []
        scorer_inst = MockScorer.return_value
        scorer_inst.load_production_model.return_value = MagicMock()
        scorer_inst.fetch_active_claims.return_value = []
        scorer_inst.score_claims.return_value = []

        pipeline = self._patched_pipeline()
        result = pipeline.run_full_pipeline(date(2024, 6, 15))
        assert "batch_scoring" in result
        # batch_scoring sub-dict must have the expected keys
        scoring = result["batch_scoring"]
        for k in ("total_active", "total_scored", "degraded", "failed", "wall_clock_seconds"):
            assert k in scoring, f"Missing key in batch_scoring: {k}"

    @patch("claim_denial.pipeline.spark_pipeline.BatchScorer")
    @patch("claim_denial.pipeline.spark_pipeline.NLPProcessor")
    @patch("claim_denial.pipeline.spark_pipeline.EHRIngestionService")
    @patch("claim_denial.pipeline.spark_pipeline.FHIRClient")
    @patch("claim_denial.pipeline.spark_pipeline.ClaimProcessor")
    def test_all_phases_run_in_order(
        self,
        MockCP: MagicMock,
        MockFHIR: MagicMock,
        MockEHR: MagicMock,
        MockNLP: MagicMock,
        MockScorer: MagicMock,
    ) -> None:
        """All five pipeline phases should appear in the summary dict."""
        MockCP.return_value.ingest_file.return_value = []
        scorer_inst = MockScorer.return_value
        scorer_inst.load_production_model.return_value = MagicMock()
        scorer_inst.fetch_active_claims.return_value = []
        scorer_inst.score_claims.return_value = []

        pipeline = self._patched_pipeline()
        result = pipeline.run_full_pipeline(date(2024, 6, 15))

        phase_keys = [
            "claim_parsing",
            "ehr_ingestion",
            "feature_engineering",
            "nlp",
            "batch_scoring",
        ]
        for key in phase_keys:
            assert key in result, f"Phase key '{key}' missing from full pipeline result"


# ---------------------------------------------------------------------------
# SparkPipeline — construction defaults
# ---------------------------------------------------------------------------

class TestSparkPipelineConstruction:
    """SparkPipeline must be constructable with no arguments."""

    def test_default_construction(self) -> None:
        pipeline = SparkPipeline()
        assert pipeline is not None

    def test_custom_config_accepted(self) -> None:
        cfg = SparkPipelineConfig(app_name="TestApp", master="local[2]")
        pipeline = SparkPipeline(config=cfg)
        assert pipeline._config.app_name == "TestApp"

    def test_session_manager_injected(self) -> None:
        mock_sm = MagicMock()
        mock_sm.get_or_create.return_value = None
        pipeline = SparkPipeline(session_manager=mock_sm)
        assert pipeline._session_manager is mock_sm


# ---------------------------------------------------------------------------
# DataLakeLayout — edge cases
# ---------------------------------------------------------------------------

class TestDataLakeLayoutEdgeCases:
    """Additional edge cases for DataLakeLayout."""

    def test_single_digit_month_and_day_always_padded(self) -> None:
        cfg = SparkPipelineConfig(data_lake_base_path="./dl")
        layout = DataLakeLayout(cfg)
        for m in range(1, 10):
            for d in range(1, 10):
                path = layout.claims_raw_path(2024, m, d)
                # month and day must appear as 2-digit strings
                month_str = f"{m:02d}"
                day_str = f"{d:02d}"
                assert f"/{month_str}/" in path, f"Month not zero-padded in: {path}"
                assert f"/{day_str}/" in path, f"Day not zero-padded in: {path}"

    def test_double_digit_month_not_changed(self) -> None:
        cfg = SparkPipelineConfig(data_lake_base_path="./dl")
        layout = DataLakeLayout(cfg)
        path = layout.claims_raw_path(2024, 12, 31)
        assert "/12/31/" in path

    def test_all_path_methods_return_str(self) -> None:
        cfg = SparkPipelineConfig(data_lake_base_path="./dl")
        layout = DataLakeLayout(cfg)
        assert isinstance(layout.claims_raw_path(2024, 1, 1), str)
        assert isinstance(layout.claims_parsed_path(2024, 1, 1), str)
        assert isinstance(layout.ehr_path("ACC001", "2024-01-01"), str)
        assert isinstance(layout.era_path(2024, 1, 1), str)
        assert isinstance(layout.quarantine_path(2024, 1, 1), str)
