"""
Unit tests for FeatureRecordSerializer and SchemaRegistry.

Requirements: 15.1, 15.2, 15.3, 15.5
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import pytest

from claim_denial.models import ClaimFeatureRecord
from claim_denial.schemas.registry import NoMigrationRuleError, SchemaRegistry
from claim_denial.utils.serializer import (
    CURRENT_SCHEMA_VERSION,
    FeatureRecordSerializer,
    QuarantineError,
)


# ---------------------------------------------------------------------------
# Helper: minimal ClaimFeatureRecord factories
# ---------------------------------------------------------------------------

def _make_record(
    claim_id: str = "CLM-001",
    schema_version: str = CURRENT_SCHEMA_VERSION,
    **overrides,
) -> ClaimFeatureRecord:
    """Return a minimal but fully-typed ClaimFeatureRecord."""
    defaults: dict = dict(
        schema_version=schema_version,
        claim_id=claim_id,
        claim_status="Active",
        claim_submission_date_doy=180,
        claim_submission_date_month=6,
        patient_age=45,
        gender=0,
        ins_type_medicare=1,
        ins_type_medicaid=0,
        ins_type_commercial=0,
        ins_type_tricare=0,
        ins_type_champva=0,
        ins_type_other=0,
        claim_type_professional=1,
        claim_type_institutional=0,
        adm_type_emergency=0,
        adm_type_elective=1,
        adm_type_urgent=0,
        adm_type_trauma=0,
        adm_src_physician_referral=1,
        adm_src_transfer_hospital=0,
        adm_src_transfer_snf=0,
        adm_src_er=0,
        adm_src_court_law=0,
        adm_src_not_available=0,
        partial_ehr_flag=None,
        ehr_null_match_flag=0,
        billing_provider_id_encoded=0.42,
        performing_physician_specialty="Cardiology",
        provider_historical_denial_rate=0.1,
        provider_claim_volume=200,
        facility_bed_size=300,
        principal_dx_ccs_category=100,
        secondary_dx_count=3,
        procedure_category_vector=[1, 2, 3],
        length_of_stay=5,
        comorbidity_risk_score=2,
        diagnosis_procedure_mismatch=0,
        payer_historical_denial_rate=0.15,
        payer_historical_denial_rate_by_type=0.12,
        days_since_last_payment=30,
        payer_contract_stop_loss=0,
        documents_medical_necessity=0.8,
        mentions_lack_of_pre_auth=0,
        tx_plan_complexity_embedding=[0.1, -0.2, 0.3] * 256,  # 768 floats
        nlp_notes_absent_flag=0,
        nlp_model_name="clinical-bert",
        nlp_model_version="1.0",
        predicted_denial_score=0.72,
        score_quality_flag=None,
        model_version_id="v1.2.3",
        scoring_timestamp_utc="2024-06-15T03:00:00Z",
        adjudication_outcome=None,
        adjudication_timestamp_utc=None,
    )
    defaults.update(overrides)
    return ClaimFeatureRecord(**defaults)


# ---------------------------------------------------------------------------
# 1. Round-trip for each field type
# ---------------------------------------------------------------------------

class TestRoundTripPerFieldType:
    """Verify serialize → deserialize preserves every field type."""

    def test_roundtrip_int_fields(self) -> None:
        """Requirement 15.1, 15.2 — int fields survive round-trip intact."""
        record = _make_record(patient_age=72, secondary_dx_count=7, length_of_stay=12)
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.patient_age == 72
        assert result.secondary_dx_count == 7
        assert result.length_of_stay == 12

    def test_roundtrip_float_fields(self) -> None:
        """Requirement 15.1, 15.2 — float fields survive round-trip intact."""
        record = _make_record(
            billing_provider_id_encoded=3.14159,
            provider_historical_denial_rate=0.333,
            documents_medical_necessity=0.999,
        )
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.billing_provider_id_encoded == pytest.approx(3.14159, abs=1e-5)
        assert result.provider_historical_denial_rate == pytest.approx(0.333, abs=1e-5)
        assert result.documents_medical_necessity == pytest.approx(0.999, abs=1e-5)

    def test_roundtrip_string_fields(self) -> None:
        """Requirement 15.1, 15.2 — string fields survive round-trip intact."""
        record = _make_record(
            claim_status="Pending",
            performing_physician_specialty="Neurology",
            nlp_model_name="bert-base",
            nlp_model_version="2.1",
        )
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.claim_status == "Pending"
        assert result.performing_physician_specialty == "Neurology"
        assert result.nlp_model_name == "bert-base"
        assert result.nlp_model_version == "2.1"

    def test_roundtrip_null_union_string_none(self) -> None:
        """Requirement 15.1, 15.2 — null-union string None value round-trips to None."""
        record = _make_record(partial_ehr_flag=None, score_quality_flag=None, model_version_id=None)
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.partial_ehr_flag is None
        assert result.score_quality_flag is None
        assert result.model_version_id is None

    def test_roundtrip_null_union_string_value(self) -> None:
        """Requirement 15.1, 15.2 — null-union string non-null value round-trips intact."""
        record = _make_record(
            partial_ehr_flag="vitals,labs",
            score_quality_flag="degraded",
        )
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.partial_ehr_flag == "vitals,labs"
        assert result.score_quality_flag == "degraded"

    def test_roundtrip_null_union_float_none(self) -> None:
        """Requirement 15.1, 15.2 — null-union float None value round-trips to None."""
        record = _make_record(predicted_denial_score=None)
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.predicted_denial_score is None

    def test_roundtrip_null_union_float_value(self) -> None:
        """Requirement 15.1, 15.2 — null-union float non-null value round-trips intact."""
        record = _make_record(predicted_denial_score=0.87)
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.predicted_denial_score == pytest.approx(0.87, abs=1e-5)

    def test_roundtrip_procedure_category_vector(self) -> None:
        """Requirement 15.1, 15.2 — procedure_category_vector (int array) survives round-trip."""
        proc_vector = [10, 20, 30, 99, 0]
        record = _make_record(procedure_category_vector=proc_vector)
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.procedure_category_vector == proc_vector

    def test_roundtrip_procedure_category_vector_empty(self) -> None:
        """Requirement 15.1, 15.2 — empty procedure_category_vector round-trips to empty list."""
        record = _make_record(procedure_category_vector=[])
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.procedure_category_vector == []

    def test_roundtrip_tx_plan_complexity_embedding_768(self) -> None:
        """Requirement 15.1, 15.2 — tx_plan_complexity_embedding (768 float array) survives round-trip."""
        embedding = [float(i) / 768 for i in range(768)]
        record = _make_record(tx_plan_complexity_embedding=embedding)
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert len(result.tx_plan_complexity_embedding) == 768
        # Check a sample of values (float32 precision)
        for i in [0, 100, 500, 767]:
            assert result.tx_plan_complexity_embedding[i] == pytest.approx(embedding[i], abs=1e-5), (
                f"embedding[{i}] mismatch: {result.tx_plan_complexity_embedding[i]} != {embedding[i]}"
            )

    def test_roundtrip_tx_plan_complexity_embedding_zeros(self) -> None:
        """Requirement 15.1, 15.2 — zero-filled embedding round-trips correctly."""
        embedding = [0.0] * 768
        record = _make_record(tx_plan_complexity_embedding=embedding)
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.tx_plan_complexity_embedding == embedding

    def test_schema_version_is_embedded_as_current(self) -> None:
        """Requirement 15.1 — serializer always embeds CURRENT_SCHEMA_VERSION."""
        # Even if the record was constructed with an older version, serializer stamps current
        record = _make_record(schema_version="0.9.0")
        serializer = FeatureRecordSerializer()
        result = serializer.deserialize(serializer.serialize(record))
        assert result is not None
        assert result.schema_version == CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 2. Schema migration: version "0.9.0" → "1.0.0"
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    """Test SchemaRegistry-driven migration during deserialization."""

    def _make_serializer_with_migration(self) -> FeatureRecordSerializer:
        """Build a serializer whose registry has a 0.9.0 → 1.0.0 migration rule."""
        registry = SchemaRegistry()

        def migrate_090_to_100(record_dict: dict) -> dict:
            """Simulated migration: rename a legacy field and update schema_version."""
            result = dict(record_dict)
            result["schema_version"] = "1.0.0"
            # Example migration: add a new field default if absent
            result.setdefault("payer_contract_stop_loss", 0)
            return result

        registry.register("0.9.0", "1.0.0", migrate_090_to_100)
        return FeatureRecordSerializer(registry=registry, current_version="1.0.0")

    def _serialize_as_old_version(self) -> bytes:
        """Serialize a record stamped as 0.9.0 using a 0.9.0 serializer instance."""
        old_serializer = FeatureRecordSerializer(current_version="0.9.0")
        record = _make_record(schema_version="0.9.0")
        return old_serializer.serialize(record)

    def test_migration_090_to_100_applies_correctly(self) -> None:
        """Requirement 15.5 — migration rule 0.9.0 → 1.0.0 is applied and record is valid."""
        data = self._serialize_as_old_version()
        serializer = self._make_serializer_with_migration()
        result = serializer.deserialize(data)
        assert result is not None, "Deserialization should succeed after migration"
        assert result.schema_version == "1.0.0", (
            f"Expected schema_version=1.0.0 after migration, got {result.schema_version!r}"
        )

    def test_migration_result_is_valid_record(self) -> None:
        """Requirement 15.5 — migrated record has all expected fields."""
        data = self._serialize_as_old_version()
        serializer = self._make_serializer_with_migration()
        result = serializer.deserialize(data)
        assert result is not None
        assert result.claim_id == "CLM-001"
        assert isinstance(result.patient_age, int)
        assert isinstance(result.payer_historical_denial_rate, float)


# ---------------------------------------------------------------------------
# 3. Missing migration rule → QuarantineError
# ---------------------------------------------------------------------------

class TestMissingMigrationRule:
    """Records with unknown versions and no migration rule must raise QuarantineError."""

    def test_missing_migration_rule_raises_quarantine_error(self) -> None:
        """Requirement 15.5 — unknown version + no rule → QuarantineError raised."""
        # Serialize with an unknown old version
        old_serializer = FeatureRecordSerializer(current_version="0.5.0")
        record = _make_record(schema_version="0.5.0", claim_id="CLM-QUANTINE")
        data = old_serializer.serialize(record)

        # Deserializer has no migration rule for 0.5.0 → 1.0.0
        serializer = FeatureRecordSerializer(registry=SchemaRegistry(), current_version="1.0.0")
        with pytest.raises(QuarantineError) as exc_info:
            serializer.deserialize(data)

        err = exc_info.value
        assert err.embedded_version == "0.5.0", (
            f"Expected embedded_version='0.5.0', got {err.embedded_version!r}"
        )
        assert err.current_version == "1.0.0", (
            f"Expected current_version='1.0.0', got {err.current_version!r}"
        )
        assert err.claim_id == "CLM-QUANTINE", (
            f"Expected claim_id='CLM-QUANTINE', got {err.claim_id!r}"
        )

    def test_missing_migration_logs_schema_mismatch_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 15.5 — schema-mismatch error is logged with correct versions."""
        old_serializer = FeatureRecordSerializer(current_version="0.5.0")
        record = _make_record(schema_version="0.5.0", claim_id="CLM-LOG-001")
        data = old_serializer.serialize(record)

        serializer = FeatureRecordSerializer(registry=SchemaRegistry(), current_version="1.0.0")

        with caplog.at_level(logging.ERROR, logger="claim_denial.utils.serializer"):
            with pytest.raises(QuarantineError):
                serializer.deserialize(data)

        # Exactly one error log should mention the version mismatch
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1, "Expected at least one ERROR log entry"

        combined = " ".join(r.getMessage() for r in error_records)
        assert "0.5.0" in combined, (
            f"Embedded version '0.5.0' not found in log: {combined!r}"
        )
        assert "1.0.0" in combined, (
            f"Current version '1.0.0' not found in log: {combined!r}"
        )
        assert "CLM-LOG-001" in combined or "schema" in combined.lower(), (
            f"Expected claim ID or schema-mismatch indication in log: {combined!r}"
        )

    def test_schema_registry_raises_no_migration_rule_error_directly(self) -> None:
        """SchemaRegistry.apply_migrations raises NoMigrationRuleError when no path exists."""
        registry = SchemaRegistry()
        record_dict = {"schema_version": "0.5.0", "claim_id": "X"}

        with pytest.raises(NoMigrationRuleError) as exc_info:
            registry.apply_migrations(record_dict, "0.5.0", "1.0.0")

        assert exc_info.value.from_version == "0.5.0"
        assert exc_info.value.to_version == "1.0.0"


# ---------------------------------------------------------------------------
# 4. pretty_print: valid JSON with correct field names and types
# ---------------------------------------------------------------------------

class TestPrettyPrint:
    """FeatureRecordSerializer.pretty_print must return valid JSON."""

    def test_pretty_print_returns_valid_json(self) -> None:
        """Requirement 15.3 — pretty_print output is parseable JSON."""
        record = _make_record()
        serializer = FeatureRecordSerializer()
        output = serializer.pretty_print(record)

        # Must parse without error
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_pretty_print_contains_expected_field_names(self) -> None:
        """Requirement 15.3 — JSON contains key field names."""
        record = _make_record(claim_id="CLM-PRINT", patient_age=55)
        serializer = FeatureRecordSerializer()
        parsed = json.loads(serializer.pretty_print(record))

        assert "claim_id" in parsed
        assert "schema_version" in parsed
        assert "patient_age" in parsed
        assert "procedure_category_vector" in parsed
        assert "tx_plan_complexity_embedding" in parsed

    def test_pretty_print_field_values_match_record(self) -> None:
        """Requirement 15.3 — JSON values match the ClaimFeatureRecord fields."""
        record = _make_record(
            claim_id="CLM-VAL",
            patient_age=63,
            partial_ehr_flag="labs",
            predicted_denial_score=0.55,
        )
        serializer = FeatureRecordSerializer()
        parsed = json.loads(serializer.pretty_print(record))

        assert parsed["claim_id"] == "CLM-VAL"
        assert parsed["patient_age"] == 63
        assert parsed["partial_ehr_flag"] == "labs"
        assert abs(parsed["predicted_denial_score"] - 0.55) < 1e-5

    def test_pretty_print_null_fields_appear_as_null(self) -> None:
        """Requirement 15.3 — null optional fields appear as JSON null."""
        record = _make_record(partial_ehr_flag=None, adjudication_outcome=None)
        serializer = FeatureRecordSerializer()
        parsed = json.loads(serializer.pretty_print(record))

        assert parsed["partial_ehr_flag"] is None
        assert parsed["adjudication_outcome"] is None

    def test_pretty_print_array_fields_are_lists(self) -> None:
        """Requirement 15.3 — array fields appear as JSON arrays with correct types."""
        proc_vec = [5, 10, 15]
        embedding = [0.1, 0.2, 0.3]
        record = _make_record(
            procedure_category_vector=proc_vec,
            tx_plan_complexity_embedding=embedding,
        )
        serializer = FeatureRecordSerializer()
        parsed = json.loads(serializer.pretty_print(record))

        assert isinstance(parsed["procedure_category_vector"], list)
        assert parsed["procedure_category_vector"] == proc_vec
        assert isinstance(parsed["tx_plan_complexity_embedding"], list)
        assert len(parsed["tx_plan_complexity_embedding"]) == len(embedding)

    def test_pretty_print_indentation(self) -> None:
        """Requirement 15.3 — output is indented (human-readable)."""
        record = _make_record()
        serializer = FeatureRecordSerializer()
        output = serializer.pretty_print(record)
        # Indented JSON has newlines and leading spaces
        assert "\n" in output
        assert "  " in output

    def test_pretty_print_float_type_fields(self) -> None:
        """Requirement 15.3 — float fields have numeric types in JSON."""
        record = _make_record(
            billing_provider_id_encoded=1.23,
            payer_historical_denial_rate=0.18,
        )
        serializer = FeatureRecordSerializer()
        parsed = json.loads(serializer.pretty_print(record))
        assert isinstance(parsed["billing_provider_id_encoded"], float)
        assert isinstance(parsed["payer_historical_denial_rate"], float)

    def test_pretty_print_int_type_fields(self) -> None:
        """Requirement 15.3 — int fields have integer types in JSON."""
        record = _make_record(patient_age=40, secondary_dx_count=2)
        serializer = FeatureRecordSerializer()
        parsed = json.loads(serializer.pretty_print(record))
        assert isinstance(parsed["patient_age"], int)
        assert isinstance(parsed["secondary_dx_count"], int)
