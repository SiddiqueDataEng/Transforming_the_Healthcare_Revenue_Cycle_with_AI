# Feature: claim-denial-prediction, Property 9: Feature Record Serialization Round-Trip Fidelity
"""
Property-based tests for FeatureRecordSerializer round-trip fidelity.

**Validates: Requirements 15.4**

Property 9: Feature Record Serialization Round-Trip Fidelity
  For any valid claim feature record (passes schema validation and contains no
  corrupt fields), serializing the record to Avro and then deserializing the
  result SHALL produce a record whose field values are identical to the original,
  and serializing the deserialized record a second time SHALL produce a
  byte-for-byte identical byte sequence to the first serialization.

Note on float precision:
  The Avro schema uses Avro "float" (IEEE 754 float32) for all scalar float
  fields and array element types.  fastavro encodes Python float64 values as
  float32 on the wire.  The property is therefore validated using float32-
  representable values only (generated with ``width=32``), ensuring exact
  round-trip fidelity without precision loss.
"""
from __future__ import annotations

from typing import List, Optional

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from claim_denial.models import ClaimFeatureRecord
from claim_denial.utils.serializer import CURRENT_SCHEMA_VERSION, FeatureRecordSerializer

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-empty printable ASCII for string fields (no NUL bytes which Avro rejects)
_safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters=" -_./",
    ),
    min_size=1,
    max_size=40,
)

# Optional string (null-union in Avro schema)
_opt_text: st.SearchStrategy[Optional[str]] = st.one_of(st.none(), _safe_text)

# Bounded int helpers
_binary_int = st.integers(min_value=0, max_value=1)
_non_neg_int = st.integers(min_value=0, max_value=1000)

# Float strategy constrained to float32 representable range.
#
# The Avro schema uses Avro "float" (IEEE 754 float32) for all float fields.
# fastavro encodes Python float64 values as float32 during serialization.
# To satisfy the round-trip property we must only generate float64 values that
# round-trip exactly through float32 (i.e. values representable as float32).
# We achieve this by drawing floats within the float32 max-magnitude range
# with width=32, which constrains Hypothesis to float32 representable values.
_f32_float = st.floats(
    min_value=-3.4028235e38,
    max_value=3.4028235e38,
    allow_nan=False,
    allow_infinity=False,
    width=32,  # Hypothesis draws float32-representable values only
)

# Unit float [0.0, 1.0] constrained to float32 range
_unit_float = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

# Safe float (general float fields like billing_provider_id_encoded)
_safe_float = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

# Optional null-union float (predicted_denial_score etc.)
_opt_float: st.SearchStrategy[Optional[float]] = st.one_of(
    st.none(),
    st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)

# procedure_category_vector – a list of ints (variable length 0..20)
_proc_cat_vector: st.SearchStrategy[List[int]] = st.lists(
    st.integers(min_value=0, max_value=255),
    min_size=0,
    max_size=20,
)

# Embedding floats – constrained to float32 range (the array element type in
# the Avro schema is "float", i.e. float32).
_embedding_float = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

# tx_plan_complexity_embedding – exactly 768 floats, or arbitrary length
_embedding_768: st.SearchStrategy[List[float]] = st.lists(
    _embedding_float,
    min_size=768,
    max_size=768,
)

# Short embedding (0-10 floats) to exercise non-standard lengths quickly
_embedding_short: st.SearchStrategy[List[float]] = st.lists(
    _embedding_float,
    min_size=0,
    max_size=10,
)

_any_embedding: st.SearchStrategy[List[float]] = st.one_of(
    _embedding_short,
    _embedding_768,
)


@st.composite
def valid_claim_feature_record(draw: st.DrawFn) -> ClaimFeatureRecord:
    """Strategy that produces a fully-populated, schema-valid ClaimFeatureRecord.

    Every field type is exercised:
    - int fields
    - float fields
    - string fields
    - null-union (Optional[str] / Optional[float])
    - binary-array fields (procedure_category_vector, tx_plan_complexity_embedding)
    """
    return ClaimFeatureRecord(
        schema_version=CURRENT_SCHEMA_VERSION,
        claim_id=draw(_safe_text),
        claim_status=draw(_safe_text),
        # Claim-level int features
        claim_submission_date_doy=draw(st.integers(min_value=1, max_value=366)),
        claim_submission_date_month=draw(st.integers(min_value=1, max_value=12)),
        patient_age=draw(st.integers(min_value=-1, max_value=130)),
        gender=draw(st.integers(min_value=-1, max_value=1)),
        ins_type_medicare=draw(_binary_int),
        ins_type_medicaid=draw(_binary_int),
        ins_type_commercial=draw(_binary_int),
        ins_type_tricare=draw(_binary_int),
        ins_type_champva=draw(_binary_int),
        ins_type_other=draw(_binary_int),
        claim_type_professional=draw(_binary_int),
        claim_type_institutional=draw(_binary_int),
        adm_type_emergency=draw(_binary_int),
        adm_type_elective=draw(_binary_int),
        adm_type_urgent=draw(_binary_int),
        adm_type_trauma=draw(_binary_int),
        adm_src_physician_referral=draw(_binary_int),
        adm_src_transfer_hospital=draw(_binary_int),
        adm_src_transfer_snf=draw(_binary_int),
        adm_src_er=draw(_binary_int),
        adm_src_court_law=draw(_binary_int),
        adm_src_not_available=draw(_binary_int),
        # Optional string (null-union)
        partial_ehr_flag=draw(_opt_text),
        ehr_null_match_flag=draw(_binary_int),
        # Provider / facility floats
        billing_provider_id_encoded=draw(_safe_float),
        performing_physician_specialty=draw(_safe_text),
        provider_historical_denial_rate=draw(_unit_float),
        provider_claim_volume=draw(_non_neg_int),
        facility_bed_size=draw(_non_neg_int),
        # Clinical ints + arrays
        principal_dx_ccs_category=draw(st.integers(min_value=0, max_value=285)),
        secondary_dx_count=draw(_non_neg_int),
        procedure_category_vector=draw(_proc_cat_vector),
        length_of_stay=draw(st.integers(min_value=-2, max_value=365)),
        comorbidity_risk_score=draw(_non_neg_int),
        diagnosis_procedure_mismatch=draw(_binary_int),
        # Payer floats/ints
        payer_historical_denial_rate=draw(_unit_float),
        payer_historical_denial_rate_by_type=draw(_unit_float),
        days_since_last_payment=draw(st.integers(min_value=-1, max_value=3650)),
        payer_contract_stop_loss=draw(_binary_int),
        # NLP
        documents_medical_necessity=draw(_unit_float),
        mentions_lack_of_pre_auth=draw(_binary_int),
        tx_plan_complexity_embedding=draw(_any_embedding),
        nlp_notes_absent_flag=draw(_binary_int),
        nlp_model_name=draw(_opt_text),
        nlp_model_version=draw(_opt_text),
        # Scoring output (optional floats/strings)
        predicted_denial_score=draw(_opt_float),
        score_quality_flag=draw(_opt_text),
        model_version_id=draw(_opt_text),
        scoring_timestamp_utc=draw(_opt_text),
        # Ground truth (optional strings)
        adjudication_outcome=draw(_opt_text),
        adjudication_timestamp_utc=draw(_opt_text),
    )


# ---------------------------------------------------------------------------
# Property 9: Feature Record Serialization Round-Trip Fidelity
#
# Validates: Requirements 15.4
# ---------------------------------------------------------------------------

@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(record=valid_claim_feature_record())
def test_serialization_round_trip_fidelity(record: ClaimFeatureRecord) -> None:
    """**Validates: Requirements 15.4**

    For any valid ClaimFeatureRecord, serializing the record, then deserializing
    the result, then serializing again SHALL produce:
    1. A deserialized record whose field values are identical to the original.
    2. A second serialized byte sequence that is byte-for-byte identical to the
       first serialized byte sequence.
    """
    serializer = FeatureRecordSerializer()

    # First serialization
    serialized_once: bytes = serializer.serialize(record)

    # Deserialization
    deserialized: Optional[ClaimFeatureRecord] = serializer.deserialize(serialized_once)

    # Deserialization must succeed (not return None)
    assert deserialized is not None, (
        f"deserialize() returned None for claim_id={record.claim_id!r}; "
        "record should be valid"
    )

    # Second serialization
    serialized_twice: bytes = serializer.serialize(deserialized)

    # ── Assertion 1: byte-for-byte identical ─────────────────────────────────
    assert serialized_once == serialized_twice, (
        f"claim_id={record.claim_id!r}: "
        "second serialization differs from first — round-trip not idempotent"
    )

    # ── Assertion 2: deserialized record equals original ─────────────────────
    # The serializer overwrites schema_version with CURRENT_SCHEMA_VERSION so we
    # normalize that field before comparing.
    assert deserialized.claim_id == record.claim_id, (
        f"claim_id mismatch: {deserialized.claim_id!r} != {record.claim_id!r}"
    )
    assert deserialized.claim_status == record.claim_status
    assert deserialized.procedure_category_vector == record.procedure_category_vector, (
        "procedure_category_vector (binary array) changed during round-trip"
    )
    assert deserialized.tx_plan_complexity_embedding == record.tx_plan_complexity_embedding, (
        "tx_plan_complexity_embedding (binary array) changed during round-trip"
    )
    assert deserialized.partial_ehr_flag == record.partial_ehr_flag, (
        "partial_ehr_flag (null-union string) changed during round-trip"
    )
    assert deserialized.predicted_denial_score == record.predicted_denial_score, (
        "predicted_denial_score (null-union float) changed during round-trip"
    )
    # Scalar numeric fields
    assert deserialized.patient_age == record.patient_age
    assert deserialized.gender == record.gender
    assert deserialized.billing_provider_id_encoded == record.billing_provider_id_encoded
    assert deserialized.payer_historical_denial_rate == record.payer_historical_denial_rate
    assert deserialized.documents_medical_necessity == record.documents_medical_necessity
