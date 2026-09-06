# Feature: claim-denial-prediction, Property 8: One-Hot Encodings Are Mutually Consistent
"""
Property-based tests for ClaimLevelFeatureEngineer one-hot encoding consistency.

**Validates: Requirements 3.5, 3.6, 3.7, 3.8**

Property 8: One-Hot Encodings Are Mutually Consistent
  For any claim, the one-hot encoded vectors for `insurance_type`, `claim_type`,
  `admission_type`, and `admission_source` SHALL each contain only 0 or 1 values;
  the sum of each vector SHALL be either 0 (unrecognized / absent category) or 1
  (exactly one active category).
"""
from __future__ import annotations

from typing import Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from claim_denial.constants import (
    ADMISSION_SOURCES,
    ADMISSION_TYPES,
    CLAIM_TYPES,
    INSURANCE_TYPES,
)
from claim_denial.features.claim_level import ClaimLevelFeatureEngineer

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Recognised category values drawn directly from the constant lists
_known_insurance_types = st.sampled_from(INSURANCE_TYPES)
_known_claim_types = st.sampled_from(CLAIM_TYPES)
_known_admission_types = st.sampled_from(ADMISSION_TYPES)
_known_admission_sources = st.sampled_from(ADMISSION_SOURCES)

# Arbitrary printable strings that are very unlikely to match any known category
_unknown_string = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters=" -_/",
    ),
    min_size=1,
    max_size=50,
).filter(
    lambda s: s.strip().lower() not in {c.lower() for c in INSURANCE_TYPES}
    and s.strip().lower() not in {c.lower() for c in CLAIM_TYPES}
    and s.strip().lower() not in {c.lower() for c in ADMISSION_TYPES}
    and s.strip().lower() not in {c.lower() for c in ADMISSION_SOURCES}
)

# Each field is one of: known category (str), unknown string (str), or None
_insurance_type_strategy: st.SearchStrategy[Optional[str]] = st.one_of(
    st.none(),
    _known_insurance_types,
    _unknown_string,
)

_claim_type_strategy: st.SearchStrategy[Optional[str]] = st.one_of(
    st.none(),
    _known_claim_types,
    _unknown_string,
)

_admission_type_strategy: st.SearchStrategy[Optional[str]] = st.one_of(
    st.none(),
    _known_admission_types,
    _unknown_string,
)

_admission_source_strategy: st.SearchStrategy[Optional[str]] = st.one_of(
    st.none(),
    _known_admission_sources,
    _unknown_string,
)

# ---------------------------------------------------------------------------
# Helper – extract per-field one-hot vectors from the engineer result dicts
# ---------------------------------------------------------------------------

def _insurance_vector(result: dict) -> list[int]:
    return list(result.values())


def _claim_type_vector(result: dict) -> list[int]:
    return list(result.values())


def _admission_type_vector(result: dict) -> list[int]:
    return list(result.values())


def _admission_source_vector(result: dict) -> list[int]:
    return list(result.values())


# ---------------------------------------------------------------------------
# Property 8: One-Hot Encodings Are Mutually Consistent
#
# Validates: Requirements 3.5, 3.6, 3.7, 3.8
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    insurance_type=_insurance_type_strategy,
    claim_type=_claim_type_strategy,
    admission_type=_admission_type_strategy,
    admission_source=_admission_source_strategy,
)
def test_one_hot_encodings_are_mutually_consistent(
    insurance_type: Optional[str],
    claim_type: Optional[str],
    admission_type: Optional[str],
    admission_source: Optional[str],
) -> None:
    """**Validates: Requirements 3.5, 3.6, 3.7, 3.8**

    For any combination of insurance_type, claim_type, admission_type, and
    admission_source (including unrecognised strings and None), each one-hot
    encoded vector:
      - contains only values in {0, 1}
      - has a sum of exactly 0 (unrecognised / absent) or 1 (matched category)
    """
    # ── insurance_type (Requirement 3.5) ─────────────────────────────────────
    ins_result = ClaimLevelFeatureEngineer._one_hot_insurance_type(insurance_type)
    ins_vector = _insurance_vector(ins_result)

    assert all(v in (0, 1) for v in ins_vector), (
        f"insurance_type={insurance_type!r}: vector contains non-binary values: {ins_vector}"
    )
    ins_sum = sum(ins_vector)
    assert ins_sum in (0, 1), (
        f"insurance_type={insurance_type!r}: vector sum is {ins_sum}, expected 0 or 1"
    )
    assert len(ins_vector) == len(INSURANCE_TYPES), (
        f"insurance_type vector length {len(ins_vector)} != expected {len(INSURANCE_TYPES)}"
    )

    # ── claim_type (Requirement 3.6) ─────────────────────────────────────────
    ct_result = ClaimLevelFeatureEngineer._one_hot_claim_type(claim_type)
    ct_vector = _claim_type_vector(ct_result)

    assert all(v in (0, 1) for v in ct_vector), (
        f"claim_type={claim_type!r}: vector contains non-binary values: {ct_vector}"
    )
    ct_sum = sum(ct_vector)
    assert ct_sum in (0, 1), (
        f"claim_type={claim_type!r}: vector sum is {ct_sum}, expected 0 or 1"
    )
    assert len(ct_vector) == len(CLAIM_TYPES), (
        f"claim_type vector length {len(ct_vector)} != expected {len(CLAIM_TYPES)}"
    )

    # ── admission_type (Requirement 3.7) ─────────────────────────────────────
    at_result = ClaimLevelFeatureEngineer._one_hot_admission_type(admission_type)
    at_vector = _admission_type_vector(at_result)

    assert all(v in (0, 1) for v in at_vector), (
        f"admission_type={admission_type!r}: vector contains non-binary values: {at_vector}"
    )
    at_sum = sum(at_vector)
    assert at_sum in (0, 1), (
        f"admission_type={admission_type!r}: vector sum is {at_sum}, expected 0 or 1"
    )
    assert len(at_vector) == len(ADMISSION_TYPES), (
        f"admission_type vector length {len(at_vector)} != expected {len(ADMISSION_TYPES)}"
    )

    # ── admission_source (Requirement 3.8) ───────────────────────────────────
    as_result = ClaimLevelFeatureEngineer._one_hot_admission_source(admission_source)
    as_vector = _admission_source_vector(as_result)

    assert all(v in (0, 1) for v in as_vector), (
        f"admission_source={admission_source!r}: vector contains non-binary values: {as_vector}"
    )
    as_sum = sum(as_vector)
    assert as_sum in (0, 1), (
        f"admission_source={admission_source!r}: vector sum is {as_sum}, expected 0 or 1"
    )
    assert len(as_vector) == len(ADMISSION_SOURCES), (
        f"admission_source vector length {len(as_vector)} != expected {len(ADMISSION_SOURCES)}"
    )


@settings(max_examples=100)
@given(insurance_type=_known_insurance_types)
def test_known_insurance_type_produces_exactly_one_hot(insurance_type: str) -> None:
    """**Validates: Requirements 3.5**

    A known insurance type value SHALL produce exactly one 1 in the vector
    (at the correct position), confirming mutual exclusivity.
    """
    result = ClaimLevelFeatureEngineer._one_hot_insurance_type(insurance_type)
    vector = list(result.values())
    assert sum(vector) == 1, (
        f"Known insurance_type={insurance_type!r} produced sum {sum(vector)}, expected 1"
    )
    # The hot bit must align with the correct category key
    active_key = [k for k, v in result.items() if v == 1]
    assert len(active_key) == 1
    assert active_key[0].lower() == insurance_type.strip().lower(), (
        f"Active key {active_key[0]!r} doesn't match input {insurance_type!r}"
    )


@settings(max_examples=100)
@given(claim_type=_known_claim_types)
def test_known_claim_type_produces_exactly_one_hot(claim_type: str) -> None:
    """**Validates: Requirements 3.6**

    A known claim type value SHALL produce exactly one 1 in the vector.
    """
    result = ClaimLevelFeatureEngineer._one_hot_claim_type(claim_type)
    vector = list(result.values())
    assert sum(vector) == 1, (
        f"Known claim_type={claim_type!r} produced sum {sum(vector)}, expected 1"
    )


@settings(max_examples=100)
@given(admission_type=_known_admission_types)
def test_known_admission_type_produces_exactly_one_hot(admission_type: str) -> None:
    """**Validates: Requirements 3.7**

    A known admission type value SHALL produce exactly one 1 in the vector.
    """
    result = ClaimLevelFeatureEngineer._one_hot_admission_type(admission_type)
    vector = list(result.values())
    assert sum(vector) == 1, (
        f"Known admission_type={admission_type!r} produced sum {sum(vector)}, expected 1"
    )


@settings(max_examples=100)
@given(admission_source=_known_admission_sources)
def test_known_admission_source_produces_exactly_one_hot(admission_source: str) -> None:
    """**Validates: Requirements 3.8**

    A known admission source value SHALL produce exactly one 1 in the vector.
    """
    result = ClaimLevelFeatureEngineer._one_hot_admission_source(admission_source)
    vector = list(result.values())
    assert sum(vector) == 1, (
        f"Known admission_source={admission_source!r} produced sum {sum(vector)}, expected 1"
    )


@settings(max_examples=100)
@given(value=st.one_of(st.none(), _unknown_string))
def test_absent_or_unrecognized_insurance_type_produces_all_zeros(
    value: Optional[str],
) -> None:
    """**Validates: Requirements 3.5**

    An absent or unrecognized insurance type SHALL produce all-zero vector (sum = 0).
    """
    result = ClaimLevelFeatureEngineer._one_hot_insurance_type(value)
    vector = list(result.values())
    assert sum(vector) == 0, (
        f"Absent/unrecognized insurance_type={value!r} produced non-zero sum {sum(vector)}"
    )


@settings(max_examples=100)
@given(value=st.one_of(st.none(), _unknown_string))
def test_absent_or_unrecognized_claim_type_produces_all_zeros(
    value: Optional[str],
) -> None:
    """**Validates: Requirements 3.6**

    An absent or unrecognized claim type SHALL produce all-zero vector (sum = 0).
    """
    result = ClaimLevelFeatureEngineer._one_hot_claim_type(value)
    vector = list(result.values())
    assert sum(vector) == 0, (
        f"Absent/unrecognized claim_type={value!r} produced non-zero sum {sum(vector)}"
    )


@settings(max_examples=100)
@given(value=st.one_of(st.none(), _unknown_string))
def test_absent_or_unrecognized_admission_type_produces_all_zeros(
    value: Optional[str],
) -> None:
    """**Validates: Requirements 3.7**

    An absent or unrecognized admission type SHALL produce all-zero vector (sum = 0).
    """
    result = ClaimLevelFeatureEngineer._one_hot_admission_type(value)
    vector = list(result.values())
    assert sum(vector) == 0, (
        f"Absent/unrecognized admission_type={value!r} produced non-zero sum {sum(vector)}"
    )


@settings(max_examples=100)
@given(value=st.one_of(st.none(), _unknown_string))
def test_absent_or_unrecognized_admission_source_produces_all_zeros(
    value: Optional[str],
) -> None:
    """**Validates: Requirements 3.8**

    An absent or unrecognized admission source SHALL produce all-zero vector (sum = 0).
    """
    result = ClaimLevelFeatureEngineer._one_hot_admission_source(value)
    vector = list(result.values())
    assert sum(vector) == 0, (
        f"Absent/unrecognized admission_source={value!r} produced non-zero sum {sum(vector)}"
    )


# Feature: claim-denial-prediction, Property 10: Gender Sentinel Value Consistency
"""
Property 10: Gender Sentinel Value Consistency

**Validates: Requirements 3.4**

For any claim where the patient gender field is null, unknown, or absent,
the `gender` feature SHALL be set to −1 and no other feature computation
SHALL be blocked (i.e., compute_and_store completes without raising).
"""

from datetime import date

from claim_denial.models import ClaimRecord

# ---------------------------------------------------------------------------
# Hypothesis strategies for gender sentinel test
# ---------------------------------------------------------------------------

# "Unknown" gender strings: empty string, "U", whitespace variants, and
# arbitrary strings that are neither "M" nor "F"
_unknown_gender_string = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters=" -_/",
    ),
    min_size=0,
    max_size=20,
).filter(
    lambda s: s.strip().upper() not in ("M", "F")
)

# Combine None, explicit "U", and arbitrary non-M/F strings
_null_or_unknown_gender: st.SearchStrategy[Optional[str]] = st.one_of(
    st.none(),
    st.just("U"),
    st.just("u"),
    st.just(""),
    _unknown_gender_string,
)

# Minimal valid ClaimRecord with controllable gender
def _make_claim(gender: Optional[str]) -> ClaimRecord:
    return ClaimRecord(
        claim_id="CLM-GENDER-TEST",
        provider_npi="1234567890",
        payer_id="PAYER01",
        patient_account_number="PAN-001",
        gender=gender,
        statement_period_start=date(2024, 6, 15),
        date_of_birth=date(1980, 3, 10),
        insurance_type="Medicare",
        claim_type="Professional",
        admission_type="Elective",
        admission_source="Physician Referral",
        claim_status="Active",
    )


# ---------------------------------------------------------------------------
# Property 10: Gender Sentinel Value Consistency
#
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(gender=_null_or_unknown_gender)
def test_gender_sentinel_for_null_unknown_absent(gender: Optional[str]) -> None:
    """**Validates: Requirements 3.4**

    For any claim where the patient gender field is null, unknown, or absent,
    the `gender` feature SHALL be set to −1.
    """
    encoded = ClaimLevelFeatureEngineer._encode_gender(gender)
    assert encoded == -1, (
        f"gender={gender!r}: expected −1 (sentinel), got {encoded}"
    )


@settings(max_examples=100)
@given(gender=_null_or_unknown_gender)
def test_gender_sentinel_does_not_block_downstream_computation(
    gender: Optional[str],
) -> None:
    """**Validates: Requirements 3.4**

    When gender is null/unknown/absent, compute_and_store SHALL complete without
    raising any exception and SHALL return a ClaimFeatureRecord with gender == −1.
    All other claim-level features (temporal, one-hot) are still populated.
    """
    claim = _make_claim(gender)
    engineer = ClaimLevelFeatureEngineer()

    # Must not raise
    record = engineer.compute_and_store(claim)

    # Gender sentinel enforced
    assert record.gender == -1, (
        f"gender={gender!r}: ClaimFeatureRecord.gender expected −1, got {record.gender}"
    )

    # Downstream features are still computed — temporal fields should be populated
    assert record.claim_submission_date_doy != 0, (
        "DOY should be set from statement_period_start even when gender is sentinel"
    )
    assert record.claim_submission_date_month != 0, (
        "Month should be set from statement_period_start even when gender is sentinel"
    )

    # One-hot fields should still be set (Medicare → ins_type_medicare == 1)
    assert record.ins_type_medicare == 1, (
        "Insurance one-hot encoding should proceed normally despite gender sentinel"
    )


@settings(max_examples=100)
@given(gender=st.sampled_from(["M", "m", "F", "f"]))
def test_known_gender_values_are_not_sentinel(gender: str) -> None:
    """**Validates: Requirements 3.3**

    Valid gender values (M/F, case-insensitive) SHALL NOT produce the −1 sentinel.
    This confirms Property 10 only applies to null/unknown/absent inputs.
    """
    encoded = ClaimLevelFeatureEngineer._encode_gender(gender)
    assert encoded in (0, 1), (
        f"gender={gender!r}: expected 0 or 1, got {encoded}"
    )
    assert encoded != -1, (
        f"gender={gender!r}: should not produce sentinel −1"
    )
