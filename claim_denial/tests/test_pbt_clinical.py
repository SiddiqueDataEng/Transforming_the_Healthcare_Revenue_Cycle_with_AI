# Feature: claim-denial-prediction, Property 6: HCUP CCS Category Mapping Assigns Valid Category or Unknown
# Feature: claim-denial-prediction, Property 7: Procedure Multi-Hot Vector Has Fixed Length and Binary Values
# Feature: claim-denial-prediction, Property 5: Charlson Comorbidity Index Is Non-Negative and Bounded
# Feature: claim-denial-prediction, Property 11: Missing Date Sentinel Values Are Distinguishable

"""
Property-based tests for clinical feature engineering.

**Validates: Requirements 5.1, 5.3, 5.4, 5.5**
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from claim_denial.constants import (
    HCUP_CCS_LOOKUP,
    HCUP_CCS_RANGE_MAX,
    HCUP_CCS_RANGE_MIN,
    HCUP_CCS_UNKNOWN,
    HCUP_PROC_CATEGORIES,
    HCUP_PROC_LOOKUP,
    SENTINEL_BOTH_DATES_ABSENT,
    SENTINEL_MISSING_INT,
)
from claim_denial.features.clinical import (
    SAMPLE_CHARLSON_MAPPING,
    build_procedure_multihot_vector,
    compute_charlson_comorbidity_index,
    compute_length_of_stay,
    map_principal_dx_to_ccs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Codes that exist in the real CCS lookup (guaranteed to produce 1–285)
_known_codes = st.sampled_from(sorted(HCUP_CCS_LOOKUP.keys()))

# Arbitrary printable-text strings — most will not be in the lookup, so they
# exercise the "unknown → 0" branch.
_arbitrary_codes = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")),
    min_size=0,
    max_size=20,
)

# Mix of known codes, arbitrary strings, None, and empty string
_icd10_strategy = st.one_of(
    _known_codes,
    _arbitrary_codes,
    st.none(),
    st.just(""),
)

# CPT/HCPCS codes known to the procedure lookup
_known_proc_codes = st.sampled_from(sorted(HCUP_PROC_LOOKUP.keys()))

# Arbitrary strings that are very unlikely to be in the lookup
_unrecognized_proc_codes = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=8,
).filter(lambda s: s.strip().upper() not in HCUP_PROC_LOOKUP and s.strip() not in HCUP_PROC_LOOKUP)

# Sets of CPT/HCPCS codes: empty, all-recognized, all-unrecognized, or mixed
_proc_code_list = st.one_of(
    st.just([]),  # empty set
    st.lists(_known_proc_codes, min_size=1, max_size=10),  # all recognized
    st.lists(_unrecognized_proc_codes, min_size=1, max_size=10),  # all unrecognized
    st.lists(
        st.one_of(_known_proc_codes, _unrecognized_proc_codes),
        min_size=0,
        max_size=10,
    ),  # mixed
)

# Charlson ICD-10 codes drawn from the mapping table
_charlson_codes = st.sampled_from(sorted(SAMPLE_CHARLSON_MAPPING.keys()))

# Non-Charlson codes: arbitrary strings unlikely to match
_non_charlson_codes = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=3,
    max_size=8,
).filter(lambda s: not any(
    s.upper().replace(".", "").strip().startswith(k)
    for k in SAMPLE_CHARLSON_MAPPING
))

# Lists of secondary ICD-10 codes for Charlson testing
_secondary_icd10_list = st.one_of(
    st.just([]),  # empty list → score must be 0
    st.just(None),  # None → score must be 0
    st.lists(_charlson_codes, min_size=1, max_size=10),  # all from mapping
    st.lists(_non_charlson_codes, min_size=1, max_size=10),  # none from mapping
    st.lists(
        st.one_of(_charlson_codes, _non_charlson_codes),
        min_size=0,
        max_size=10,
    ),  # mixed
)

# Date strategy: dates in a reasonable range
_date_strategy = st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31))


# ─────────────────────────────────────────────────────────────────────────────
# Property 6 — HCUP CCS Category Mapping Assigns Valid Category or Unknown
# ─────────────────────────────────────────────────────────────────────────────


@given(icd10_code=_icd10_strategy)
@settings(max_examples=100)
def test_property_6_ccs_result_in_valid_range(icd10_code: str | None) -> None:
    """**Validates: Requirements 5.1**

    For *any* ICD-10 code string (including None, empty, and strings not
    present in the lookup), ``map_principal_dx_to_ccs`` must return an integer
    value that is either the unknown sentinel (0) or a valid HCUP CCS category
    in the closed interval [1, 285].

    Formally: result ∈ [0, 285], i.e.
        HCUP_CCS_UNKNOWN ≤ result ≤ HCUP_CCS_RANGE_MAX
    """
    result = map_principal_dx_to_ccs(icd10_code, HCUP_CCS_LOOKUP)

    assert isinstance(result, int), (
        f"Expected int, got {type(result).__name__!r} for code={icd10_code!r}"
    )
    assert HCUP_CCS_UNKNOWN <= result <= HCUP_CCS_RANGE_MAX, (
        f"CCS result {result} is outside [0, 285] for code={icd10_code!r}"
    )


@given(icd10_code=_known_codes)
@settings(max_examples=100)
def test_property_6_known_code_maps_to_nonzero_category(icd10_code: str) -> None:
    """**Validates: Requirements 5.1**

    Any ICD-10 code that exists in the CCS lookup must map to a category in
    [HCUP_CCS_RANGE_MIN, HCUP_CCS_RANGE_MAX] (i.e. never the unknown sentinel).
    """
    result = map_principal_dx_to_ccs(icd10_code, HCUP_CCS_LOOKUP)

    assert HCUP_CCS_RANGE_MIN <= result <= HCUP_CCS_RANGE_MAX, (
        f"Known code {icd10_code!r} mapped to {result}, expected [{HCUP_CCS_RANGE_MIN}, {HCUP_CCS_RANGE_MAX}]"
    )
    assert result != HCUP_CCS_UNKNOWN, (
        f"Known code {icd10_code!r} unexpectedly returned the unknown sentinel 0"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 7 — Procedure Multi-Hot Vector Has Fixed Length and Binary Values
# ─────────────────────────────────────────────────────────────────────────────


@given(proc_codes=_proc_code_list)
@settings(max_examples=100)
def test_property_7_procedure_vector_fixed_length_and_binary(
    proc_codes: List[str],
) -> None:
    """**Validates: Requirements 5.3**

    For *any* set of CPT/HCPCS codes (including empty sets and sets with all
    unrecognized codes), the resulting ``procedure_category_vector`` SHALL have
    length exactly equal to the total number of HCUP procedure categories in
    the active reference version, and every element SHALL be 0 or 1.
    """
    expected_length = len(HCUP_PROC_CATEGORIES)
    vector = build_procedure_multihot_vector(proc_codes, HCUP_PROC_LOOKUP, expected_length)

    assert len(vector) == expected_length, (
        f"Vector length {len(vector)} != expected {expected_length} "
        f"for codes={proc_codes!r}"
    )
    for idx, val in enumerate(vector):
        assert val in (0, 1), (
            f"Element at index {idx} is {val!r}, expected 0 or 1 "
            f"for codes={proc_codes!r}"
        )


@given(proc_codes=st.just([]))
@settings(max_examples=10)
def test_property_7_empty_codes_produce_zero_vector(
    proc_codes: List[str],
) -> None:
    """**Validates: Requirements 5.3**

    An empty set of codes must produce a zero vector of the correct length.
    """
    expected_length = len(HCUP_PROC_CATEGORIES)
    vector = build_procedure_multihot_vector(proc_codes, HCUP_PROC_LOOKUP, expected_length)

    assert len(vector) == expected_length
    assert all(v == 0 for v in vector), (
        f"Expected all zeros for empty code set, got {vector}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 5 — Charlson Comorbidity Index Is Non-Negative and Bounded
# ─────────────────────────────────────────────────────────────────────────────


@given(secondary_codes=_secondary_icd10_list)
@settings(max_examples=100)
def test_property_5_charlson_index_non_negative(
    secondary_codes: Optional[List[str]],
) -> None:
    """**Validates: Requirements 5.5**

    For *any* list of secondary ICD-10 codes drawn from the Charlson mapping
    table, the computed ``comorbidity_risk_score`` SHALL be a non-negative
    integer.
    """
    score = compute_charlson_comorbidity_index(secondary_codes, SAMPLE_CHARLSON_MAPPING)

    assert isinstance(score, int), (
        f"Expected int, got {type(score).__name__!r} for codes={secondary_codes!r}"
    )
    assert score >= 0, (
        f"Charlson score {score} is negative for codes={secondary_codes!r}"
    )


@given(secondary_codes=st.one_of(st.just([]), st.just(None)))
@settings(max_examples=10)
def test_property_5_empty_codes_produce_zero_score(
    secondary_codes: Optional[List[str]],
) -> None:
    """**Validates: Requirements 5.5**

    A claim with no secondary codes (empty list or None) SHALL produce a
    Charlson score of 0.
    """
    score = compute_charlson_comorbidity_index(secondary_codes, SAMPLE_CHARLSON_MAPPING)

    assert score == 0, (
        f"Expected score=0 for empty/None codes, got {score}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 11 — Missing Date Sentinel Values Are Distinguishable
# ─────────────────────────────────────────────────────────────────────────────


@given(
    admission_date=_date_strategy,
    los_days=st.integers(min_value=0, max_value=365),
)
@settings(max_examples=100)
def test_property_11_both_dates_present_non_negative(
    admission_date: date,
    los_days: int,
) -> None:
    """**Validates: Requirements 5.4**

    When both admission and discharge dates are present (with discharge >= admission),
    ``compute_length_of_stay`` SHALL return a non-negative integer.
    """
    discharge_date = admission_date + timedelta(days=los_days)
    result = compute_length_of_stay(admission_date, discharge_date)

    assert isinstance(result, int), (
        f"Expected int, got {type(result).__name__!r}"
    )
    assert result >= 0, (
        f"LOS {result} is negative when both dates are present "
        f"(admission={admission_date}, discharge={discharge_date})"
    )
    assert result == los_days, (
        f"LOS {result} != expected {los_days} "
        f"(admission={admission_date}, discharge={discharge_date})"
    )


@given(admission_date=_date_strategy)
@settings(max_examples=100)
def test_property_11_discharge_absent_returns_sentinel_minus_one(
    admission_date: date,
) -> None:
    """**Validates: Requirements 5.4**

    When only the discharge date is absent (admission date is present),
    ``compute_length_of_stay`` SHALL return −1 (SENTINEL_MISSING_INT).
    """
    result = compute_length_of_stay(admission_date, None)

    assert result == SENTINEL_MISSING_INT, (
        f"Expected {SENTINEL_MISSING_INT} when only discharge is absent, "
        f"got {result} (admission={admission_date})"
    )


@given(discharge_date=_date_strategy)
@settings(max_examples=100)
def test_property_11_both_dates_absent_returns_sentinel_minus_two(
    discharge_date: date,
) -> None:
    """**Validates: Requirements 5.4**

    When both admission and discharge dates are absent,
    ``compute_length_of_stay`` SHALL return −2 (SENTINEL_BOTH_DATES_ABSENT).
    The discharge_date parameter here is unused — this test passes None/None.
    """
    result = compute_length_of_stay(None, None)

    assert result == SENTINEL_BOTH_DATES_ABSENT, (
        f"Expected {SENTINEL_BOTH_DATES_ABSENT} when both dates are absent, "
        f"got {result}"
    )


@given(
    admission_date=st.one_of(_date_strategy, st.none()),
    discharge_date=st.one_of(_date_strategy, st.none()),
)
@settings(max_examples=100)
def test_property_11_sentinel_values_are_distinguishable(
    admission_date: Optional[date],
    discharge_date: Optional[date],
) -> None:
    """**Validates: Requirements 5.4**

    For *any* combination of present/absent dates, the returned sentinel or
    LOS value must match the correct case:
    - Both absent → −2
    - Only discharge absent → −1
    - Both present → non-negative integer
    """
    result = compute_length_of_stay(admission_date, discharge_date)

    if admission_date is None and discharge_date is None:
        assert result == SENTINEL_BOTH_DATES_ABSENT, (
            f"Both absent: expected {SENTINEL_BOTH_DATES_ABSENT}, got {result}"
        )
    elif discharge_date is None:
        # admission present, discharge absent
        assert result == SENTINEL_MISSING_INT, (
            f"Discharge absent only: expected {SENTINEL_MISSING_INT}, got {result}"
        )
    else:
        # discharge is present
        if admission_date is None:
            # admission absent, discharge present → one date missing → -1
            assert result == SENTINEL_MISSING_INT, (
                f"Admission absent only: expected {SENTINEL_MISSING_INT}, got {result}"
            )
        else:
            assert isinstance(result, int)
            assert result >= 0, (
                f"Both dates present: expected non-negative, got {result} "
                f"(admission={admission_date}, discharge={discharge_date})"
            )
