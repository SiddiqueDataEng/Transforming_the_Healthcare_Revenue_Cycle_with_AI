# Feature: claim-denial-prediction, Property 3: Leave-One-Out Target Encoding Produces In-Range Values
# Feature: claim-denial-prediction, Property 4: Population-Mean Fallback Is In Range
"""
Property-based tests for ProviderFacilityFeatureEngineer.

**Validates: Requirements 4.1, 4.2, 4.9, 6.5**

Property 3: Leave-One-Out Target Encoding Produces In-Range Values
  For any billing provider NPI with more than 30 claims with known adjudication
  outcomes in the trailing 90-day window, the leave-one-out target-encoded value
  SHALL be a decimal in [0.0, 1.0].

Property 4: Population-Mean Fallback Is In Range
  For any provider or payer that falls below the minimum claim volume threshold
  (≤30 claims for providers, <50 claims for payers), the assigned population-mean
  denial rate SHALL be a decimal value in [0.0, 1.0] computed only from entities
  with at least one known adjudication outcome in the same 90-day window.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from claim_denial.features.provider_facility import (
    LOO_ENCODING_MIN_CLAIMS,
    ProviderFacilityFeatureEngineer,
)
from claim_denial.models import AdjudicationStatus

# ---------------------------------------------------------------------------
# Constants mirrored from provider_facility module
# ---------------------------------------------------------------------------

_DENIED = AdjudicationStatus.DENIED.value   # "denied"
_PAID = AdjudicationStatus.PAID.value       # "paid"

# A fixed reference date so window arithmetic is deterministic within each run
_REF_DATE = date(2024, 6, 15)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Synthetic NPI strings: 10-digit numeric strings (no real PHI/NPI)
_npi_strategy = st.from_regex(r"\d{10}", fullmatch=True)

# Adjudication outcome: denied or paid (known outcomes only)
_known_outcome = st.sampled_from([_DENIED, _PAID])

# Submission date within the trailing 90-day window
_window_date = st.dates(
    min_value=_REF_DATE - timedelta(days=90),
    max_value=_REF_DATE,
)

# A single known-outcome claim for a given NPI
def _claim_for_npi(npi: str, outcome: str, submission_date: date) -> dict:
    return {
        "npi": npi,
        "submission_date": submission_date,
        "adjudication_outcome": outcome,
    }


@st.composite
def _provider_with_loo_claims(draw) -> tuple[str, List[dict]]:
    """
    Draw a synthetic NPI and >30 known-outcome claims in the trailing window.

    Returns (npi, window_claims) where len(window_claims for that npi) > 30.
    """
    npi = draw(_npi_strategy)
    # Generate between 31 and 80 known-outcome claims for this provider
    n_claims = draw(st.integers(min_value=LOO_ENCODING_MIN_CLAIMS + 1, max_value=80))
    outcomes = draw(
        st.lists(_known_outcome, min_size=n_claims, max_size=n_claims)
    )
    dates = draw(
        st.lists(_window_date, min_size=n_claims, max_size=n_claims)
    )
    provider_claims = [
        _claim_for_npi(npi, outcome, d)
        for outcome, d in zip(outcomes, dates)
    ]
    # Optionally include some claims from other providers (population context)
    other_npi = draw(st.from_regex(r"\d{10}", fullmatch=True).filter(lambda x: x != npi))
    n_other = draw(st.integers(min_value=0, max_value=20))
    other_outcomes = draw(st.lists(_known_outcome, min_size=n_other, max_size=n_other))
    other_dates = draw(st.lists(_window_date, min_size=n_other, max_size=n_other))
    other_claims = [
        _claim_for_npi(other_npi, oc, od)
        for oc, od in zip(other_outcomes, other_dates)
    ]
    return npi, provider_claims + other_claims


@st.composite
def _provider_below_threshold(draw) -> tuple[str, List[dict]]:
    """
    Draw a provider NPI with ≤30 known-outcome claims in the trailing window.

    Also draws a non-empty population of other claims so the population-mean
    can be computed from at least one entity with a known outcome.
    """
    npi = draw(_npi_strategy)
    # 0 to 30 known-outcome claims for this provider
    n_claims = draw(st.integers(min_value=0, max_value=LOO_ENCODING_MIN_CLAIMS))
    outcomes = draw(st.lists(_known_outcome, min_size=n_claims, max_size=n_claims))
    dates = draw(st.lists(_window_date, min_size=n_claims, max_size=n_claims))
    provider_claims = [
        _claim_for_npi(npi, outcome, d)
        for outcome, d in zip(outcomes, dates)
    ]

    # At least one other entity with a known outcome so population mean is
    # computable (requirement: "entities with ≥1 known outcome")
    other_npi = draw(st.from_regex(r"\d{10}", fullmatch=True).filter(lambda x: x != npi))
    n_other = draw(st.integers(min_value=1, max_value=30))
    other_outcomes = draw(st.lists(_known_outcome, min_size=n_other, max_size=n_other))
    other_dates = draw(st.lists(_window_date, min_size=n_other, max_size=n_other))
    other_claims = [
        _claim_for_npi(other_npi, oc, od)
        for oc, od in zip(other_outcomes, other_dates)
    ]

    return npi, provider_claims + other_claims


@st.composite
def _empty_window(draw) -> tuple[str, List[dict]]:
    """Provider NPI with absolutely zero claims in the window (cold-start case)."""
    npi = draw(_npi_strategy)
    return npi, []


# ---------------------------------------------------------------------------
# Property 3: Leave-One-Out Target Encoding Produces In-Range Values
#
# Validates: Requirements 4.1
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(scenario=_provider_with_loo_claims())
def test_property_3_loo_encoding_in_range(scenario: tuple) -> None:
    """**Validates: Requirements 4.1**

    For any billing provider NPI with more than 30 claims with known adjudication
    outcomes in the trailing 90-day window, the leave-one-out target-encoded value
    SHALL be a decimal in [0.0, 1.0].
    """
    npi, window_claims = scenario
    engineer = ProviderFacilityFeatureEngineer()

    result = engineer.compute_billing_provider_id_encoded(
        npi=npi,
        window_claims=window_claims,
        reference_date=_REF_DATE,
    )

    assert isinstance(result, float), (
        f"Expected float, got {type(result).__name__!r} for NPI={npi!r}"
    )
    assert 0.0 <= result <= 1.0, (
        f"LOO encoded value {result} is outside [0.0, 1.0] for NPI={npi!r} "
        f"with {sum(1 for c in window_claims if c['npi'] == npi)} provider claims"
    )


# ---------------------------------------------------------------------------
# Property 4: Population-Mean Fallback Is In Range
#
# Validates: Requirements 4.2, 4.9, 6.5
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(scenario=_provider_below_threshold())
def test_property_4_population_mean_fallback_in_range(scenario: tuple) -> None:
    """**Validates: Requirements 4.2, 4.9, 6.5**

    For any provider that falls below the minimum claim volume threshold
    (≤30 known-outcome claims), the assigned population-mean denial rate SHALL be
    a decimal value in [0.0, 1.0].

    Tests both `compute_billing_provider_id_encoded` (Req 4.2) and
    `compute_provider_historical_denial_rate` (Req 4.9) population-mean paths.
    """
    npi, window_claims = scenario
    engineer = ProviderFacilityFeatureEngineer()

    # --- billing_provider_id_encoded population-mean path (Requirement 4.2) ---
    encoded = engineer.compute_billing_provider_id_encoded(
        npi=npi,
        window_claims=window_claims,
        reference_date=_REF_DATE,
    )
    assert isinstance(encoded, float), (
        f"billing_provider_id_encoded: expected float, got {type(encoded).__name__!r}"
    )
    assert 0.0 <= encoded <= 1.0, (
        f"billing_provider_id_encoded population-mean {encoded} is outside [0.0, 1.0]"
        f" for NPI={npi!r} with {sum(1 for c in window_claims if c['npi'] == npi)} provider claims"
    )

    # --- provider_historical_denial_rate population-mean path (Requirement 4.9) ---
    denial_rate = engineer.compute_provider_historical_denial_rate(
        npi=npi,
        window_claims=window_claims,
        reference_date=_REF_DATE,
    )
    assert isinstance(denial_rate, float), (
        f"provider_historical_denial_rate: expected float, got {type(denial_rate).__name__!r}"
    )
    assert 0.0 <= denial_rate <= 1.0, (
        f"provider_historical_denial_rate population-mean {denial_rate} is outside [0.0, 1.0]"
        f" for NPI={npi!r}"
    )


@settings(max_examples=100)
@given(scenario=_empty_window())
def test_property_4_population_mean_empty_window_returns_zero(scenario: tuple) -> None:
    """**Validates: Requirements 4.2, 4.9**

    When the window is completely empty (no claims at all), the population-mean
    denial rate SHALL be 0.0 (the defined safe default for the empty case).
    This is still a value in [0.0, 1.0].
    """
    npi, window_claims = scenario
    engineer = ProviderFacilityFeatureEngineer()

    encoded = engineer.compute_billing_provider_id_encoded(
        npi=npi,
        window_claims=window_claims,
        reference_date=_REF_DATE,
    )
    assert encoded == 0.0, (
        f"Empty window: expected 0.0, got {encoded} for NPI={npi!r}"
    )

    denial_rate = engineer.compute_provider_historical_denial_rate(
        npi=npi,
        window_claims=window_claims,
        reference_date=_REF_DATE,
    )
    assert denial_rate == 0.0, (
        f"Empty window: expected 0.0, got {denial_rate} for NPI={npi!r}"
    )


@settings(max_examples=100)
@given(
    npi=_npi_strategy,
    n_denied=st.integers(min_value=0, max_value=20),
    n_paid=st.integers(min_value=0, max_value=20),
)
def test_property_4_population_mean_computed_only_from_known_outcomes(
    npi: str, n_denied: int, n_paid: int
) -> None:
    """**Validates: Requirements 4.2, 4.9**

    The population-mean fallback is computed ONLY from entities with at least one
    known adjudication outcome.  Pending/unknown claims SHALL NOT contribute.
    """
    # Create a different NPI for the population pool
    other_npi = "0" * 10 if npi != "0" * 10 else "1" * 10

    known_claims = (
        [_claim_for_npi(other_npi, _DENIED, _REF_DATE)] * n_denied
        + [_claim_for_npi(other_npi, _PAID, _REF_DATE)] * n_paid
    )
    # Add pending claims that should NOT affect the population mean
    pending_claims = [
        {
            "npi": other_npi,
            "submission_date": _REF_DATE,
            "adjudication_outcome": "pending",
        }
        for _ in range(5)
    ]
    all_claims = known_claims + pending_claims

    engineer = ProviderFacilityFeatureEngineer()
    result = engineer.compute_billing_provider_id_encoded(
        npi=npi,
        window_claims=all_claims,
        reference_date=_REF_DATE,
    )

    assert 0.0 <= result <= 1.0, (
        f"Population-mean {result} outside [0.0, 1.0] with "
        f"n_denied={n_denied}, n_paid={n_paid}"
    )

    # Verify the value matches the expected population mean from known outcomes only
    total_known = n_denied + n_paid
    if total_known == 0:
        expected = 0.0
    else:
        expected = n_denied / total_known

    assert abs(result - expected) < 1e-9, (
        f"Population-mean {result} != expected {expected} (pending claims must be excluded)"
    )
