# Feature: claim-denial-prediction, Property 14: Payer Denial Rate by Type Is In Range and Consistent with Payer Rate
"""
Property-based tests for PayerBehaviorFeatureEngineer denial-rate computations.

**Validates: Requirements 6.1, 6.2**

Property 14: Payer Denial Rate by Type Is In Range and Consistent with Payer Rate
  For any payer and claim type (Professional or Institutional),
  `payer_historical_denial_rate_by_type` SHALL be a decimal in [0.0, 1.0];
  and the overall `payer_historical_denial_rate` for the same payer SHALL also
  be in [0.0, 1.0].
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from claim_denial.features.payer_behavior import (
    PayerBehaviorFeatureEngineer,
    WindowClaim,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_PAYER_IDS = ["BCBS01", "AETNA02", "UHC03", "CIGNA04", "HUM05"]
_CLAIM_TYPES = ["professional", "institutional", None]
_STATUSES = ["denied", "paid", "pending", "adjusted"]

_payer_id_strategy = st.sampled_from(_PAYER_IDS)
_claim_type_strategy: st.SearchStrategy[Optional[str]] = st.one_of(
    st.none(),
    st.sampled_from(["professional", "institutional"]),
)


@st.composite
def window_claim_strategy(draw) -> WindowClaim:
    """Generate a single WindowClaim with random but valid fields."""
    payer_id = draw(st.sampled_from(_PAYER_IDS))
    claim_type = draw(st.one_of(st.none(), st.sampled_from(["professional", "institutional"])))
    # Bias toward known-outcome statuses to exercise the rate computation path
    status = draw(st.sampled_from(["denied", "paid", "denied", "paid", "pending"]))
    # Use a fixed submission date — Property 14 tests rates, not time filtering
    return WindowClaim(
        payer_id=payer_id,
        claim_type=claim_type,
        adjudication_status=status,
        submission_date=date(2024, 6, 1),
    )


@st.composite
def window_claims_list(draw) -> List[WindowClaim]:
    """Generate a list of 0–200 WindowClaims covering multiple payers and types."""
    n = draw(st.integers(min_value=0, max_value=200))
    return [draw(window_claim_strategy()) for _ in range(n)]


# ---------------------------------------------------------------------------
# Property 14: Payer Denial Rate by Type Is In Range and Consistent with Payer Rate
#
# Validates: Requirements 6.1, 6.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    payer_id=_payer_id_strategy,
    claim_type=_claim_type_strategy,
    window_claims=window_claims_list(),
)
def test_payer_denial_rates_are_in_range(
    payer_id: str,
    claim_type: Optional[str],
    window_claims: List[WindowClaim],
) -> None:
    """**Validates: Requirements 6.1, 6.2**

    For any payer and claim type combination (including absent claim type),
    both `payer_historical_denial_rate` and `payer_historical_denial_rate_by_type`
    SHALL be decimal values in the closed interval [0.0, 1.0].
    """
    engineer = PayerBehaviorFeatureEngineer()

    overall_rate = engineer.compute_payer_historical_denial_rate(payer_id, window_claims)
    type_rate = engineer.compute_payer_historical_denial_rate_by_type(
        payer_id, claim_type, window_claims
    )

    assert 0.0 <= overall_rate <= 1.0, (
        f"payer_id={payer_id!r}: overall rate {overall_rate} is outside [0.0, 1.0]"
    )
    assert 0.0 <= type_rate <= 1.0, (
        f"payer_id={payer_id!r}, claim_type={claim_type!r}: "
        f"type rate {type_rate} is outside [0.0, 1.0]"
    )


@settings(max_examples=100)
@given(
    payer_id=_payer_id_strategy,
    claim_type=_claim_type_strategy,
    window_claims=window_claims_list(),
)
def test_payer_denial_rates_are_floats(
    payer_id: str,
    claim_type: Optional[str],
    window_claims: List[WindowClaim],
) -> None:
    """**Validates: Requirements 6.1, 6.2**

    Both denial rate outputs SHALL be Python floats (not None, not int coerced
    outside [0.0, 1.0]).
    """
    engineer = PayerBehaviorFeatureEngineer()

    overall_rate = engineer.compute_payer_historical_denial_rate(payer_id, window_claims)
    type_rate = engineer.compute_payer_historical_denial_rate_by_type(
        payer_id, claim_type, window_claims
    )

    assert isinstance(overall_rate, float), (
        f"payer_historical_denial_rate should be float, got {type(overall_rate)}"
    )
    assert isinstance(type_rate, float), (
        f"payer_historical_denial_rate_by_type should be float, got {type(type_rate)}"
    )


@settings(max_examples=100)
@given(
    payer_id=_payer_id_strategy,
    window_claims=window_claims_list(),
)
def test_payer_denial_rate_both_claim_types_in_range(
    payer_id: str,
    window_claims: List[WindowClaim],
) -> None:
    """**Validates: Requirements 6.1, 6.2**

    For both Professional and Institutional claim types explicitly,
    the type-specific denial rate SHALL remain in [0.0, 1.0].
    """
    engineer = PayerBehaviorFeatureEngineer()

    for claim_type in ("professional", "institutional"):
        rate = engineer.compute_payer_historical_denial_rate_by_type(
            payer_id, claim_type, window_claims
        )
        assert 0.0 <= rate <= 1.0, (
            f"payer_id={payer_id!r}, claim_type={claim_type!r}: "
            f"type rate {rate} is outside [0.0, 1.0]"
        )
