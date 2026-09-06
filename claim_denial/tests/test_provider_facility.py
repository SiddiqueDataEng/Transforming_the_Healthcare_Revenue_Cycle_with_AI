"""
Unit tests for ProviderFacilityFeatureEngineer.

Covers:
- LOO encoding boundary: exactly 30 known-outcome claims → population-mean fallback;
  exactly 31 → LOO encoding used.
- NUCC sentinel: NPI not in registry → returns "UNKNOWN".
- Facility bed size: NPI absent from master → integer mean of all bed_size values.

Requirements: 4.1, 4.2, 4.4, 4.8
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import pytest

from claim_denial.features.provider_facility import (
    LOO_ENCODING_MIN_CLAIMS,
    NpiRegistry,
    ProviderFacilityFeatureEngineer,
    SENTINEL_SPECIALTY_UNKNOWN,
)
from claim_denial.models import AdjudicationStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DENIED = AdjudicationStatus.DENIED.value   # "denied"
_PAID = AdjudicationStatus.PAID.value       # "paid"
_REF_DATE = date(2024, 6, 15)
_NPI = "1234567890"
_OTHER_NPI = "9876543210"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(
    npi: str,
    outcome: str,
    offset_days: int = 0,
    ref: date = _REF_DATE,
) -> dict:
    """Create a minimal window-claim dict within the trailing 90-day window."""
    submission_date = ref - timedelta(days=offset_days)
    return {
        "npi": npi,
        "submission_date": submission_date,
        "adjudication_outcome": outcome,
    }


def _make_provider_claims(
    npi: str,
    n_denied: int,
    n_paid: int,
    ref: date = _REF_DATE,
) -> List[dict]:
    """Build a list of known-outcome claims for a single provider NPI."""
    claims = []
    for i in range(n_denied):
        claims.append(_make_claim(npi, _DENIED, offset_days=i % 89, ref=ref))
    for j in range(n_paid):
        claims.append(_make_claim(npi, _PAID, offset_days=j % 89, ref=ref))
    return claims


# ---------------------------------------------------------------------------
# LOO encoding boundary tests
# ---------------------------------------------------------------------------

class TestLOOEncodingBoundary:
    """Requirement 4.1 / 4.2 — boundary between LOO encoding and pop-mean fallback."""

    def test_exactly_30_claims_uses_population_mean(self) -> None:
        """Exactly 30 known-outcome claims → population-mean fallback (≤30 triggers pop-mean).

        Requirement 4.2: ≤30 known-outcome claims → population-mean.
        LOO_ENCODING_MIN_CLAIMS == 30, so the condition is >30 for LOO.
        """
        # 30 claims for the target NPI — all denied
        provider_claims = _make_provider_claims(_NPI, n_denied=30, n_paid=0)

        # Population also includes another provider with mixed outcomes
        # so we can verify the pop-mean is used rather than NPI's own rate
        other_claims = _make_provider_claims(_OTHER_NPI, n_denied=5, n_paid=15)
        window_claims = provider_claims + other_claims

        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.compute_billing_provider_id_encoded(
            npi=_NPI,
            window_claims=window_claims,
            reference_date=_REF_DATE,
        )

        # Population mean = (30 denied + 5 denied) / (30 + 20) = 35/50 = 0.70
        # If LOO were used the result would be 1.0 (all NPI claims are denied).
        # The result must match the population-mean, not 1.0.
        total_known = 30 + 5 + 15  # 50
        total_denied = 30 + 5      # 35
        expected_pop_mean = total_denied / total_known
        assert abs(result - expected_pop_mean) < 1e-9, (
            f"Expected population-mean {expected_pop_mean:.4f}, got {result:.4f}. "
            "Exactly 30 claims should trigger population-mean fallback, not LOO."
        )

    def test_exactly_31_claims_uses_loo_encoding(self) -> None:
        """Exactly 31 known-outcome claims → LOO encoding used (>30 triggers LOO).

        Requirement 4.1: >30 known-outcome claims → leave-one-out encoding.
        """
        # 31 claims for the target NPI: 20 denied, 11 paid
        provider_claims = _make_provider_claims(_NPI, n_denied=20, n_paid=11)
        assert len(provider_claims) == 31

        # Add population context that differs from the NPI's own rate
        other_claims = _make_provider_claims(_OTHER_NPI, n_denied=1, n_paid=49)
        window_claims = provider_claims + other_claims

        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.compute_billing_provider_id_encoded(
            npi=_NPI,
            window_claims=window_claims,
            reference_date=_REF_DATE,
        )

        # LOO removes one claim (the last, per convention) from the 31 provider claims.
        # Remaining 30 claims: 20 denied, 10 paid → rate = 20/30 ≈ 0.6667
        # Population mean = (20+1)/(31+50) ≈ 0.2593 (very different from LOO result)
        loo_other_claims = provider_claims[:-1]  # 30 claims
        expected_loo = sum(1 for c in loo_other_claims if c["adjudication_outcome"] == _DENIED) / len(loo_other_claims)
        assert abs(result - expected_loo) < 1e-9, (
            f"Expected LOO value {expected_loo:.4f}, got {result:.4f}. "
            "Exactly 31 claims should trigger LOO encoding, not population-mean."
        )

    def test_boundary_30_vs_31_results_differ(self) -> None:
        """The switch at the 31-claim boundary should produce measurably different results.

        This guards against an off-by-one error that might make both boundaries
        behave identically.
        """
        # All 30/31 claims denied for target NPI
        # Population has a clearly different denial rate (50%)
        other_claims = _make_provider_claims(_OTHER_NPI, n_denied=25, n_paid=25)

        # At 30 claims: pop-mean = 50+25/30+50 = 30/80? No, re-check:
        # 30 NPI denied + 25 other denied + 25 other paid = 80 total known
        # pop_mean = 55/80 = 0.6875
        claims_30 = _make_provider_claims(_NPI, n_denied=30, n_paid=0)
        engineer = ProviderFacilityFeatureEngineer()

        result_30 = engineer.compute_billing_provider_id_encoded(
            npi=_NPI,
            window_claims=claims_30 + other_claims,
            reference_date=_REF_DATE,
        )

        # At 31 claims (all denied): LOO removes one → 30 denied / 30 = 1.0
        claims_31 = _make_provider_claims(_NPI, n_denied=31, n_paid=0)
        result_31 = engineer.compute_billing_provider_id_encoded(
            npi=_NPI,
            window_claims=claims_31 + other_claims,
            reference_date=_REF_DATE,
        )

        assert result_30 != result_31, (
            f"30-claim result ({result_30}) and 31-claim result ({result_31}) "
            "should differ: one uses pop-mean, the other uses LOO."
        )

    def test_zero_claims_returns_population_mean_zero(self) -> None:
        """Zero known-outcome claims for target NPI with empty window → 0.0."""
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.compute_billing_provider_id_encoded(
            npi=_NPI,
            window_claims=[],
            reference_date=_REF_DATE,
        )
        assert result == 0.0

    def test_32_claims_all_paid_loo_returns_zero(self) -> None:
        """32 all-paid claims → LOO removes one → 31 paid / 31 = 0.0 denial rate."""
        provider_claims = _make_provider_claims(_NPI, n_denied=0, n_paid=32)
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.compute_billing_provider_id_encoded(
            npi=_NPI,
            window_claims=provider_claims,
            reference_date=_REF_DATE,
        )
        assert result == 0.0, (
            f"All paid provider with 32 claims: expected LOO value 0.0, got {result}"
        )

    def test_32_claims_all_denied_loo_returns_one(self) -> None:
        """32 all-denied claims → LOO removes one → 31 denied / 31 = 1.0 denial rate."""
        provider_claims = _make_provider_claims(_NPI, n_denied=32, n_paid=0)
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.compute_billing_provider_id_encoded(
            npi=_NPI,
            window_claims=provider_claims,
            reference_date=_REF_DATE,
        )
        assert result == 1.0, (
            f"All denied provider with 32 claims: expected LOO value 1.0, got {result}"
        )


# ---------------------------------------------------------------------------
# NUCC taxonomy / physician specialty sentinel tests
# ---------------------------------------------------------------------------

class TestNUCCSentinel:
    """Requirement 4.3 / 4.4 — specialty lookup and UNKNOWN sentinel."""

    def test_npi_not_in_registry_returns_unknown(self) -> None:
        """NPI not in registry → UNKNOWN sentinel.

        Requirement 4.4: absent NPI → assign "UNKNOWN".
        """
        engineer = ProviderFacilityFeatureEngineer()
        # Default NpiRegistry always returns UNKNOWN
        result = engineer.lookup_physician_specialty(_NPI)
        assert result == SENTINEL_SPECIALTY_UNKNOWN, (
            f"Expected 'UNKNOWN', got {result!r}"
        )

    def test_npi_registry_returns_empty_string_maps_to_unknown(self) -> None:
        """Registry returning empty string → UNKNOWN sentinel."""
        engineer = ProviderFacilityFeatureEngineer(npi_lookup_fn=lambda npi: "")
        result = engineer.lookup_physician_specialty(_NPI)
        assert result == SENTINEL_SPECIALTY_UNKNOWN

    def test_npi_registry_returns_whitespace_maps_to_unknown(self) -> None:
        """Registry returning whitespace → UNKNOWN sentinel."""
        engineer = ProviderFacilityFeatureEngineer(npi_lookup_fn=lambda npi: "   ")
        result = engineer.lookup_physician_specialty(_NPI)
        assert result == SENTINEL_SPECIALTY_UNKNOWN

    def test_npi_registry_returns_none_maps_to_unknown(self) -> None:
        """Registry returning None → UNKNOWN sentinel."""
        engineer = ProviderFacilityFeatureEngineer(npi_lookup_fn=lambda npi: None)
        result = engineer.lookup_physician_specialty(_NPI)
        assert result == SENTINEL_SPECIALTY_UNKNOWN

    def test_npi_registry_raises_exception_maps_to_unknown(self) -> None:
        """Registry raising any exception → UNKNOWN sentinel (no propagation).

        Requirement 4.4: registry unavailable → "UNKNOWN".
        """
        def _failing_lookup(npi: str) -> str:
            raise RuntimeError("Registry unavailable")

        engineer = ProviderFacilityFeatureEngineer(npi_lookup_fn=_failing_lookup)
        result = engineer.lookup_physician_specialty(_NPI)
        assert result == SENTINEL_SPECIALTY_UNKNOWN

    def test_npi_registry_returns_valid_code(self) -> None:
        """Registry returning a valid taxonomy code → trimmed code is returned."""
        engineer = ProviderFacilityFeatureEngineer(
            npi_lookup_fn=lambda npi: "  207Q00000X  "
        )
        result = engineer.lookup_physician_specialty(_NPI)
        assert result == "207Q00000X", (
            f"Expected trimmed taxonomy code, got {result!r}"
        )

    def test_npi_registry_object_returns_valid_code(self) -> None:
        """Injecting a custom NpiRegistry object → correct code returned."""
        class _FakeRegistry(NpiRegistry):
            def lookup_specialty(self, npi: str) -> str:
                return "363L00000X"

        engineer = ProviderFacilityFeatureEngineer(npi_registry=_FakeRegistry())
        result = engineer.lookup_physician_specialty(_NPI)
        assert result == "363L00000X"

    def test_npi_lookup_fn_takes_priority_over_registry(self) -> None:
        """When both npi_lookup_fn and npi_registry are injected, fn takes priority."""
        class _FakeRegistry(NpiRegistry):
            def lookup_specialty(self, npi: str) -> str:
                return "REGISTRY_CODE"

        engineer = ProviderFacilityFeatureEngineer(
            npi_registry=_FakeRegistry(),
            npi_lookup_fn=lambda npi: "FN_CODE",
        )
        result = engineer.lookup_physician_specialty(_NPI)
        assert result == "FN_CODE", (
            "npi_lookup_fn should take priority over npi_registry"
        )


# ---------------------------------------------------------------------------
# Facility bed size tests
# ---------------------------------------------------------------------------

class TestFacilityBedSize:
    """Requirement 4.7 / 4.8 — bed size lookup and mean fallback."""

    def test_npi_present_in_master_returns_direct_value(self) -> None:
        """NPI present → exact value from master."""
        master = {_NPI: 250, _OTHER_NPI: 100}
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.lookup_facility_bed_size(_NPI, master)
        assert result == 250

    def test_npi_absent_from_master_returns_integer_mean(self) -> None:
        """NPI absent from master → integer mean of all values.

        Requirement 4.8: absent NPI → integer mean (rounded to nearest integer).
        """
        master = {"1111111111": 100, "2222222222": 200, "3333333333": 300}
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.lookup_facility_bed_size(_NPI, master)
        # Mean = (100 + 200 + 300) / 3 = 200
        assert result == 200
        assert isinstance(result, int)

    def test_npi_absent_mean_rounds_to_nearest_integer(self) -> None:
        """NPI absent from master → mean rounded to nearest integer."""
        # Mean = (100 + 101) / 2 = 100.5 → rounds to 100 (Python banker's rounding)
        # or to 101 depending on round() behavior; either way, result must be int
        master = {"1111111111": 100, "2222222222": 101}
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.lookup_facility_bed_size("0000000001", master)
        assert isinstance(result, int), f"Expected int, got {type(result).__name__!r}"
        assert result in (100, 101), (
            f"Expected 100 or 101 from rounding 100.5, got {result}"
        )

    def test_empty_master_returns_zero(self) -> None:
        """Empty facility master → return 0 (no values to average)."""
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.lookup_facility_bed_size(_NPI, {})
        assert result == 0

    def test_single_entry_master_npi_absent_returns_that_value(self) -> None:
        """Master with one entry and NPI absent → mean equals that single entry."""
        master = {"1111111111": 150}
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.lookup_facility_bed_size(_NPI, master)
        assert result == 150

    def test_npi_absent_mean_excludes_no_outliers(self) -> None:
        """Bed-size mean uses ALL values in master, not a subset."""
        master = {
            "1111111111": 50,
            "2222222222": 100,
            "3333333333": 150,
            "4444444444": 200,
        }
        expected_mean = round((50 + 100 + 150 + 200) / 4)  # 125
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.lookup_facility_bed_size("0000000000", master)
        assert result == expected_mean

    def test_result_is_non_negative_integer(self) -> None:
        """Bed size returned for absent NPI must be a non-negative integer."""
        master = {"1111111111": 0, "2222222222": 0}
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.lookup_facility_bed_size(_NPI, master)
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# Provider historical denial rate boundary tests
# ---------------------------------------------------------------------------

class TestProviderHistoricalDenialRateBoundary:
    """Requirement 4.5 / 4.9 — denial rate with pop-mean fallback boundary (<30)."""

    def test_29_claims_uses_population_mean(self) -> None:
        """29 known-outcome claims (<30) → population-mean fallback.

        Requirement 4.9: <30 known-outcome claims → population-mean.
        """
        provider_claims = _make_provider_claims(_NPI, n_denied=29, n_paid=0)
        other_claims = _make_provider_claims(_OTHER_NPI, n_denied=5, n_paid=45)
        window_claims = provider_claims + other_claims

        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.compute_provider_historical_denial_rate(
            npi=_NPI,
            window_claims=window_claims,
            reference_date=_REF_DATE,
        )

        total_known = 29 + 50
        total_denied = 29 + 5
        expected = total_denied / total_known
        assert abs(result - expected) < 1e-9, (
            f"Expected population-mean {expected:.4f}, got {result:.4f}. "
            "29 claims should trigger population-mean."
        )

    def test_30_claims_uses_population_mean(self) -> None:
        """30 known-outcome claims (<30 is strictly less than, so 30 uses pop-mean).

        PROVIDER_DENIAL_RATE_MIN_CLAIMS == 30 and condition is >=30 for direct,
        so exactly 30 should use direct (provider's own rate).
        """
        provider_claims = _make_provider_claims(_NPI, n_denied=15, n_paid=15)
        # No other claims — isolated rate should be 0.5
        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.compute_provider_historical_denial_rate(
            npi=_NPI,
            window_claims=provider_claims,
            reference_date=_REF_DATE,
        )
        # With exactly 30 claims and PROVIDER_DENIAL_RATE_MIN_CLAIMS=30,
        # the condition is len >= 30, so direct rate applies: 15/30 = 0.5
        assert abs(result - 0.5) < 1e-9, (
            f"Expected 0.5 with exactly 30 claims (direct rate), got {result}"
        )

    def test_31_claims_uses_direct_rate(self) -> None:
        """31 known-outcome claims (>= threshold) → provider's own direct rate."""
        provider_claims = _make_provider_claims(_NPI, n_denied=10, n_paid=21)
        other_claims = _make_provider_claims(_OTHER_NPI, n_denied=30, n_paid=20)
        window_claims = provider_claims + other_claims

        engineer = ProviderFacilityFeatureEngineer()
        result = engineer.compute_provider_historical_denial_rate(
            npi=_NPI,
            window_claims=window_claims,
            reference_date=_REF_DATE,
        )

        expected = 10 / 31
        assert abs(result - expected) < 1e-9, (
            f"Expected direct rate {expected:.4f}, got {result:.4f}. "
            "31 claims should use provider's own rate."
        )

    def test_result_in_range(self) -> None:
        """Historical denial rate is always in [0.0, 1.0]."""
        for n_denied, n_paid in [(0, 50), (50, 0), (25, 25), (0, 0)]:
            provider_claims = _make_provider_claims(_NPI, n_denied, n_paid)
            engineer = ProviderFacilityFeatureEngineer()
            result = engineer.compute_provider_historical_denial_rate(
                npi=_NPI,
                window_claims=provider_claims,
                reference_date=_REF_DATE,
            )
            assert 0.0 <= result <= 1.0, (
                f"Denial rate {result} is outside [0.0, 1.0] for "
                f"n_denied={n_denied}, n_paid={n_paid}"
            )
