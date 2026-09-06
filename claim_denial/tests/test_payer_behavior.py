"""
Unit tests for PayerBehaviorFeatureEngineer.

Coverage:
  - Denial rate boundary: exactly 49 claims → population-mean fallback;
    exactly 50 claims → computed rate (Requirement 6.5)
  - days_since_last_payment: no prior paid ERA → −1 cold-start sentinel;
    ERA exists → non-negative integer (Requirement 6.3)
  - Stop-loss lookup: payer absent from contract reference → 0 and
    missing-payer-ID warning logged (Requirement 6.4)

Requirements: 6.3, 6.4, 6.5
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List

import pytest

from claim_denial.features.payer_behavior import (
    PAYER_MIN_CLAIM_THRESHOLD,
    ContractEntry,
    ERARecord,
    InMemoryFeatureStore,
    PayerBehaviorFeatureEngineer,
    WindowClaim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window_claims(
    payer_id: str,
    n_denied: int,
    n_paid: int,
    claim_type: str | None = None,
) -> List[WindowClaim]:
    """Build *n_denied* denied + *n_paid* paid WindowClaims for *payer_id*."""
    claims: List[WindowClaim] = []
    for _ in range(n_denied):
        claims.append(
            WindowClaim(
                payer_id=payer_id,
                claim_type=claim_type,
                adjudication_status="denied",
                submission_date=date(2024, 6, 1),
            )
        )
    for _ in range(n_paid):
        claims.append(
            WindowClaim(
                payer_id=payer_id,
                claim_type=claim_type,
                adjudication_status="paid",
                submission_date=date(2024, 6, 1),
            )
        )
    return claims


def _make_era(payer_id: str, payment_date: date, status: str = "paid") -> ERARecord:
    return ERARecord(payer_id=payer_id, payment_date=payment_date, status=status)


def _make_contract(
    payer_id: str,
    start: date = date(2020, 1, 1),
    end: date | None = None,
    stop_loss_active: bool = False,
) -> ContractEntry:
    return ContractEntry(
        payer_id=payer_id,
        effective_date_start=start,
        effective_date_end=end,
        stop_loss_active=stop_loss_active,
    )


@pytest.fixture
def engineer() -> PayerBehaviorFeatureEngineer:
    return PayerBehaviorFeatureEngineer(feature_store=InMemoryFeatureStore())


# ===========================================================================
# Requirement 6.5 — Denial rate boundary (threshold = 50)
# ===========================================================================

class TestDenialRateBoundary:
    """Exactly 49 claims falls below the threshold; exactly 50 uses computed rate."""

    # ---- Overall payer denial rate ----------------------------------------

    def test_49_claims_uses_population_mean(self, engineer):
        """With 49 known-outcome claims the population-mean fallback is used."""
        # 49 claims for target payer (all paid → rate would be 0.0 if computed)
        # Add a background payer with 50% denial rate to give a distinct pop mean
        target_claims = _make_window_claims("PAYER_A", n_denied=0, n_paid=49)
        other_claims = _make_window_claims("PAYER_B", n_denied=5, n_paid=5)
        window = target_claims + other_claims

        rate = engineer.compute_payer_historical_denial_rate("PAYER_A", window)

        # 49 < 50 → population mean from ALL known-outcome claims
        # total known: 49 paid + 5 denied + 5 paid = 59; denied = 5 → 5/59 ≈ 0.0847
        expected_pop_mean = 5 / 59
        assert rate == pytest.approx(expected_pop_mean, abs=1e-6), (
            f"Expected population-mean {expected_pop_mean:.6f} for 49 claims, got {rate}"
        )

    def test_50_claims_uses_computed_rate(self, engineer):
        """With exactly 50 known-outcome claims the per-payer computed rate is used."""
        # 10 denied + 40 paid = 50 → rate = 10/50 = 0.20
        window = _make_window_claims("PAYER_A", n_denied=10, n_paid=40)

        rate = engineer.compute_payer_historical_denial_rate("PAYER_A", window)

        assert rate == pytest.approx(0.20, abs=1e-6), (
            f"Expected computed rate 0.20 for exactly 50 claims, got {rate}"
        )

    def test_threshold_constant_is_50(self):
        """Confirm the module-level threshold constant equals 50."""
        assert PAYER_MIN_CLAIM_THRESHOLD == 50

    # ---- Payer denial rate by claim type ----------------------------------

    def test_49_type_claims_uses_population_mean(self, engineer):
        """By-type rate: 49 Professional claims for payer → population-mean fallback."""
        target_claims = _make_window_claims(
            "PAYER_A", n_denied=0, n_paid=49, claim_type="professional"
        )
        # Background: 50% denial rate payer
        other_claims = _make_window_claims(
            "PAYER_B", n_denied=5, n_paid=5, claim_type="professional"
        )
        window = target_claims + other_claims

        rate = engineer.compute_payer_historical_denial_rate_by_type(
            "PAYER_A", "professional", window
        )

        expected_pop_mean = 5 / 59
        assert rate == pytest.approx(expected_pop_mean, abs=1e-6)

    def test_50_type_claims_uses_computed_rate(self, engineer):
        """By-type rate: exactly 50 Professional claims → computed rate."""
        # 20 denied + 30 paid = 50; rate = 20/50 = 0.40
        window = _make_window_claims(
            "PAYER_A", n_denied=20, n_paid=30, claim_type="professional"
        )

        rate = engineer.compute_payer_historical_denial_rate_by_type(
            "PAYER_A", "professional", window
        )

        assert rate == pytest.approx(0.40, abs=1e-6)

    def test_rate_is_clamped_to_unit_interval(self, engineer):
        """Computed rate must stay in [0.0, 1.0] even at extremes."""
        # All denied
        all_denied = _make_window_claims("PAYER_A", n_denied=50, n_paid=0)
        rate_max = engineer.compute_payer_historical_denial_rate("PAYER_A", all_denied)
        assert rate_max == pytest.approx(1.0)

        # All paid
        all_paid = _make_window_claims("PAYER_B", n_denied=0, n_paid=50)
        rate_min = engineer.compute_payer_historical_denial_rate("PAYER_B", all_paid)
        assert rate_min == pytest.approx(0.0)


# ===========================================================================
# Requirement 6.3 — days_since_last_payment
# ===========================================================================

class TestDaysSinceLastPayment:
    """Cold-start → −1; ERA exists → non-negative integer."""

    def test_no_prior_era_returns_cold_start_sentinel(self, engineer):
        """No paid ERA records for the payer → returns −1."""
        result = engineer.compute_days_since_last_payment(
            payer_id="PAYER_UNKNOWN",
            era_records=[],
            submission_date=date(2024, 6, 1),
        )
        assert result == -1

    def test_no_paid_era_for_payer_returns_sentinel(self, engineer):
        """ERA records exist but none are 'paid' for the target payer → −1."""
        era_records = [
            _make_era("PAYER_A", date(2024, 5, 15), status="denied"),
            _make_era("PAYER_B", date(2024, 5, 20), status="paid"),  # wrong payer
        ]
        result = engineer.compute_days_since_last_payment(
            payer_id="PAYER_A",
            era_records=era_records,
            submission_date=date(2024, 6, 1),
        )
        assert result == -1

    def test_single_paid_era_returns_correct_days(self, engineer):
        """Single paid ERA: days = submission_date − payment_date."""
        era_records = [_make_era("PAYER_A", date(2024, 5, 22), status="paid")]
        result = engineer.compute_days_since_last_payment(
            payer_id="PAYER_A",
            era_records=era_records,
            submission_date=date(2024, 6, 1),
        )
        # 2024-06-01 − 2024-05-22 = 10 days
        assert result == 10

    def test_multiple_paid_eras_uses_most_recent(self, engineer):
        """Multiple paid ERAs: pick the most recent one before submission date."""
        era_records = [
            _make_era("PAYER_A", date(2024, 4, 1), status="paid"),
            _make_era("PAYER_A", date(2024, 5, 25), status="paid"),  # most recent
            _make_era("PAYER_A", date(2024, 3, 10), status="paid"),
        ]
        result = engineer.compute_days_since_last_payment(
            payer_id="PAYER_A",
            era_records=era_records,
            submission_date=date(2024, 6, 1),
        )
        # 2024-06-01 − 2024-05-25 = 7 days
        assert result == 7

    def test_payment_on_submission_date_returns_zero(self, engineer):
        """Paid ERA on the submission date itself → 0 days."""
        era_records = [_make_era("PAYER_A", date(2024, 6, 1), status="paid")]
        result = engineer.compute_days_since_last_payment(
            payer_id="PAYER_A",
            era_records=era_records,
            submission_date=date(2024, 6, 1),
        )
        assert result == 0

    def test_result_is_non_negative_integer(self, engineer):
        """When a paid ERA exists, result is always a non-negative integer."""
        era_records = [_make_era("PAYER_A", date(2024, 5, 1), status="paid")]
        result = engineer.compute_days_since_last_payment(
            payer_id="PAYER_A",
            era_records=era_records,
            submission_date=date(2024, 6, 1),
        )
        assert isinstance(result, int)
        assert result >= 0


# ===========================================================================
# Requirement 6.4 — Stop-loss lookup: absent payer
# ===========================================================================

class TestStopLossLookup:
    """Payer absent from contract reference → 0; missing payer ID is logged at WARNING."""

    def test_absent_payer_returns_zero(self, engineer):
        """Payer not in contract reference → 0."""
        contract_ref = [
            _make_contract("PAYER_B", stop_loss_active=True),
        ]
        result = engineer.lookup_payer_contract_stop_loss(
            payer_id="PAYER_ABSENT",
            effective_date=date(2024, 6, 1),
            contract_reference=contract_ref,
        )
        assert result == 0

    def test_absent_payer_logs_warning(self, caplog, engineer):
        """Absent payer ID triggers a WARNING-level log message."""
        contract_ref = [_make_contract("PAYER_B")]

        with caplog.at_level(logging.WARNING, logger="claim_denial.features.payer_behavior"):
            engineer.lookup_payer_contract_stop_loss(
                payer_id="PAYER_MISSING",
                effective_date=date(2024, 6, 1),
                contract_reference=contract_ref,
            )

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("PAYER_MISSING" in msg for msg in warning_messages), (
            f"Expected WARNING mentioning 'PAYER_MISSING', got: {warning_messages}"
        )

    def test_empty_contract_reference_returns_zero_and_logs(self, caplog, engineer):
        """Empty contract reference → 0 and WARNING logged for absent payer."""
        with caplog.at_level(logging.WARNING, logger="claim_denial.features.payer_behavior"):
            result = engineer.lookup_payer_contract_stop_loss(
                payer_id="PAYER_X",
                effective_date=date(2024, 6, 1),
                contract_reference=[],
            )

        assert result == 0
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("PAYER_X" in msg for msg in warning_messages)

    def test_present_payer_stop_loss_active_returns_one(self, engineer):
        """Payer present with active stop-loss → 1."""
        contract_ref = [
            _make_contract(
                "PAYER_A",
                start=date(2024, 1, 1),
                end=None,
                stop_loss_active=True,
            )
        ]
        result = engineer.lookup_payer_contract_stop_loss(
            payer_id="PAYER_A",
            effective_date=date(2024, 6, 1),
            contract_reference=contract_ref,
        )
        assert result == 1

    def test_present_payer_stop_loss_inactive_returns_zero(self, engineer):
        """Payer present but stop-loss inactive → 0 (no warning logged)."""
        contract_ref = [
            _make_contract(
                "PAYER_A",
                start=date(2024, 1, 1),
                end=None,
                stop_loss_active=False,
            )
        ]
        result = engineer.lookup_payer_contract_stop_loss(
            payer_id="PAYER_A",
            effective_date=date(2024, 6, 1),
            contract_reference=contract_ref,
        )
        assert result == 0

    def test_payer_present_but_contract_expired_returns_zero_no_warning(
        self, caplog, engineer
    ):
        """Payer in table but effective range doesn't cover the date → 0.

        This is NOT the 'missing payer' case, so no WARNING should be emitted.
        """
        contract_ref = [
            _make_contract(
                "PAYER_A",
                start=date(2020, 1, 1),
                end=date(2023, 12, 31),  # expired
                stop_loss_active=True,
            )
        ]
        with caplog.at_level(logging.WARNING, logger="claim_denial.features.payer_behavior"):
            result = engineer.lookup_payer_contract_stop_loss(
                payer_id="PAYER_A",
                effective_date=date(2024, 6, 1),
                contract_reference=contract_ref,
            )

        assert result == 0
        # No WARNING should be emitted because the payer IS in the reference table
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert not any("PAYER_A" in msg for msg in warning_messages), (
            "Should not log WARNING when payer exists but date range doesn't cover claim date"
        )
