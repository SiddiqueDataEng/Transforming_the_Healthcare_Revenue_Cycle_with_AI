"""
Unit tests for TrainingDatasetBuilder.

**Validates: Requirements 8.1, 8.2, 8.6**

Tests cover:
- 90-day window yields ≥1,000 labeled records → no window expansion
- 90-day window yields <1,000 labeled records → expands to 180-day window
- Claims with non-paid/non-denied statuses are excluded; excluded count logged to MLflow
- <1,000 labeled records after 180-day expansion → InsufficientDataError raised,
  training does not proceed
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from claim_denial.training.dataset_builder import (
    InsufficientDataError,
    TrainingDataset,
    TrainingDatasetBuilder,
    _WINDOW_EXPANDED_DAYS,
    _WINDOW_INITIAL_DAYS,
    _MIN_LABELED_RECORDS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONSTRUCTION_DATE = date(2024, 6, 1)


def _make_record(
    outcome: str,
    days_before_construction: int = 30,
    construction_date: date = _CONSTRUCTION_DATE,
) -> dict:
    """Build a minimal feature-store record within the specified window."""
    sub_date = construction_date - timedelta(days=days_before_construction)
    return {
        "claim_id": f"CLM-{days_before_construction}-{outcome}",
        "submission_date": sub_date,
        "adjudication_outcome": outcome,
    }


def _generate_records(
    n_paid: int,
    n_denied: int,
    n_other: int = 0,
    days_before: int = 30,
    construction_date: date = _CONSTRUCTION_DATE,
) -> List[dict]:
    """Produce a batch of labeled + excluded records all within the window."""
    records: List[dict] = []
    for i in range(n_paid):
        r = _make_record("paid", days_before, construction_date)
        r["claim_id"] = f"CLM-PAID-{i}"
        records.append(r)
    for i in range(n_denied):
        r = _make_record("denied", days_before, construction_date)
        r["claim_id"] = f"CLM-DENIED-{i}"
        records.append(r)
    for i in range(n_other):
        r = _make_record("pending", days_before, construction_date)
        r["claim_id"] = f"CLM-OTHER-{i}"
        records.append(r)
    return records


def _generate_records_at_day(
    n_paid: int,
    n_denied: int,
    n_other: int,
    days_before: int,
    construction_date: date = _CONSTRUCTION_DATE,
    offset: int = 0,
) -> List[dict]:
    """Produce records stamped at a specific offset before construction_date."""
    records: List[dict] = []
    for i in range(n_paid):
        r = _make_record("paid", days_before, construction_date)
        r["claim_id"] = f"CLM-PAID-{days_before}-{i+offset}"
        records.append(r)
    for i in range(n_denied):
        r = _make_record("denied", days_before, construction_date)
        r["claim_id"] = f"CLM-DENIED-{days_before}-{i+offset}"
        records.append(r)
    for i in range(n_other):
        r = _make_record("pending", days_before, construction_date)
        r["claim_id"] = f"CLM-OTHER-{days_before}-{i+offset}"
        records.append(r)
    return records


# ---------------------------------------------------------------------------
# Requirement 8.1 — 90-day window ≥1,000 records: no expansion
# ---------------------------------------------------------------------------

class TestWindowExpansion:
    """Tests for the 90-day / 180-day window logic (Requirement 8.1)."""

    def test_90_day_window_sufficient_no_expansion(self):
        """
        When the 90-day window yields ≥1,000 labeled records, the builder
        SHALL NOT expand to 180 days.  All returned records must fall within
        the 90-day window.
        """
        # 1,000 records inside the 90-day window (days 1..89)
        records_in_window = _generate_records(n_paid=700, n_denied=300, days_before=45)
        # 200 additional records that are 91–150 days old (outside 90d, inside 180d)
        records_outside_90d = _generate_records(
            n_paid=150, n_denied=50, days_before=120
        )
        all_records = records_in_window + records_outside_90d

        builder = TrainingDatasetBuilder()
        dataset = builder.build(
            construction_date=_CONSTRUCTION_DATE,
            feature_store_records=all_records,
        )

        # All returned records should have submission_date within 90-day window
        window_start = _CONSTRUCTION_DATE - timedelta(days=_WINDOW_INITIAL_DAYS)
        for d in dataset.submission_dates:
            assert d >= window_start, (
                f"Record date {d} is outside the 90-day window (start={window_start}). "
                "Builder should not have expanded to 180 days."
            )

        # Total should equal only the in-window records
        assert len(dataset) == 1_000, (
            f"Expected 1,000 in-window records, got {len(dataset)}"
        )

    def test_90_day_window_insufficient_expands_to_180_days(self):
        """
        When the 90-day window yields <1,000 labeled records, the builder
        SHALL expand to the 180-day window.  Records from days 91–180 must
        then be included.
        """
        # Only 500 records within 90 days
        records_in_90d = _generate_records(n_paid=400, n_denied=100, days_before=45)
        # 600 more records between 91–179 days old (within 180d window only)
        records_91_to_179 = _generate_records(
            n_paid=400, n_denied=200, days_before=150
        )
        all_records = records_in_90d + records_91_to_179

        builder = TrainingDatasetBuilder()
        dataset = builder.build(
            construction_date=_CONSTRUCTION_DATE,
            feature_store_records=all_records,
        )

        # Total should include both window batches: 500 + 600 = 1,100
        assert len(dataset) == 1_100, (
            f"Expected 1,100 records after 180-day expansion, got {len(dataset)}"
        )

        # At least one record must come from beyond the 90-day boundary
        window_90_start = _CONSTRUCTION_DATE - timedelta(days=_WINDOW_INITIAL_DAYS)
        window_180_start = _CONSTRUCTION_DATE - timedelta(days=_WINDOW_EXPANDED_DAYS)
        has_extended = any(
            window_180_start <= d < window_90_start
            for d in dataset.submission_dates
        )
        assert has_extended, (
            "Expected records from the 91–180-day range after expansion, "
            "but none were found."
        )

    def test_90_day_window_exactly_1000_records_no_expansion(self):
        """Edge case: exactly 1,000 labeled records in 90 days → no expansion."""
        records_in_window = _generate_records(n_paid=800, n_denied=200, days_before=45)
        records_outside_90d = _generate_records(
            n_paid=500, n_denied=500, days_before=150
        )
        all_records = records_in_window + records_outside_90d

        builder = TrainingDatasetBuilder()
        dataset = builder.build(
            construction_date=_CONSTRUCTION_DATE,
            feature_store_records=all_records,
        )

        window_start = _CONSTRUCTION_DATE - timedelta(days=_WINDOW_INITIAL_DAYS)
        for d in dataset.submission_dates:
            assert d >= window_start, (
                f"Record date {d} outside 90-day window despite sufficient volume."
            )
        assert len(dataset) == 1_000

    def test_90_day_window_999_records_expands(self):
        """Edge case: exactly 999 labeled records in 90 days → expansion triggered."""
        records_in_90d = _generate_records(n_paid=666, n_denied=333, days_before=45)
        # One more record to make it exactly 999 but we only add 999 total
        # Trim to 999 by removing the last 1
        records_in_90d = records_in_90d[:999]
        records_outside_90d = _generate_records(
            n_paid=100, n_denied=50, days_before=150
        )
        all_records = records_in_90d + records_outside_90d

        builder = TrainingDatasetBuilder()
        dataset = builder.build(
            construction_date=_CONSTRUCTION_DATE,
            feature_store_records=all_records,
        )

        # Should include records from both windows
        assert len(dataset) == 999 + 150


# ---------------------------------------------------------------------------
# Requirement 8.2 — Exclusion logging
# ---------------------------------------------------------------------------

class TestExclusionLogging:
    """Claims with non-paid/non-denied status are excluded; count is logged (Req 8.2)."""

    def test_non_paid_non_denied_claims_excluded(self):
        """
        Claims with statuses other than "paid" or "denied" must be excluded from
        the labeled dataset.
        """
        paid = _generate_records(n_paid=600, n_denied=0, days_before=30)
        denied = _generate_records(n_paid=0, n_denied=400, days_before=30)
        # Mix of 'other' statuses: pending, adjusted, partially paid
        others: List[dict] = []
        for i, status in enumerate(["pending", "adjusted", "partial", "void", "error"] * 50):
            sub_date = _CONSTRUCTION_DATE - timedelta(days=30)
            others.append({
                "claim_id": f"CLM-OTHER-{i}",
                "submission_date": sub_date,
                "adjudication_outcome": status,
            })

        all_records = paid + denied + others

        builder = TrainingDatasetBuilder()
        dataset = builder.build(
            construction_date=_CONSTRUCTION_DATE,
            feature_store_records=all_records,
        )

        # Only paid and denied should appear
        assert len(dataset) == 1_000, (
            f"Expected 1,000 labeled records (paid + denied), got {len(dataset)}"
        )
        for lbl in dataset.labels:
            assert lbl in (0, 1), f"Unexpected label {lbl}; must be 0 (paid) or 1 (denied)"

    def test_excluded_count_logged_to_mlflow(self):
        """
        The count of excluded (non-paid, non-denied) claims must be logged to
        MLflow via `mlflow.log_metric('excluded_claim_count', ...)`.
        """
        n_excluded = 42
        paid = _generate_records(n_paid=700, n_denied=300, days_before=30)
        others: List[dict] = []
        for i in range(n_excluded):
            sub_date = _CONSTRUCTION_DATE - timedelta(days=30)
            others.append({
                "claim_id": f"CLM-EXCL-{i}",
                "submission_date": sub_date,
                "adjudication_outcome": "pending",
            })
        all_records = paid + others

        # Patch the mlflow shim inside dataset_builder
        with patch(
            "claim_denial.training.dataset_builder.mlflow"
        ) as mock_mlflow:
            builder = TrainingDatasetBuilder()
            builder.build(
                construction_date=_CONSTRUCTION_DATE,
                feature_store_records=all_records,
            )

        # Assert log_metric was called with ('excluded_claim_count', 42)
        mock_mlflow.log_metric.assert_any_call("excluded_claim_count", n_excluded)

    def test_zero_exclusions_still_logged(self):
        """
        When all records have paid/denied outcomes, excluded count = 0 must
        still be logged.
        """
        records = _generate_records(n_paid=700, n_denied=300, days_before=30)

        with patch(
            "claim_denial.training.dataset_builder.mlflow"
        ) as mock_mlflow:
            builder = TrainingDatasetBuilder()
            builder.build(
                construction_date=_CONSTRUCTION_DATE,
                feature_store_records=records,
            )

        mock_mlflow.log_metric.assert_any_call("excluded_claim_count", 0)

    def test_all_records_excluded_raises_insufficient_data(self):
        """
        If all records in the window have 'other' status, zero labeled records
        remain → InsufficientDataError raised (covers exclusion + halt, Req 8.6).
        """
        # 200 'pending' records in both windows — not enough after exclusions
        others = []
        for i in range(200):
            sub_date = _CONSTRUCTION_DATE - timedelta(days=30)
            others.append({
                "claim_id": f"CLM-PENDING-{i}",
                "submission_date": sub_date,
                "adjudication_outcome": "pending",
            })
        # Also 200 in the 91-179 day range
        for i in range(200):
            sub_date = _CONSTRUCTION_DATE - timedelta(days=150)
            others.append({
                "claim_id": f"CLM-PENDING-EXT-{i}",
                "submission_date": sub_date,
                "adjudication_outcome": "pending",
            })

        builder = TrainingDatasetBuilder()
        with pytest.raises(InsufficientDataError):
            builder.build(
                construction_date=_CONSTRUCTION_DATE,
                feature_store_records=others,
            )


# ---------------------------------------------------------------------------
# Requirement 8.6 — Halt condition
# ---------------------------------------------------------------------------

class TestHaltCondition:
    """Fewer than 1,000 labeled records after expansion → InsufficientDataError (Req 8.6)."""

    def test_insufficient_records_after_expansion_raises(self):
        """
        When both the 90-day and 180-day windows yield fewer than 1,000 labeled
        records combined, InsufficientDataError must be raised.
        """
        # 300 labeled records total (well below 1,000 even after expansion)
        records = _generate_records(n_paid=200, n_denied=100, days_before=45)

        builder = TrainingDatasetBuilder()
        with pytest.raises(InsufficientDataError) as exc_info:
            builder.build(
                construction_date=_CONSTRUCTION_DATE,
                feature_store_records=records,
            )

        error_message = str(exc_info.value)
        assert "300" in error_message, (
            f"Error message should contain actual record count (300): {error_message}"
        )
        assert "1000" in error_message or "1,000" in error_message, (
            f"Error message should reference the minimum required (1,000): {error_message}"
        )

    def test_halt_condition_logged_to_mlflow(self):
        """
        When InsufficientDataError is raised, the error must be logged to MLflow
        with the actual record count and minimum required (Requirement 8.6).
        """
        records = _generate_records(n_paid=200, n_denied=100, days_before=45)

        with patch(
            "claim_denial.training.dataset_builder.mlflow"
        ) as mock_mlflow:
            builder = TrainingDatasetBuilder()
            with pytest.raises(InsufficientDataError):
                builder.build(
                    construction_date=_CONSTRUCTION_DATE,
                    feature_store_records=records,
                )

        # MLflow must log the actual record count
        mock_mlflow.log_metric.assert_any_call("labeled_record_count", 300)
        mock_mlflow.log_metric.assert_any_call(
            "min_required_record_count", _MIN_LABELED_RECORDS
        )

    def test_halt_condition_does_not_return_dataset(self):
        """
        InsufficientDataError must be raised — build() must NOT return a dataset
        when the threshold is not met.
        """
        records = _generate_records(n_paid=100, n_denied=50, days_before=45)

        builder = TrainingDatasetBuilder()
        result = None
        raised = False
        try:
            result = builder.build(
                construction_date=_CONSTRUCTION_DATE,
                feature_store_records=records,
            )
        except InsufficientDataError:
            raised = True

        assert raised, "InsufficientDataError was not raised"
        assert result is None, "build() should not return a dataset on InsufficientDataError"

    def test_exactly_1000_records_does_not_raise(self):
        """Edge case: exactly 1,000 labeled records must NOT raise InsufficientDataError."""
        records = _generate_records(n_paid=700, n_denied=300, days_before=45)
        assert len(records) == 1_000

        builder = TrainingDatasetBuilder()
        # Should not raise
        dataset = builder.build(
            construction_date=_CONSTRUCTION_DATE,
            feature_store_records=records,
        )
        assert len(dataset) == 1_000

    def test_999_records_after_180d_expansion_raises(self):
        """
        999 labeled records across both windows (90d + 180d expansion) must raise
        InsufficientDataError.
        """
        # 500 in 90-day window (triggers expansion)
        records_90d = _generate_records(n_paid=350, n_denied=150, days_before=45)
        # 499 more in the 91-180-day range (total 999)
        records_180d = _generate_records(n_paid=333, n_denied=166, days_before=150)
        records_180d = records_180d[:499]  # trim to exactly 499

        all_records = records_90d + records_180d

        builder = TrainingDatasetBuilder()
        with pytest.raises(InsufficientDataError):
            builder.build(
                construction_date=_CONSTRUCTION_DATE,
                feature_store_records=all_records,
            )
