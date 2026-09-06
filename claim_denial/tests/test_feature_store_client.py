"""
Unit tests for FeatureStoreClient (claim_denial/utils/feature_store_client.py).

All patient data is synthetic — no real PHI.

Requirements tested: 14.2, 14.5
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from claim_denial.utils.feature_store_client import (
    ALERT_TYPE_RETENTION,
    FeatureStoreClient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(d: date) -> str:
    """Return an ISO-8601 UTC datetime string for the given date."""
    return datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=timezone.utc).isoformat()


def _days_ago(n: int) -> str:
    """Return an ISO-8601 UTC string for a date *n* days ago."""
    return _iso(datetime.now(tz=timezone.utc).date() - timedelta(days=n))


# ---------------------------------------------------------------------------
# upsert tests  (Requirement 14.2)
# ---------------------------------------------------------------------------

class TestUpsert:
    """upsert: creates new row, merges into existing row, retry behaviour."""

    def test_creates_new_row(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        client.upsert("CLM-001", {"claim_status": "Active", "score": 0.9})
        assert client.get("CLM-001") == {"claim_status": "Active", "score": 0.9}

    def test_merges_into_existing_row(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        client.upsert("CLM-002", {"claim_status": "Active"})
        client.upsert("CLM-002", {"predicted_denial_score": 0.75})
        row = client.get("CLM-002")
        assert row["claim_status"] == "Active"
        assert row["predicted_denial_score"] == 0.75

    def test_overwrites_existing_column(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        client.upsert("CLM-003", {"claim_status": "Active"})
        client.upsert("CLM-003", {"claim_status": "Closed"})
        assert client.get("CLM-003")["claim_status"] == "Closed"

    def test_retries_on_failure_succeeds_on_third_attempt(self):
        """Two failures then success → row is written on the 3rd attempt."""
        sleep_calls: list = []
        backend = {}

        call_count = 0

        class FailingBackend(dict):
            def __setitem__(self, key, value):
                nonlocal call_count
                # The first two calls to __setitem__ for "CLM-R" raise
                if key == "CLM-R" and call_count < 2:
                    call_count += 1
                    raise IOError("transient backend error")
                super().__setitem__(key, value)

        failing_store = FailingBackend()
        client = FeatureStoreClient(
            backend=failing_store,
            sleep_fn=lambda s: sleep_calls.append(s),
            max_retries=3,
            backoff_base_seconds=1.0,
        )
        client.upsert("CLM-R", {"claim_status": "Active"})

        # Row must have been written eventually
        assert failing_store.get("CLM-R") == {"claim_status": "Active"}
        # Two sleeps should have occurred (after attempt 1 and attempt 2)
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0

    def test_all_retries_fail_logs_error_and_continues(self, caplog):
        """All 3 retries fail → ERROR logged, no exception raised."""

        class AlwaysFailStore(dict):
            def __setitem__(self, key, value):
                raise IOError("persistent backend error")

        client = FeatureStoreClient(
            backend=AlwaysFailStore(),
            sleep_fn=lambda _: None,
            max_retries=3,
            backoff_base_seconds=1.0,
        )

        with caplog.at_level(logging.ERROR):
            # Should NOT raise
            client.upsert("CLM-FAIL", {"claim_status": "Active"})

        # An ERROR record must be present mentioning the claim_id
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("CLM-FAIL" in r.message for r in error_records)


# ---------------------------------------------------------------------------
# get tests  (Requirement 14.2)
# ---------------------------------------------------------------------------

class TestGet:
    def test_returns_row_for_known_claim(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        client.upsert("CLM-GET-1", {"claim_status": "Active"})
        row = client.get("CLM-GET-1")
        assert row is not None
        assert row["claim_status"] == "Active"

    def test_returns_none_for_unknown_claim(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        assert client.get("DOES-NOT-EXIST") is None


# ---------------------------------------------------------------------------
# scan_active_claims tests  (Requirement 14.2)
# ---------------------------------------------------------------------------

class TestScanActiveClaims:
    def test_returns_only_active_rows(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        client.upsert("CLM-A1", {"claim_status": "Active"})
        client.upsert("CLM-A2", {"claim_status": "Closed"})
        client.upsert("CLM-A3", {"claim_status": "Active"})

        active = client.scan_active_claims()
        active_ids = {r["claim_id"] for r in active}
        assert active_ids == {"CLM-A1", "CLM-A3"}

    def test_empty_store_returns_empty_list(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        assert client.scan_active_claims() == []

    def test_returned_rows_include_claim_id(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        client.upsert("CLM-ID-CHECK", {"claim_status": "Active", "score": 0.5})
        rows = client.scan_active_claims()
        assert len(rows) == 1
        assert rows[0]["claim_id"] == "CLM-ID-CHECK"
        assert rows[0]["claim_status"] == "Active"

    def test_no_active_claims_returns_empty_list(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        client.upsert("CLM-CLOSED", {"claim_status": "Closed"})
        assert client.scan_active_claims() == []


# ---------------------------------------------------------------------------
# check_retention_window tests  (Requirement 14.5)
# ---------------------------------------------------------------------------

class TestCheckRetentionWindow:
    def test_oldest_record_beyond_threshold_emits_alert(self):
        """Oldest record > 90 days old → retention alert emitted."""
        alert_client = MagicMock()
        client = FeatureStoreClient(
            alert_client=alert_client,
            sleep_fn=lambda _: None,
            retention_days=90,
        )
        # Insert two records — one very old, one recent
        client.upsert("CLM-OLD", {"scoring_timestamp_utc": _days_ago(120)})
        client.upsert("CLM-NEW", {"scoring_timestamp_utc": _days_ago(10)})

        client.check_retention_window()

        alert_client.assert_called_once()
        alert_type, message = alert_client.call_args[0]
        assert alert_type == ALERT_TYPE_RETENTION
        assert "retention" in message.lower() or "120" in message or "oldest" in message.lower()

    def test_oldest_record_within_threshold_no_alert(self):
        """Oldest record ≤ 90 days old → no alert."""
        alert_client = MagicMock()
        client = FeatureStoreClient(
            alert_client=alert_client,
            sleep_fn=lambda _: None,
            retention_days=90,
        )
        client.upsert("CLM-RECENT1", {"scoring_timestamp_utc": _days_ago(80)})
        client.upsert("CLM-RECENT2", {"scoring_timestamp_utc": _days_ago(30)})

        client.check_retention_window()

        alert_client.assert_not_called()

    def test_fewer_than_two_timestamped_records_logs_warning_no_alert(self, caplog):
        """Fewer than 2 parseable timestamps → warning logged, no alert."""
        alert_client = MagicMock()
        client = FeatureStoreClient(
            alert_client=alert_client,
            sleep_fn=lambda _: None,
            retention_days=90,
        )
        # Only one record with a valid timestamp
        client.upsert("CLM-ONLY", {"scoring_timestamp_utc": _days_ago(200)})

        with caplog.at_level(logging.WARNING):
            client.check_retention_window()

        alert_client.assert_not_called()
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("insufficient" in r.message.lower() or "need" in r.message.lower()
                   or "2" in r.message for r in warning_records)

    def test_zero_timestamped_records_logs_warning_no_alert(self, caplog):
        """No timestamps at all → warning logged, no alert."""
        alert_client = MagicMock()
        client = FeatureStoreClient(
            alert_client=alert_client,
            sleep_fn=lambda _: None,
            retention_days=90,
        )
        client.upsert("CLM-NO-TS", {"claim_status": "Active"})

        with caplog.at_level(logging.WARNING):
            client.check_retention_window()

        alert_client.assert_not_called()

    def test_uses_adjudication_timestamp_when_scoring_absent(self):
        """Falls back to adjudication_timestamp_utc when scoring_timestamp_utc absent."""
        alert_client = MagicMock()
        client = FeatureStoreClient(
            alert_client=alert_client,
            sleep_fn=lambda _: None,
            retention_days=90,
        )
        client.upsert("CLM-ADJ1", {"adjudication_timestamp_utc": _days_ago(100)})
        client.upsert("CLM-ADJ2", {"adjudication_timestamp_utc": _days_ago(50)})

        client.check_retention_window()

        alert_client.assert_called_once()

    def test_exact_threshold_boundary_no_alert(self):
        """Record dated exactly 90 days ago is NOT older than threshold → no alert."""
        alert_client = MagicMock()
        client = FeatureStoreClient(
            alert_client=alert_client,
            sleep_fn=lambda _: None,
            retention_days=90,
        )
        client.upsert("CLM-BOUNDARY1", {"scoring_timestamp_utc": _days_ago(90)})
        client.upsert("CLM-BOUNDARY2", {"scoring_timestamp_utc": _days_ago(45)})

        client.check_retention_window()

        alert_client.assert_not_called()


# ---------------------------------------------------------------------------
# delete tests
# ---------------------------------------------------------------------------

class TestDelete:
    def test_removes_existing_row(self):
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        client.upsert("CLM-DEL", {"claim_status": "Active"})
        client.delete("CLM-DEL")
        assert client.get("CLM-DEL") is None

    def test_noop_for_absent_key(self):
        """delete on a non-existent key must not raise."""
        client = FeatureStoreClient(sleep_fn=lambda _: None)
        # Should complete without error
        client.delete("NONEXISTENT-KEY")
