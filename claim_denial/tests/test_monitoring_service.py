"""
Unit tests for MonitoringService.

Coverage:
  - Rolling Precision: < 30 claims → None; ≥ 30 claims → correct value computed.
  - Threshold-based alerting:
      precision = 0.84 → degradation alert emitted, retraining_trigger NOT set.
      precision = 0.79 → urgent alert emitted AND retraining_trigger = "active".
  - Alert retry logic:
      delivery fails × 3 → final delivery-failure record is logged.
      success on 2nd retry → no final delivery-failure log entry.
  - ERA/835 linkage:
      adjudication_outcome and adjudication_timestamp_utc written correctly.
      Unknown claim_id creates a stub row.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call

import pytest

from claim_denial.monitoring.monitoring_service import (
    ALERT_DEGRADATION,
    ALERT_URGENT,
    MonitoringService,
    _parse_date,
)
from claim_denial.models import AdjudicationStatus, MonitoringRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AS_OF = date(2024, 6, 15)
WINDOW_START = AS_OF - timedelta(days=30)


def _make_scored_row(
    *,
    score: float,
    outcome: str,
    ref_date: date = AS_OF,
) -> Dict[str, Any]:
    """Return a minimal Feature Store row with the fields MonitoringService needs."""
    return {
        "predicted_denial_score": score,
        "adjudication_outcome": outcome,
        "scoring_timestamp_utc": ref_date.isoformat(),
    }


def _build_feature_store(
    n_true_positives: int,
    n_false_positives: int,
    n_true_negatives: int = 0,
    ref_date: date = AS_OF,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a feature store dict with controlled TP / FP / TN counts.

    All rows fall within the [WINDOW_START, AS_OF] window.
    """
    store: Dict[str, Dict[str, Any]] = {}
    idx = 0

    for _ in range(n_true_positives):
        store[f"CLM-TP-{idx}"] = _make_scored_row(
            score=0.9, outcome="denied", ref_date=ref_date
        )
        idx += 1

    for _ in range(n_false_positives):
        store[f"CLM-FP-{idx}"] = _make_scored_row(
            score=0.7, outcome="paid", ref_date=ref_date
        )
        idx += 1

    for _ in range(n_true_negatives):
        store[f"CLM-TN-{idx}"] = _make_scored_row(
            score=0.3, outcome="paid", ref_date=ref_date
        )
        idx += 1

    return store


def _no_sleep(seconds: float) -> None:
    """Replace time.sleep to avoid real delays during tests."""


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def alert_client():
    return MagicMock()


@pytest.fixture
def service(alert_client):
    """MonitoringService with an empty feature store and mocked alert client."""
    return MonitoringService(
        feature_store={},
        alert_client=alert_client,
        sleep_fn=_no_sleep,
    )


# ===========================================================================
# ERA/835 Linkage — Requirement 12.1
# ===========================================================================

class TestLinkAdjudicationOutcome:
    """adjudication_outcome and adjudication_timestamp_utc are written correctly."""

    def test_links_outcome_to_existing_claim(self, service):
        service._feature_store["CLM-001"] = {"predicted_denial_score": 0.8}
        era = {
            "claim_id": "CLM-001",
            "adjudication_status": "denied",
            "processed_timestamp_utc": "2024-06-15T10:00:00+00:00",
        }
        service.link_adjudication_outcome(era)
        row = service._feature_store["CLM-001"]
        assert row["adjudication_outcome"] == "denied"
        assert row["adjudication_timestamp_utc"] == "2024-06-15T10:00:00+00:00"

    def test_links_paid_status(self, service):
        service._feature_store["CLM-002"] = {"predicted_denial_score": 0.2}
        era = {
            "claim_id": "CLM-002",
            "adjudication_status": "paid",
            "processed_timestamp_utc": "2024-06-14T08:30:00+00:00",
        }
        service.link_adjudication_outcome(era)
        row = service._feature_store["CLM-002"]
        assert row["adjudication_outcome"] == "paid"

    def test_creates_stub_row_for_unknown_claim(self, service):
        """ERA arrives before the claim is scored; a stub row is created."""
        era = {
            "claim_id": "CLM-UNKNOWN",
            "adjudication_status": "denied",
            "processed_timestamp_utc": "2024-06-15T12:00:00+00:00",
        }
        service.link_adjudication_outcome(era)
        assert "CLM-UNKNOWN" in service._feature_store
        assert service._feature_store["CLM-UNKNOWN"]["adjudication_outcome"] == "denied"

    def test_datetime_object_timestamp_serialised_to_string(self, service):
        """A datetime object is converted to an ISO-8601 string."""
        service._feature_store["CLM-003"] = {}
        ts = datetime(2024, 6, 15, 9, 0, 0, tzinfo=timezone.utc)
        era = {
            "claim_id": "CLM-003",
            "adjudication_status": "paid",
            "processed_timestamp_utc": ts,
        }
        service.link_adjudication_outcome(era)
        stored = service._feature_store["CLM-003"]["adjudication_timestamp_utc"]
        assert isinstance(stored, str)
        assert "2024-06-15" in stored

    def test_overwrites_existing_outcome(self, service):
        """A second ERA for the same claim overwrites the previous outcome."""
        service._feature_store["CLM-004"] = {"adjudication_outcome": "pending"}
        era = {
            "claim_id": "CLM-004",
            "adjudication_status": "denied",
            "processed_timestamp_utc": "2024-06-15T11:00:00",
        }
        service.link_adjudication_outcome(era)
        assert service._feature_store["CLM-004"]["adjudication_outcome"] == "denied"


# ===========================================================================
# Rolling Precision — Requirement 12.2
# ===========================================================================

class TestComputeRollingPrecision:
    """Precision window logic and threshold for returning None."""

    def test_returns_none_when_fewer_than_30_claims(self, alert_client):
        """29 qualifying claims → returns None."""
        store = _build_feature_store(n_true_positives=20, n_false_positives=9)
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        assert svc.compute_rolling_precision(AS_OF) is None

    def test_returns_none_with_zero_claims(self, alert_client):
        svc = MonitoringService({}, alert_client, sleep_fn=_no_sleep)
        assert svc.compute_rolling_precision(AS_OF) is None

    def test_returns_precision_with_exactly_30_claims(self, alert_client):
        """30 qualifying claims → precision computed correctly."""
        # 24 TP, 6 FP → precision = 24/30 = 0.8
        store = _build_feature_store(n_true_positives=24, n_false_positives=6)
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        precision = svc.compute_rolling_precision(AS_OF)
        assert precision is not None
        assert abs(precision - 0.8) < 1e-9

    def test_correct_precision_many_claims(self, alert_client):
        """40 TP + 10 FP = 50 total → precision = 0.8."""
        store = _build_feature_store(n_true_positives=40, n_false_positives=10)
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        precision = svc.compute_rolling_precision(AS_OF)
        assert precision is not None
        assert abs(precision - 0.8) < 1e-9

    def test_perfect_precision(self, alert_client):
        """All 30 are TP → precision = 1.0."""
        store = _build_feature_store(n_true_positives=30, n_false_positives=0)
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        precision = svc.compute_rolling_precision(AS_OF)
        assert precision == 1.0

    def test_claims_outside_window_excluded(self, alert_client):
        """Claims dated before the 30-day window are ignored."""
        old_date = AS_OF - timedelta(days=31)
        store = _build_feature_store(
            n_true_positives=30, n_false_positives=0, ref_date=old_date
        )
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        # All claims outside the window → returns None
        assert svc.compute_rolling_precision(AS_OF) is None

    def test_claims_with_unknown_outcome_excluded(self, alert_client):
        """Claims with 'pending' or 'adjusted' outcomes are excluded."""
        store = _build_feature_store(n_true_positives=20, n_false_positives=5)
        # Add 10 pending claims inside the window — should not count
        for i in range(10):
            store[f"CLM-PENDING-{i}"] = _make_scored_row(
                score=0.9, outcome="pending", ref_date=AS_OF
            )
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        # Only 25 qualifying (< 30) → None
        assert svc.compute_rolling_precision(AS_OF) is None

    def test_true_negatives_do_not_inflate_precision(self, alert_client):
        """True negatives (low score, paid) don't affect precision numerator/denominator."""
        # 20 TP, 5 FP, 25 TN = 50 total; precision = 20 / 25 = 0.8
        store = _build_feature_store(
            n_true_positives=20, n_false_positives=5, n_true_negatives=25
        )
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        precision = svc.compute_rolling_precision(AS_OF)
        assert precision is not None
        assert abs(precision - (20 / 25)) < 1e-9

    def test_returns_none_when_no_predicted_denials(self, alert_client):
        """If no claim has score ≥ 0.5 in the window, precision is undefined → None."""
        store = {}
        for i in range(30):
            store[f"CLM-TN-{i}"] = _make_scored_row(
                score=0.2, outcome="paid", ref_date=AS_OF
            )
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        assert svc.compute_rolling_precision(AS_OF) is None


# ===========================================================================
# Threshold-Based Alerting — Requirements 12.4, 12.5
# ===========================================================================

class TestAlertThresholds:
    """
    precision = 0.84 → degradation alert; no retraining_trigger.
    precision = 0.79 → urgent alert AND retraining_trigger = "active".
    precision ≥ 0.85 → no alert.
    """

    def _make_svc_with_precision(self, alert_client, precision: float):
        """
        Build a service whose feature store yields the target precision.
        Precision = TP / (TP + FP).  We use 30 total predicted-denial claims.
        """
        tp = round(precision * 30)
        fp = 30 - tp
        # Add enough TN so there are ≥ 30 qualifying claims in the window
        tn = max(0, 30 - tp - fp)
        store = _build_feature_store(
            n_true_positives=tp, n_false_positives=fp, n_true_negatives=tn
        )
        retraining_state: dict = {}
        svc = MonitoringService(
            store, alert_client, sleep_fn=_no_sleep,
            retraining_state=retraining_state,
        )
        return svc

    def test_precision_084_emits_degradation_alert_only(self, alert_client):
        """precision ≈ 0.84 → degradation alert; retraining_trigger NOT set."""
        # 26 TP / 31 total → 26/31 ≈ 0.839
        store = _build_feature_store(n_true_positives=26, n_false_positives=5)
        retraining_state: dict = {}
        svc = MonitoringService(
            store, alert_client, sleep_fn=_no_sleep,
            retraining_state=retraining_state,
        )
        svc.run_post_pipeline_monitoring(AS_OF)

        alert_client.assert_called_once()
        alert_type_called = alert_client.call_args[0][0]
        assert alert_type_called == ALERT_DEGRADATION
        assert retraining_state.get("retraining_trigger") != "active"

    def test_precision_079_emits_urgent_alert_and_sets_retraining_trigger(
        self, alert_client
    ):
        """precision = 0.79 → urgent alert AND retraining_trigger = 'active'."""
        # 24 TP / 30 FP+TP where FP = 30-24 = 6  BUT we want 0.79:
        # 24/30 = 0.8 — too high. Use 24 TP, 6 FP but tweak ratio:
        # 23 TP / 29 = ~0.793, still ≥0.80. Let's go 22 TP / 28 = ~0.786.
        # Actually want < 0.80: use 23 TP + 8 FP = 31 → 23/31 ≈ 0.742 < 0.80 ✓
        store = _build_feature_store(n_true_positives=23, n_false_positives=8)
        retraining_state: dict = {}
        svc = MonitoringService(
            store, alert_client, sleep_fn=_no_sleep,
            retraining_state=retraining_state,
        )
        svc.run_post_pipeline_monitoring(AS_OF)

        alert_client.assert_called_once()
        alert_type_called = alert_client.call_args[0][0]
        assert alert_type_called == ALERT_URGENT
        assert retraining_state.get("retraining_trigger") == "active"

    def test_precision_above_085_no_alert(self, alert_client):
        """precision ≥ 0.85 → no alert emitted."""
        # 27 TP, 3 FP = 30 total → precision = 0.9
        store = _build_feature_store(n_true_positives=27, n_false_positives=3)
        retraining_state: dict = {}
        svc = MonitoringService(
            store, alert_client, sleep_fn=_no_sleep,
            retraining_state=retraining_state,
        )
        svc.run_post_pipeline_monitoring(AS_OF)

        alert_client.assert_not_called()
        assert "retraining_trigger" not in retraining_state

    def test_none_precision_no_alert(self, alert_client):
        """When precision is None (< 30 claims) no alerts are emitted."""
        store = _build_feature_store(n_true_positives=10, n_false_positives=5)
        svc = MonitoringService(store, alert_client, sleep_fn=_no_sleep)
        svc.run_post_pipeline_monitoring(AS_OF)
        alert_client.assert_not_called()

    def test_exactly_080_is_degradation_not_urgent(self, alert_client):
        """Precision exactly 0.80 is in [0.80, 0.85) — degradation alert, no trigger."""
        # 24 TP / 30 total → 0.8
        store = _build_feature_store(n_true_positives=24, n_false_positives=6)
        retraining_state: dict = {}
        svc = MonitoringService(
            store, alert_client, sleep_fn=_no_sleep,
            retraining_state=retraining_state,
        )
        svc.run_post_pipeline_monitoring(AS_OF)

        alert_client.assert_called_once()
        assert alert_client.call_args[0][0] == ALERT_DEGRADATION
        assert retraining_state.get("retraining_trigger") != "active"

    def test_exactly_085_no_alert(self, alert_client):
        """Precision exactly 0.85 is NOT below 0.85 — no alert."""
        # Need precision exactly 0.85; 34 TP, 6 FP = 40 total → 34/40 = 0.85
        store = _build_feature_store(n_true_positives=34, n_false_positives=6)
        retraining_state: dict = {}
        svc = MonitoringService(
            store, alert_client, sleep_fn=_no_sleep,
            retraining_state=retraining_state,
        )
        svc.run_post_pipeline_monitoring(AS_OF)
        alert_client.assert_not_called()


# ===========================================================================
# Alert Retry Logic — Requirement 12.6
# ===========================================================================

class TestAlertRetry:
    """
    Delivery fails × 3 → final delivery-failure record logged; no exception raised.
    Success on 2nd attempt → no final-failure log; sleep called once.
    """

    def test_all_retries_exhausted_logs_delivery_failure(self, caplog):
        """After 3 consecutive failures, a DELIVERY FAILURE error is logged."""
        failing_client = MagicMock(side_effect=RuntimeError("network error"))
        sleep_calls: list = []

        def _track_sleep(s):
            sleep_calls.append(s)

        svc = MonitoringService(
            {},
            failing_client,
            alert_retry_count=3,
            alert_retry_interval_seconds=0,
            sleep_fn=_track_sleep,
        )

        with caplog.at_level(logging.ERROR, logger="claim_denial.monitoring.monitoring_service"):
            svc._dispatch_alert(ALERT_DEGRADATION, 0.81)

        assert failing_client.call_count == 3
        assert len(sleep_calls) == 2  # sleep between attempt 1→2 and 2→3; not after 3
        assert any("DELIVERY FAILURE" in record.message for record in caplog.records)

    def test_success_on_second_attempt_no_failure_log(self, caplog):
        """Alert succeeds on the 2nd try — no failure log entry; sleep called once."""
        call_count = 0

        def _flaky_client(alert_type, message):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first attempt fails")
            # second attempt succeeds (no exception)

        sleep_calls: list = []

        def _track_sleep(s):
            sleep_calls.append(s)

        svc = MonitoringService(
            {},
            _flaky_client,
            alert_retry_count=3,
            alert_retry_interval_seconds=0,
            sleep_fn=_track_sleep,
        )

        with caplog.at_level(logging.ERROR, logger="claim_denial.monitoring.monitoring_service"):
            svc._dispatch_alert(ALERT_DEGRADATION, 0.82)

        assert call_count == 2
        assert len(sleep_calls) == 1
        assert not any("DELIVERY FAILURE" in record.message for record in caplog.records)

    def test_immediate_success_no_sleep_no_failure_log(self, alert_client, caplog):
        """Alert succeeds immediately — sleep never called; no failure log."""
        sleep_calls: list = []

        svc = MonitoringService(
            {},
            alert_client,
            alert_retry_count=3,
            alert_retry_interval_seconds=300,
            sleep_fn=lambda s: sleep_calls.append(s),
        )

        with caplog.at_level(logging.ERROR, logger="claim_denial.monitoring.monitoring_service"):
            svc._dispatch_alert(ALERT_URGENT, 0.75)

        alert_client.assert_called_once()
        assert len(sleep_calls) == 0
        assert not any("DELIVERY FAILURE" in record.message for record in caplog.records)

    def test_retry_interval_passed_to_sleep(self):
        """The configured interval is forwarded to the sleep function."""
        call_count = 0

        def _always_fail(alert_type, message):
            raise RuntimeError("fail")

        sleep_calls: list = []

        svc = MonitoringService(
            {},
            _always_fail,
            alert_retry_count=3,
            alert_retry_interval_seconds=42,
            sleep_fn=lambda s: sleep_calls.append(s),
        )
        svc._dispatch_alert(ALERT_DEGRADATION, 0.81)
        # sleep is called between attempts 1→2 and 2→3 with the configured interval
        assert sleep_calls == [42, 42]


# ===========================================================================
# write_monitoring_dashboard — Requirement 12.3
# ===========================================================================

class TestWriteMonitoringDashboard:
    """MonitoringRecord is persisted to the dashboard store."""

    def test_record_appended_to_list_store(self, service):
        store: list = []
        service._dashboard_store = store
        record = MonitoringRecord(
            computation_date=AS_OF,
            rolling_precision=0.88,
            window_days=30,
            total_scored_claims=50,
            predicted_denials=20,
            true_positive_denials=18,
            false_positive_denials=2,
            computation_timestamp_utc=datetime.now(tz=timezone.utc),
        )
        service.write_monitoring_dashboard(record)
        assert len(store) == 1
        assert store[0] is record

    def test_dict_input_accepted(self, service):
        store: list = []
        service._dashboard_store = store
        metrics = {
            "computation_date": AS_OF,
            "rolling_precision": 0.91,
            "window_days": 30,
            "total_scored_claims": 60,
            "predicted_denials": 25,
            "true_positive_denials": 23,
            "false_positive_denials": 2,
            "computation_timestamp_utc": datetime.now(tz=timezone.utc),
        }
        service.write_monitoring_dashboard(metrics)
        assert len(store) == 1
        assert isinstance(store[0], MonitoringRecord)
        assert store[0].rolling_precision == 0.91

    def test_multiple_entries_accumulate(self, service):
        store: list = []
        service._dashboard_store = store
        for day_offset in range(3):
            record = MonitoringRecord(
                computation_date=AS_OF - timedelta(days=day_offset),
                rolling_precision=0.9 - day_offset * 0.01,
                window_days=30,
                total_scored_claims=50,
                predicted_denials=20,
                true_positive_denials=18,
                false_positive_denials=2,
                computation_timestamp_utc=datetime.now(tz=timezone.utc),
            )
            service.write_monitoring_dashboard(record)
        assert len(store) == 3


# ===========================================================================
# run_post_pipeline_monitoring — integration smoke tests
# ===========================================================================

class TestRunPostPipelineMonitoring:
    """Integration-level tests covering the orchestration entry point."""

    def test_dashboard_entry_written_even_when_precision_is_none(self, alert_client):
        """Fewer than 30 claims → dashboard entry written with precision=None."""
        store: list = []
        svc = MonitoringService(
            feature_store={},
            alert_client=alert_client,
            dashboard_store=store,
            sleep_fn=_no_sleep,
        )
        svc.run_post_pipeline_monitoring(AS_OF)
        assert len(store) == 1
        assert store[0].rolling_precision is None

    def test_dashboard_entry_written_with_computed_precision(self, alert_client):
        fs = _build_feature_store(n_true_positives=27, n_false_positives=3)
        dashboard: list = []
        svc = MonitoringService(
            feature_store=fs,
            alert_client=alert_client,
            dashboard_store=dashboard,
            sleep_fn=_no_sleep,
        )
        svc.run_post_pipeline_monitoring(AS_OF)
        assert len(dashboard) == 1
        assert dashboard[0].rolling_precision is not None
        assert abs(dashboard[0].rolling_precision - 0.9) < 1e-9

    def test_window_days_is_30(self, alert_client):
        dashboard: list = []
        svc = MonitoringService(
            feature_store={},
            alert_client=alert_client,
            dashboard_store=dashboard,
            sleep_fn=_no_sleep,
        )
        svc.run_post_pipeline_monitoring(AS_OF)
        assert dashboard[0].window_days == 30


# ===========================================================================
# _parse_date utility
# ===========================================================================

class TestParseDate:
    def test_iso_date_string(self):
        assert _parse_date("2024-06-15") == date(2024, 6, 15)

    def test_iso_datetime_with_tz(self):
        assert _parse_date("2024-06-15T10:30:00+00:00") == date(2024, 6, 15)

    def test_iso_datetime_without_tz(self):
        assert _parse_date("2024-06-15T10:30:00") == date(2024, 6, 15)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_date("not-a-date")
