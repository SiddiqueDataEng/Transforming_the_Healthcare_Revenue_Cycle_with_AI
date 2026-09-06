"""
MonitoringService — post-deployment monitoring for the Claim Denial Prediction system.

Responsibilities
----------------
- Link ERA/835 adjudication outcomes to Feature Store claim records (Req 12.1).
- Compute rolling 30-day Precision (Req 12.2).
- Write monitoring dashboard entries (Req 12.3).
- Emit degradation / urgent alerts with retry logic (Req 12.4, 12.5, 12.6).
- Trigger retraining when precision drops below 0.80 (Req 12.5).

All patient data is synthetic; no real PHI is processed here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from claim_denial.models import AdjudicationStatus, MonitoringRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert type constants
# ---------------------------------------------------------------------------

ALERT_DEGRADATION = "degradation"   # precision in [0.80, 0.85)
ALERT_URGENT = "urgent"             # precision < 0.80


# ---------------------------------------------------------------------------
# MonitoringService
# ---------------------------------------------------------------------------

class MonitoringService:
    """
    Tracks real-world model precision against ERA/835 adjudication outcomes.

    Parameters
    ----------
    feature_store:
        Dict-like mapping claim_id → dict of claim feature fields.
        Must support __getitem__, __setitem__, __contains__, and .values().
    alert_client:
        Callable(alert_type: str, message: str) → None.
        Injected so unit tests can mock alert delivery.
    dashboard_store:
        Optional list or dict-like store for persisting MonitoringRecord entries.
        If None, records are only logged.
    alert_retry_count:
        Number of retry attempts after the initial failure (default 3).
    alert_retry_interval_seconds:
        Seconds to wait between retries (default 300 = 5 minutes).
    sleep_fn:
        Injected sleep function for testability (defaults to time.sleep).
    retraining_state:
        Optional mutable dict where ``retraining_trigger`` key is set to
        ``"active"`` when precision drops below 0.80.  If not provided an
        internal dict is created.
    """

    def __init__(
        self,
        feature_store,
        alert_client: Callable[[str, str], None],
        dashboard_store=None,
        alert_retry_count: int = 3,
        alert_retry_interval_seconds: int = 300,
        sleep_fn: Callable[[float], None] = time.sleep,
        retraining_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._feature_store = feature_store
        self._alert_client = alert_client
        self._dashboard_store = dashboard_store if dashboard_store is not None else []
        self._alert_retry_count = alert_retry_count
        self._alert_retry_interval_seconds = alert_retry_interval_seconds
        self._sleep = sleep_fn
        self.retraining_state: Dict[str, Any] = (
            retraining_state if retraining_state is not None else {}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def link_adjudication_outcome(self, era_record: Dict[str, Any]) -> None:
        """
        Link an ERA/835 adjudication outcome to the matching Feature Store row.

        Parameters
        ----------
        era_record:
            Dict with at minimum:
              - ``claim_id`` (str)
              - ``adjudication_status`` (str) — "paid", "denied", "pending", …
              - ``processed_timestamp_utc`` (str | datetime) — ERA processing time

        The method writes two fields to the Feature Store row keyed by
        ``claim_id``:
          - ``adjudication_outcome``
          - ``adjudication_timestamp_utc``

        Requirement 12.1
        """
        claim_id: str = era_record["claim_id"]
        adjudication_status: str = era_record["adjudication_status"]
        processed_ts = era_record["processed_timestamp_utc"]

        # Normalise timestamp to ISO-8601 string
        if isinstance(processed_ts, datetime):
            ts_str = processed_ts.isoformat()
        else:
            ts_str = str(processed_ts)

        if claim_id not in self._feature_store:
            logger.warning(
                "link_adjudication_outcome: claim_id=%s not found in Feature Store; "
                "creating stub row.",
                claim_id,
            )
            self._feature_store[claim_id] = {}

        self._feature_store[claim_id]["adjudication_outcome"] = adjudication_status
        self._feature_store[claim_id]["adjudication_timestamp_utc"] = ts_str

        logger.info(
            "Linked adjudication outcome claim_id=%s status=%s ts=%s",
            claim_id,
            adjudication_status,
            ts_str,
        )

    def compute_rolling_precision(self, as_of_date: date) -> Optional[float]:
        """
        Compute rolling 30-day Precision anchored at ``as_of_date``.

        A claim is included when:
          - It has a ``predicted_denial_score`` (not None).
          - It has a *known* adjudication outcome — "paid" or "denied"
            (pending / adjusted are excluded).
          - Its ``scoring_timestamp_utc`` (or ``adjudication_timestamp_utc``)
            falls within the 30-day window ending at ``as_of_date``.

        Returns ``None`` when fewer than 30 qualifying claims exist.

        Precision = true_positives / (true_positives + false_positives)
        where a *predicted denial* is ``predicted_denial_score >= 0.5``.

        Requirement 12.2
        """
        window_start = as_of_date - timedelta(days=30)

        qualifying: List[Dict[str, Any]] = []
        for row in self._feature_store.values():
            score = row.get("predicted_denial_score")
            outcome = row.get("adjudication_outcome")

            if score is None:
                continue
            if outcome not in (AdjudicationStatus.DENIED.value, AdjudicationStatus.PAID.value,
                               "denied", "paid"):
                continue

            # Determine the reference date for window filtering.
            # Prefer scoring_timestamp_utc; fall back to adjudication_timestamp_utc.
            ts_str = row.get("scoring_timestamp_utc") or row.get("adjudication_timestamp_utc")
            if ts_str is None:
                continue
            try:
                ref_date = _parse_date(ts_str)
            except (ValueError, TypeError):
                logger.warning(
                    "compute_rolling_precision: cannot parse timestamp '%s'; skipping row.",
                    ts_str,
                )
                continue

            if window_start < ref_date <= as_of_date:
                qualifying.append(row)

        if len(qualifying) < 30:
            logger.info(
                "compute_rolling_precision: only %d qualifying claims in window "
                "(need ≥30); returning None.",
                len(qualifying),
            )
            return None

        true_positives = 0
        false_positives = 0
        for row in qualifying:
            score = row["predicted_denial_score"]
            outcome = row["adjudication_outcome"]
            predicted_denial = score >= 0.5
            actual_denial = outcome in (AdjudicationStatus.DENIED.value, "denied")

            if predicted_denial and actual_denial:
                true_positives += 1
            elif predicted_denial and not actual_denial:
                false_positives += 1

        denominator = true_positives + false_positives
        if denominator == 0:
            # No predicted denials at all → precision is undefined; return None
            logger.info(
                "compute_rolling_precision: no predicted denials in window; "
                "returning None."
            )
            return None

        precision = true_positives / denominator
        logger.info(
            "compute_rolling_precision: as_of=%s window_start=%s "
            "total=%d TP=%d FP=%d precision=%.4f",
            as_of_date, window_start, len(qualifying),
            true_positives, false_positives, precision,
        )
        return precision

    def write_monitoring_dashboard(
        self, metrics: Union[MonitoringRecord, Dict[str, Any]]
    ) -> None:
        """
        Persist a MonitoringRecord (or equivalent dict) to the dashboard store.

        Writes the full monitoring schema:
          computation_date, rolling_precision, window_days, total_scored_claims,
          predicted_denials, true_positive_denials, false_positive_denials,
          computation_timestamp_utc.

        Requirement 12.3
        """
        if isinstance(metrics, MonitoringRecord):
            record = metrics
        else:
            # Accept dict input for flexibility
            record = MonitoringRecord(
                computation_date=metrics["computation_date"],
                rolling_precision=metrics.get("rolling_precision"),
                window_days=metrics.get("window_days", 30),
                total_scored_claims=metrics.get("total_scored_claims", 0),
                predicted_denials=metrics.get("predicted_denials", 0),
                true_positive_denials=metrics.get("true_positive_denials", 0),
                false_positive_denials=metrics.get("false_positive_denials", 0),
                computation_timestamp_utc=metrics.get(
                    "computation_timestamp_utc",
                    datetime.now(tz=timezone.utc),
                ),
            )

        if isinstance(self._dashboard_store, list):
            self._dashboard_store.append(record)
        elif hasattr(self._dashboard_store, "__setitem__"):
            key = str(record.computation_date)
            self._dashboard_store[key] = record
        else:
            logger.warning("write_monitoring_dashboard: unrecognised store type; record only logged.")

        logger.info(
            "Monitoring dashboard entry written: date=%s precision=%s total_claims=%d",
            record.computation_date,
            record.rolling_precision,
            record.total_scored_claims,
        )

    def run_post_pipeline_monitoring(self, as_of_date: date) -> None:
        """
        Main orchestration entry point — called after nightly pipeline completion.

        Steps:
          1. Compute rolling 30-day Precision.
          2. Collect window statistics.
          3. Write monitoring dashboard entry.
          4. Emit alerts and/or set retraining trigger based on thresholds.

        Must complete within 30 minutes of pipeline completion (Req 12.4, 12.5).
        """
        window_start = as_of_date - timedelta(days=30)

        # Collect window statistics in one pass
        total_scored = 0
        predicted_denials = 0
        true_positives = 0
        false_positives = 0

        for row in self._feature_store.values():
            score = row.get("predicted_denial_score")
            outcome = row.get("adjudication_outcome")

            if score is None:
                continue

            # Only claims with a timestamp in the window contribute to stats
            ts_str = row.get("scoring_timestamp_utc") or row.get("adjudication_timestamp_utc")
            if ts_str is None:
                continue
            try:
                ref_date = _parse_date(ts_str)
            except (ValueError, TypeError):
                continue

            if not (window_start < ref_date <= as_of_date):
                continue

            # Must have a known outcome to contribute to precision stats
            if outcome not in (AdjudicationStatus.DENIED.value, AdjudicationStatus.PAID.value,
                               "denied", "paid"):
                continue

            total_scored += 1
            predicted_denial = score >= 0.5
            actual_denial = outcome in (AdjudicationStatus.DENIED.value, "denied")

            if predicted_denial:
                predicted_denials += 1
                if actual_denial:
                    true_positives += 1
                else:
                    false_positives += 1

        rolling_precision = self.compute_rolling_precision(as_of_date)

        record = MonitoringRecord(
            computation_date=as_of_date,
            rolling_precision=rolling_precision,
            window_days=30,
            total_scored_claims=total_scored,
            predicted_denials=predicted_denials,
            true_positive_denials=true_positives,
            false_positive_denials=false_positives,
            computation_timestamp_utc=datetime.now(tz=timezone.utc),
        )
        self.write_monitoring_dashboard(record)

        if rolling_precision is None:
            logger.info(
                "run_post_pipeline_monitoring: precision is None (insufficient data); "
                "no alerts emitted."
            )
            return

        if rolling_precision < 0.80:
            logger.warning(
                "run_post_pipeline_monitoring: URGENT — precision=%.4f < 0.80; "
                "emitting urgent alert and setting retraining_trigger=active.",
                rolling_precision,
            )
            self._dispatch_alert(ALERT_URGENT, rolling_precision)
            self.retraining_state["retraining_trigger"] = "active"
            logger.info("retraining_trigger set to 'active'.")

        elif rolling_precision < 0.85:
            logger.warning(
                "run_post_pipeline_monitoring: DEGRADATION — precision=%.4f in [0.80, 0.85); "
                "emitting degradation alert.",
                rolling_precision,
            )
            self._dispatch_alert(ALERT_DEGRADATION, rolling_precision)

        else:
            logger.info(
                "run_post_pipeline_monitoring: precision=%.4f — within acceptable range; "
                "no alerts emitted.",
                rolling_precision,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_alert(self, alert_type: str, precision_value: float) -> None:
        """
        Deliver an alert, retrying up to ``alert_retry_count`` times on failure.

        Retries are separated by ``alert_retry_interval_seconds`` (default 5 min).
        After exhausting all retries a final delivery-failure record is logged.

        Requirement 12.6
        """
        message = (
            f"Claim denial model alert [{alert_type.upper()}]: "
            f"rolling 30-day precision = {precision_value:.4f}"
        )

        for attempt in range(1, self._alert_retry_count + 1):
            try:
                self._alert_client(alert_type, message)
                logger.info(
                    "_dispatch_alert: alert_type=%s delivered on attempt %d",
                    alert_type, attempt,
                )
                return  # success — stop retrying
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_dispatch_alert: attempt %d/%d failed for alert_type=%s: %s",
                    attempt, self._alert_retry_count, alert_type, exc,
                )
                if attempt < self._alert_retry_count:
                    self._sleep(self._alert_retry_interval_seconds)

        # All retries exhausted
        logger.error(
            "_dispatch_alert: DELIVERY FAILURE — alert_type=%s precision=%.4f "
            "failed after %d attempt(s). Final delivery-failure record logged.",
            alert_type, precision_value, self._alert_retry_count,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _parse_date(ts: str) -> date:
    """
    Parse an ISO-8601 timestamp string or date string into a ``date`` object.

    Handles both full datetime strings (``2024-03-15T01:23:45+00:00``) and
    plain date strings (``2024-03-15``).
    """
    # Try full datetime first
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.date()
        except ValueError:
            continue

    # Last resort: fromisoformat (Python 3.7+)
    try:
        return datetime.fromisoformat(ts).date()
    except ValueError:
        pass

    raise ValueError(f"Cannot parse timestamp: {ts!r}")
