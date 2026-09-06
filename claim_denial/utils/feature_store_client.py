"""
FeatureStoreClient — authoritative Feature Store abstraction.

Wraps HBase/Cassandra operations but uses an in-memory dict backing store by
default so all pipeline components (BatchScorer, MonitoringService,
FeatureEngineers, etc.) and unit tests can run without external infrastructure.

Requirements: 14.2, 14.5
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Retention threshold in days (Requirement 14.5)
_DEFAULT_RETENTION_DAYS: int = 90

# Minimum number of timestamped records required before a retention alert can
# be emitted (fewer → log warning only, no alert).
_MIN_TIMESTAMPED_RECORDS_FOR_ALERT: int = 2

# Alert type used for retention window violations.
ALERT_TYPE_RETENTION = "retention_warning"


# ---------------------------------------------------------------------------
# Timestamp utility  (mirrors monitoring_service._parse_date)
# ---------------------------------------------------------------------------

def _parse_date(ts: str) -> date:
    """
    Parse an ISO-8601 timestamp or date string into a ``date`` object.

    Handles full datetime strings and plain date strings.

    Raises
    ------
    ValueError
        When the string cannot be parsed.
    """
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(ts, fmt).date()
        except ValueError:
            continue

    # Last-resort fromisoformat (Python 3.7+)
    try:
        return datetime.fromisoformat(ts).date()
    except ValueError:
        pass

    raise ValueError(f"Cannot parse timestamp: {ts!r}")


# ---------------------------------------------------------------------------
# FeatureStoreClient
# ---------------------------------------------------------------------------

class FeatureStoreClient:
    """
    Feature Store abstraction for the Claim Denial Prediction pipeline.

    Uses an in-memory dict by default; swap in a real HBase/Cassandra client
    via the ``backend`` parameter for production use.

    Parameters
    ----------
    backend:
        Optional dict-like or HBase/Cassandra client.  When ``None`` an
        in-memory ``dict`` is used.
    alert_client:
        Optional callable ``(alert_type: str, message: str) → None`` used to
        deliver retention alerts.
    max_retries:
        Maximum number of write attempts before giving up (default 3).
    backoff_base_seconds:
        Back-off base in seconds; delay = ``backoff_base_seconds * 2^(attempt-1)``
        → 1 s, 2 s, 4 s with the default of 1.0.
    sleep_fn:
        Injected sleep function; defaults to ``time.sleep``.  Override in
        tests to avoid real sleeps.
    retention_days:
        Number of days used as the retention window threshold (default 90).
    """

    def __init__(
        self,
        backend=None,
        alert_client: Optional[Callable[[str, str], None]] = None,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        sleep_fn: Optional[Callable[[float], None]] = None,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        self._store: Dict[str, Dict[str, Any]] = backend if backend is not None else {}
        self._alert_client = alert_client
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        self._retention_days = retention_days

    # ------------------------------------------------------------------
    # upsert  (Requirement 14.2)
    # ------------------------------------------------------------------

    def upsert(self, claim_id: str, columns: Dict[str, Any]) -> None:
        """
        Merge *columns* into the Feature Store row keyed by *claim_id*.

        Creates the row if it does not exist.  Retries up to ``max_retries``
        times with exponential back-off on any backend exception.  After all
        retries are exhausted the failure is logged (ERROR) and processing
        continues — no exception is re-raised (matches BatchScorer behaviour).

        Requirement 14.2
        """
        delay = self._backoff_base
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                if claim_id not in self._store:
                    self._store[claim_id] = {}
                self._store[claim_id].update(columns)
                if attempt > 1:
                    logger.info(
                        "FeatureStoreClient.upsert: write succeeded for "
                        "claim_id='%s' on attempt %d.",
                        claim_id,
                        attempt,
                    )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "FeatureStoreClient.upsert: attempt %d/%d failed for "
                        "claim_id='%s' (retrying in %.1fs): %s",
                        attempt,
                        self._max_retries,
                        claim_id,
                        delay,
                        exc,
                    )
                    self._sleep(delay)
                    delay *= 2

        # All retries exhausted — log structured error, do NOT re-raise.
        logger.error(
            "FeatureStoreClient.upsert: all %d write attempts failed for "
            "claim_id='%s': %s",
            self._max_retries,
            claim_id,
            last_exc,
        )

    # ------------------------------------------------------------------
    # get  (Requirement 14.2)
    # ------------------------------------------------------------------

    def get(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the Feature Store row dict for *claim_id*, or ``None`` if absent.

        Requirement 14.2
        """
        return self._store.get(claim_id)

    # ------------------------------------------------------------------
    # scan_active_claims  (Requirement 14.2)
    # ------------------------------------------------------------------

    def scan_active_claims(self) -> List[Dict[str, Any]]:
        """
        Return all rows where ``claim_status == "Active"``.

        Each returned dict includes at minimum the ``claim_id`` and
        ``claim_status`` keys.

        Requirement 14.2
        """
        results: List[Dict[str, Any]] = []
        for claim_id, row in self._store.items():
            if row.get("claim_status") == "Active":
                # Ensure claim_id is in the returned dict.
                enriched = dict(row)
                enriched["claim_id"] = claim_id
                results.append(enriched)
        return results

    # ------------------------------------------------------------------
    # check_retention_window  (Requirement 14.5)
    # ------------------------------------------------------------------

    def check_retention_window(self) -> None:
        """
        Examine all rows and find the oldest record by timestamp.

        Logic
        -----
        - For each row, parse ``scoring_timestamp_utc`` or
          ``adjudication_timestamp_utc`` (whichever is present and parseable)
          to determine the record date.
        - If fewer than 2 rows have a parseable timestamp, log a warning and
          do NOT emit an alert.
        - Otherwise, compare the oldest record's date against
          ``datetime.now(UTC) - retention_days``:
          - Older than threshold → emit a retention alert via ``alert_client``
            (if configured).
          - Within threshold → no alert.
        - Always log a structured message with the oldest record date, the
          threshold date, and whether an alert was emitted.

        Requirement 14.5
        """
        now_utc: date = datetime.now(tz=timezone.utc).date()
        threshold: date = now_utc - __import__("datetime").timedelta(days=self._retention_days)

        timestamped_dates: List[date] = []
        for row in self._store.values():
            ts_str = row.get("scoring_timestamp_utc") or row.get("adjudication_timestamp_utc")
            if ts_str is None:
                continue
            try:
                rec_date = _parse_date(str(ts_str))
                timestamped_dates.append(rec_date)
            except (ValueError, TypeError):
                continue

        if len(timestamped_dates) < _MIN_TIMESTAMPED_RECORDS_FOR_ALERT:
            logger.warning(
                "FeatureStoreClient.check_retention_window: only %d record(s) have a "
                "parseable timestamp (need >= %d); skipping retention alert.",
                len(timestamped_dates),
                _MIN_TIMESTAMPED_RECORDS_FOR_ALERT,
            )
            return

        oldest_date: date = min(timestamped_dates)
        alert_emitted: bool = oldest_date < threshold

        logger.info(
            "FeatureStoreClient.check_retention_window: oldest_record_date=%s "
            "threshold=%s retention_days=%d alert_emitted=%s",
            oldest_date.isoformat(),
            threshold.isoformat(),
            self._retention_days,
            alert_emitted,
        )

        if alert_emitted:
            message = (
                f"Retention window violation: oldest record date {oldest_date.isoformat()} "
                f"is more than {self._retention_days} days before {now_utc.isoformat()} "
                f"(threshold: {threshold.isoformat()})."
            )
            if self._alert_client is not None:
                self._alert_client(ALERT_TYPE_RETENTION, message)
            else:
                logger.warning(
                    "FeatureStoreClient.check_retention_window: retention alert triggered "
                    "but no alert_client configured. Message: %s",
                    message,
                )

    # ------------------------------------------------------------------
    # delete  (test helper)
    # ------------------------------------------------------------------

    def delete(self, claim_id: str) -> None:
        """
        Remove the row for *claim_id* entirely.  No-op if absent.

        Primarily used in test teardown.
        """
        self._store.pop(claim_id, None)
