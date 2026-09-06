"""
Payer Behavior Feature Engineering

Computes payer-level denial tendency features for the Claim Denial Prediction
pipeline.  All methods are pure (or near-pure) functions for testability.

Feature Store is a local in-process dict (str → dict) for unit-test portability;
production code replaces `InMemoryFeatureStore` with a real HBase / Cassandra
client that exposes the same `.upsert(claim_id, columns)` interface.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature Store protocol — thin interface so tests can inject a mock or
# use the provided InMemoryFeatureStore without depending on Cassandra/HBase.
# ---------------------------------------------------------------------------


class FeatureStoreProtocol(Protocol):
    """Minimal write interface for the Feature Store."""

    def upsert(self, claim_id: str, columns: Dict[str, Any]) -> None:
        """Upsert *columns* into the row keyed by *claim_id*."""
        ...


class InMemoryFeatureStore:
    """
    Local dict-backed Feature Store for unit tests and development.

    Thread-safety is intentionally omitted here; add a lock if needed.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def upsert(self, claim_id: str, columns: Dict[str, Any]) -> None:
        if claim_id not in self._store:
            self._store[claim_id] = {}
        self._store[claim_id].update(columns)

    def get(self, claim_id: str) -> Dict[str, Any]:
        """Return a copy of the stored row (empty dict if absent)."""
        return dict(self._store.get(claim_id, {}))

    def __contains__(self, claim_id: str) -> bool:  # pragma: no cover
        return claim_id in self._store


# ---------------------------------------------------------------------------
# Typed claim record used by window-based computations.
# ---------------------------------------------------------------------------


class WindowClaim:
    """
    Lightweight representation of a historical claim record used inside the
    90-day trailing window.

    Attributes
    ----------
    payer_id:
        Payer identifier matching `ClaimRecord.payer_id`.
    claim_type:
        "professional" or "institutional" (lower-case); ``None`` when absent.
    adjudication_status:
        "denied", "paid", or another status.  Only "denied" and "paid" are
        treated as *known-outcome* claims; everything else is excluded from
        rate computations.
    submission_date:
        The date the claim was submitted (used externally to build the window).
    """

    __slots__ = ("payer_id", "claim_type", "adjudication_status", "submission_date")

    def __init__(
        self,
        payer_id: str,
        claim_type: Optional[str],
        adjudication_status: str,
        submission_date: date,
    ) -> None:
        self.payer_id = payer_id
        self.claim_type = claim_type
        self.adjudication_status = adjudication_status.lower() if adjudication_status else ""
        self.submission_date = submission_date

    @property
    def is_known_outcome(self) -> bool:
        """True when adjudication status is either 'paid' or 'denied'."""
        return self.adjudication_status in ("paid", "denied")

    @property
    def is_denied(self) -> bool:
        return self.adjudication_status == "denied"


# ---------------------------------------------------------------------------
# ERA record used by days_since_last_payment.
# ---------------------------------------------------------------------------


class ERARecord:
    """
    Minimal ERA/835 record.

    Attributes
    ----------
    payer_id:
        Payer identifier.
    payment_date:
        The date the ERA was received / payment was made.
    status:
        "paid", "denied", "adjusted", or similar.  Only "paid" is considered
        for the days-since-last-payment computation.
    """

    __slots__ = ("payer_id", "payment_date", "status")

    def __init__(self, payer_id: str, payment_date: date, status: str) -> None:
        self.payer_id = payer_id
        self.payment_date = payment_date
        self.status = status.lower() if status else ""

    @property
    def is_paid(self) -> bool:
        return self.status == "paid"


# ---------------------------------------------------------------------------
# Contract reference entry used by stop-loss lookup.
# ---------------------------------------------------------------------------


class ContractEntry:
    """
    A single row from the payer contract reference table.

    Attributes
    ----------
    payer_id:
        Payer identifier.
    effective_date_start:
        First calendar date on which this contract row is in effect.
    effective_date_end:
        Last calendar date on which this contract row is in effect
        (inclusive).  ``None`` means the contract is open-ended / currently
        active.
    stop_loss_active:
        True when a stop-loss clause is active under this contract row.
    """

    __slots__ = ("payer_id", "effective_date_start", "effective_date_end", "stop_loss_active")

    def __init__(
        self,
        payer_id: str,
        effective_date_start: date,
        effective_date_end: Optional[date],
        stop_loss_active: bool,
    ) -> None:
        self.payer_id = payer_id
        self.effective_date_start = effective_date_start
        self.effective_date_end = effective_date_end
        self.stop_loss_active = stop_loss_active

    def covers(self, query_date: date) -> bool:
        """Return True when *query_date* falls within this contract row's range."""
        if query_date < self.effective_date_start:
            return False
        if self.effective_date_end is not None and query_date > self.effective_date_end:
            return False
        return True


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

#: Minimum number of known-outcome claims required for a per-payer rate;
#: below this threshold the population-mean fallback is used.
PAYER_MIN_CLAIM_THRESHOLD: int = 50


# ---------------------------------------------------------------------------
# PayerBehaviorFeatureEngineer
# ---------------------------------------------------------------------------


class PayerBehaviorFeatureEngineer:
    """
    Computes the four payer-behavior features and upserts them to the Feature
    Store.

    Parameters
    ----------
    feature_store:
        Any object implementing :class:`FeatureStoreProtocol`.  Defaults to a
        new :class:`InMemoryFeatureStore` instance when not provided.

    Usage
    -----
    >>> eng = PayerBehaviorFeatureEngineer()
    >>> features = eng.compute_all(
    ...     claim_id="CLM001",
    ...     payer_id="BCBS01",
    ...     claim_type="professional",
    ...     window_claims=[...],
    ...     era_records=[...],
    ...     submission_date=date(2024, 6, 1),
    ...     effective_date=date(2024, 6, 1),
    ...     contract_reference=[...],
    ... )
    """

    def __init__(
        self,
        feature_store: Optional[FeatureStoreProtocol] = None,
    ) -> None:
        self._feature_store: FeatureStoreProtocol = (
            feature_store if feature_store is not None else InMemoryFeatureStore()
        )

    # ------------------------------------------------------------------
    # 6.1  Payer historical denial rate (overall)
    # ------------------------------------------------------------------

    def compute_payer_historical_denial_rate(
        self,
        payer_id: str,
        window_claims: Sequence[WindowClaim],
    ) -> float:
        """
        Fraction of known-outcome claims denied by *payer_id* in the trailing
        90-day window.

        Requirement 6.1, 6.5

        Parameters
        ----------
        payer_id:
            Identifier of the payer to compute the rate for.
        window_claims:
            All claims (any payer) with known adjudication outcomes inside the
            trailing 90-day window.  The caller is responsible for applying the
            date filter before passing the list.

        Returns
        -------
        float
            A value in [0.0, 1.0].
            - If *payer_id* has **≥ 50** known-outcome claims → computed rate.
            - If *payer_id* has **< 50** known-outcome claims → population-mean
              denial rate computed from all payers that have at least one
              known outcome.
        """
        # Split into per-payer and all-known-outcome subsets
        payer_known: List[WindowClaim] = [
            c for c in window_claims if c.payer_id == payer_id and c.is_known_outcome
        ]

        if len(payer_known) >= PAYER_MIN_CLAIM_THRESHOLD:
            denied = sum(1 for c in payer_known if c.is_denied)
            rate = denied / len(payer_known)
        else:
            rate = self._population_mean_denial_rate(window_claims)

        return _clamp(rate)

    # ------------------------------------------------------------------
    # 6.2  Payer historical denial rate by claim type
    # ------------------------------------------------------------------

    def compute_payer_historical_denial_rate_by_type(
        self,
        payer_id: str,
        claim_type: Optional[str],
        window_claims: Sequence[WindowClaim],
    ) -> float:
        """
        Fraction of known-outcome claims of *claim_type* denied by *payer_id*
        over the trailing 90-day window.

        Requirement 6.2, 6.5

        Parameters
        ----------
        payer_id:
            Payer identifier.
        claim_type:
            "professional" or "institutional" (case-insensitive).  If ``None``
            or unrecognised, falls back to the population-mean denial rate.
        window_claims:
            Trailing 90-day window of claims with known outcomes.

        Returns
        -------
        float
            A value in [0.0, 1.0] — computed type-specific rate or population
            mean when *payer_id* has < 50 known-outcome claims for that type.
        """
        claim_type_norm = claim_type.lower() if claim_type else ""

        payer_type_known: List[WindowClaim] = [
            c
            for c in window_claims
            if c.payer_id == payer_id
            and (c.claim_type or "").lower() == claim_type_norm
            and c.is_known_outcome
        ]

        if len(payer_type_known) >= PAYER_MIN_CLAIM_THRESHOLD:
            denied = sum(1 for c in payer_type_known if c.is_denied)
            rate = denied / len(payer_type_known)
        else:
            rate = self._population_mean_denial_rate(window_claims)

        return _clamp(rate)

    # ------------------------------------------------------------------
    # 6.3  Days since last payment
    # ------------------------------------------------------------------

    def compute_days_since_last_payment(
        self,
        payer_id: str,
        era_records: Sequence[ERARecord],
        submission_date: date,
    ) -> int:
        """
        Calendar days between the most recent paid ERA record for *payer_id*
        and the claim *submission_date*.

        Requirement 6.3

        Parameters
        ----------
        payer_id:
            Payer identifier.
        era_records:
            All ERA/835 records available (any date, any payer).
        submission_date:
            The date the claim is being submitted.

        Returns
        -------
        int
            - Non-negative integer: days since the most recent paid ERA.
            - ``-1``: cold-start — no prior paid ERA exists for *payer_id*.

        Notes
        -----
        If a paid ERA has a `payment_date` *after* `submission_date` (data
        anomaly), the function still returns 0 (clamped, non-negative) for
        that record but picks the *most-recent-before-or-on* submission date
        when multiple paid ERAs exist.  If all paid ERAs are in the future,
        the function returns 0 for the most-recent one.
        """
        paid_eras: List[ERARecord] = [
            r for r in era_records if r.payer_id == payer_id and r.is_paid
        ]

        if not paid_eras:
            return -1  # cold-start sentinel

        # Prefer the most-recent paid ERA on or before the submission date.
        # If none are ≤ submission_date, fall back to the closest future date.
        eras_on_or_before = [r for r in paid_eras if r.payment_date <= submission_date]
        if eras_on_or_before:
            latest = max(eras_on_or_before, key=lambda r: r.payment_date)
        else:
            # All paid ERAs are in the future — take the closest one.
            latest = min(paid_eras, key=lambda r: r.payment_date)

        delta = (submission_date - latest.payment_date).days
        return max(0, delta)  # non-negative guarantee

    # ------------------------------------------------------------------
    # 6.4  Payer contract stop-loss lookup
    # ------------------------------------------------------------------

    def lookup_payer_contract_stop_loss(
        self,
        payer_id: str,
        effective_date: date,
        contract_reference: Sequence[ContractEntry],
    ) -> int:
        """
        Binary indicator of whether a stop-loss clause is active for *payer_id*
        on *effective_date*.

        Requirement 6.4

        Parameters
        ----------
        payer_id:
            Payer identifier to look up.
        effective_date:
            The contract effective date to evaluate (typically the claim
            submission date).
        contract_reference:
            Rows from the payer contract reference table.

        Returns
        -------
        int
            - ``1``: stop-loss clause is active for this payer on this date.
            - ``0``: either the clause is inactive OR the payer has no entry
              in the reference table (missing-payer-ID is logged at WARNING).
        """
        matching: List[ContractEntry] = [
            entry
            for entry in contract_reference
            if entry.payer_id == payer_id and entry.covers(effective_date)
        ]

        if not matching:
            # Distinguish between "payer absent entirely" and "payer present
            # but no contract covers this date".
            payer_exists = any(entry.payer_id == payer_id for entry in contract_reference)
            if not payer_exists:
                logger.warning(
                    "Payer ID '%s' is absent from the contract reference table; "
                    "defaulting payer_contract_stop_loss to 0.",
                    payer_id,
                )
            return 0

        # If multiple rows match (e.g., overlapping date ranges), stop-loss is
        # active if *any* matching row has stop_loss_active = True.
        return int(any(entry.stop_loss_active for entry in matching))

    # ------------------------------------------------------------------
    # Orchestration: compute all four features and upsert to Feature Store
    # ------------------------------------------------------------------

    def compute_all(
        self,
        claim_id: str,
        payer_id: str,
        claim_type: Optional[str],
        window_claims: Sequence[WindowClaim],
        era_records: Sequence[ERARecord],
        submission_date: date,
        effective_date: date,
        contract_reference: Sequence[ContractEntry],
    ) -> Dict[str, Any]:
        """
        Compute all four payer-behavior features, upsert them to the Feature
        Store, and return the computed column dict.

        Parameters
        ----------
        claim_id:
            Feature Store row key.
        payer_id:
            Payer identifier for the claim being engineered.
        claim_type:
            "professional" or "institutional".
        window_claims:
            Trailing 90-day window of claims with adjudication outcomes.
        era_records:
            ERA/835 records for days-since-last-payment computation.
        submission_date:
            Claim submission date (used for days-since-last-payment).
        effective_date:
            Contract effective date (used for stop-loss lookup).
        contract_reference:
            Payer contract reference table rows.

        Returns
        -------
        dict
            ``{"payer_historical_denial_rate": float,
               "payer_historical_denial_rate_by_type": float,
               "days_since_last_payment": int,
               "payer_contract_stop_loss": int}``
        """
        columns: Dict[str, Any] = {
            "payer_historical_denial_rate": self.compute_payer_historical_denial_rate(
                payer_id, window_claims
            ),
            "payer_historical_denial_rate_by_type": self.compute_payer_historical_denial_rate_by_type(
                payer_id, claim_type, window_claims
            ),
            "days_since_last_payment": self.compute_days_since_last_payment(
                payer_id, era_records, submission_date
            ),
            "payer_contract_stop_loss": self.lookup_payer_contract_stop_loss(
                payer_id, effective_date, contract_reference
            ),
        }

        self._feature_store.upsert(claim_id, columns)
        return columns

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _population_mean_denial_rate(window_claims: Sequence[WindowClaim]) -> float:
        """
        Mean denial rate across *all* payers that have at least one
        known-outcome claim in *window_claims*.

        Returns 0.5 (the neutral sentinel) when no payer has any known-outcome
        claim in the window.
        """
        known: List[WindowClaim] = [c for c in window_claims if c.is_known_outcome]
        if not known:
            return 0.5  # neutral sentinel — no data available
        denied = sum(1 for c in known if c.is_denied)
        return denied / len(known)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [*lo*, *hi*] — ensures denial rates stay in [0, 1]."""
    return max(lo, min(hi, value))
