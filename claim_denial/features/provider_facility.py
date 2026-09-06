"""
Provider and Facility Feature Engineering.

Implements :class:`ProviderFacilityFeatureEngineer` which computes the five
provider/facility features required by Requirements 4.1–4.9:

- ``billing_provider_id_encoded``  – leave-one-out target encoding (LOO) when
  the provider has >30 claims with known outcomes in the trailing 90-day
  window; population-mean fallback otherwise.
- ``performing_physician_specialty`` – NUCC taxonomy code from the NPI
  registry; "UNKNOWN" sentinel when absent or unavailable.
- ``provider_historical_denial_rate`` – fraction denied over trailing 90 days
  using only known-outcome claims; population-mean fallback when <30 claims.
- ``provider_claim_volume`` – count of ALL claims in trailing 90 days
  (regardless of adjudication status).
- ``facility_bed_size`` – integer from the facility master keyed by NPI;
  integer mean of all values when NPI is absent.

All patient/provider data used in tests must be synthetic (no real PHI/NPI).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from statistics import mean
from typing import Callable, Dict, List, Optional

from claim_denial.models import AdjudicationStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A "window claim" dict carries at minimum:
#   npi              : str
#   submission_date  : date
#   adjudication_outcome : str | None   (None = still pending / unknown)
WindowClaim = Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAILING_WINDOW_DAYS: int = 90
LOO_ENCODING_MIN_CLAIMS: int = 30  # >30 triggers LOO; ≤30 triggers pop-mean
PROVIDER_DENIAL_RATE_MIN_CLAIMS: int = 30  # <30 triggers pop-mean

KNOWN_OUTCOME_STATUSES = frozenset(
    {AdjudicationStatus.DENIED.value, AdjudicationStatus.PAID.value}
)

SENTINEL_SPECIALTY_UNKNOWN: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Helper: window filter
# ---------------------------------------------------------------------------


def _in_trailing_window(submission_date: date, reference_date: date) -> bool:
    """Return True when *submission_date* falls within the trailing 90-day window
    relative to *reference_date* (inclusive on both ends)."""
    window_start = reference_date - timedelta(days=TRAILING_WINDOW_DAYS)
    return window_start <= submission_date <= reference_date


def _known_outcome(claim: WindowClaim) -> bool:
    """Return True when the claim has a fully-adjudicated (paid or denied) outcome."""
    return claim.get("adjudication_outcome") in KNOWN_OUTCOME_STATUSES


def _is_denied(claim: WindowClaim) -> bool:
    """Return True when the claim has a denied adjudication outcome."""
    return claim.get("adjudication_outcome") == AdjudicationStatus.DENIED.value


# ---------------------------------------------------------------------------
# Feature Store mock
# ---------------------------------------------------------------------------


class InMemoryFeatureStore:
    """
    Lightweight dict-backed Feature Store mock for local / test use.

    In production this would be replaced by an HBase or Cassandra client.
    Keys are Claim IDs; values are dicts of feature columns.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict] = {}

    def upsert(self, claim_id: str, columns: Dict) -> None:
        """Merge *columns* into the existing row for *claim_id* (creates if absent)."""
        if claim_id not in self._store:
            self._store[claim_id] = {}
        self._store[claim_id].update(columns)

    def get(self, claim_id: str) -> Optional[Dict]:
        """Return the feature row for *claim_id*, or None if not present."""
        return self._store.get(claim_id)

    def get_all(self) -> Dict[str, Dict]:
        """Return a copy of the entire store (for inspection / testing)."""
        return dict(self._store)


# ---------------------------------------------------------------------------
# NPI Registry abstraction
# ---------------------------------------------------------------------------


class NpiRegistry:
    """
    Thin abstraction over an NPI taxonomy-code lookup.

    The default implementation is a no-op (returns UNKNOWN for every NPI).
    Inject a concrete implementation or a callable for production/testing use.
    """

    def lookup_specialty(self, npi: str) -> str:  # pragma: no cover
        """Return the NUCC taxonomy code for *npi*, or SENTINEL_SPECIALTY_UNKNOWN."""
        return SENTINEL_SPECIALTY_UNKNOWN


# ---------------------------------------------------------------------------
# Main engineer class
# ---------------------------------------------------------------------------


class ProviderFacilityFeatureEngineer:
    """
    Compute provider and facility features for a single claim and upsert them
    into the Feature Store.

    Parameters
    ----------
    feature_store :
        An :class:`InMemoryFeatureStore` (or any object with a compatible
        ``upsert(claim_id, columns)`` method).  A fresh
        :class:`InMemoryFeatureStore` is created when not supplied.
    npi_registry :
        An object exposing ``lookup_specialty(npi) -> str``.  Defaults to a
        no-op :class:`NpiRegistry` that always returns ``"UNKNOWN"``.
    npi_lookup_fn :
        Optional override: a plain callable ``(npi: str) -> str`` used instead
        of *npi_registry*.  Useful for simple lambda-based injection in tests.
    """

    def __init__(
        self,
        feature_store: Optional[object] = None,
        npi_registry: Optional[NpiRegistry] = None,
        npi_lookup_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._store = feature_store if feature_store is not None else InMemoryFeatureStore()
        self._npi_registry = npi_registry if npi_registry is not None else NpiRegistry()
        self._npi_lookup_fn = npi_lookup_fn  # takes priority over npi_registry

    # ------------------------------------------------------------------
    # 4.1 / 4.2 – billing_provider_id_encoded
    # ------------------------------------------------------------------

    def compute_billing_provider_id_encoded(
        self,
        npi: str,
        window_claims: List[WindowClaim],
        reference_date: Optional[date] = None,
    ) -> float:
        """
        Leave-one-out target encoding for the billing provider NPI.

        Requirements 4.1 / 4.2
        -----------------------
        * When the provider has **>30** claims with known outcomes in the
          trailing 90-day window, compute the LOO-encoded value for the
          *current* claim: mean denial rate of all OTHER claims from the same
          provider in the window.
        * Otherwise (≤30 known-outcome claims) fall back to the
          **population-mean** denial rate across all providers that have at
          least one known-outcome claim in the same window.

        Parameters
        ----------
        npi :
            The billing provider NPI of the claim being scored.
        window_claims :
            List of historical claim dicts with keys ``npi``,
            ``submission_date``, and ``adjudication_outcome``.  Typically
            covers the full 90-day window across ALL providers (used both for
            the LOO numerator and the population-mean fallback).
        reference_date :
            The "today" anchor date for the 90-day look-back.  Defaults to
            ``date.today()``.

        Returns
        -------
        float
            Encoded value in ``[0.0, 1.0]``.
        """
        ref = reference_date or date.today()

        # Filter to window
        window_known = [
            c for c in window_claims
            if _in_trailing_window(c["submission_date"], ref) and _known_outcome(c)
        ]

        # Provider's own known-outcome claims in window
        provider_known = [c for c in window_known if c["npi"] == npi]

        if len(provider_known) > LOO_ENCODING_MIN_CLAIMS:
            # Leave-one-out: exclude the current claim from the computation.
            # "All other claims from that provider" = all provider_known minus
            # the single current record.  With only a window list (no current
            # claim object), LOO is defined as the mean of all provider_known
            # MINUS one (the most recent, as per convention).
            # Per design Property 3: result must be in [0.0, 1.0].
            other_claims = provider_known[:-1]  # drop one claim (LOO)
            if not other_claims:
                # Edge case: exactly 31 claims – after removing one we have 30.
                # Fall through to population mean.
                pass
            else:
                denied_count = sum(1 for c in other_claims if _is_denied(c))
                return denied_count / len(other_claims)

        # Population-mean fallback (Requirement 4.2)
        return self._compute_population_mean_denial_rate(window_known)

    # ------------------------------------------------------------------
    # 4.3 / 4.4 – performing_physician_specialty
    # ------------------------------------------------------------------

    def lookup_physician_specialty(self, npi: str) -> str:
        """
        Return the NUCC Health Care Provider Taxonomy code for *npi*.

        Requirements 4.3 / 4.4
        -----------------------
        * Look up the taxonomy code via the injected NPI registry (or
          callable override).
        * Return ``"UNKNOWN"`` when the NPI is absent, the registry returns
          an empty/None value, or any exception is raised.

        Parameters
        ----------
        npi :
            10-digit NPI string (synthetic; no real PHI).

        Returns
        -------
        str
            Trimmed alphanumeric NUCC taxonomy code, or ``"UNKNOWN"``.
        """
        try:
            if self._npi_lookup_fn is not None:
                code = self._npi_lookup_fn(npi)
            else:
                code = self._npi_registry.lookup_specialty(npi)

            if not code or not code.strip():
                return SENTINEL_SPECIALTY_UNKNOWN
            return code.strip()
        except Exception:  # noqa: BLE001
            logger.warning("NPI registry lookup failed for NPI %s; returning UNKNOWN.", npi)
            return SENTINEL_SPECIALTY_UNKNOWN

    # ------------------------------------------------------------------
    # 4.5 / 4.9 – provider_historical_denial_rate
    # ------------------------------------------------------------------

    def compute_provider_historical_denial_rate(
        self,
        npi: str,
        window_claims: List[WindowClaim],
        reference_date: Optional[date] = None,
    ) -> float:
        """
        Fraction of the provider's known-outcome claims that were denied,
        over the trailing 90-day window.

        Requirements 4.5 / 4.9
        -----------------------
        * Uses only claims with known adjudication outcomes (paid or denied).
        * When the provider has **<30** known-outcome claims, falls back to
          the population-mean denial rate computed across ALL providers with
          ≥1 known-outcome claim in the same window.

        Parameters
        ----------
        npi :
            Billing provider NPI.
        window_claims :
            Historical claims across all providers; must include
            ``npi``, ``submission_date``, and ``adjudication_outcome``.
        reference_date :
            Anchor date for the 90-day window.  Defaults to ``date.today()``.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.
        """
        ref = reference_date or date.today()

        window_known = [
            c for c in window_claims
            if _in_trailing_window(c["submission_date"], ref) and _known_outcome(c)
        ]

        provider_known = [c for c in window_known if c["npi"] == npi]

        if len(provider_known) >= PROVIDER_DENIAL_RATE_MIN_CLAIMS:
            denied_count = sum(1 for c in provider_known if _is_denied(c))
            return denied_count / len(provider_known)

        # Fallback: population-mean (Requirement 4.9)
        return self._compute_population_mean_denial_rate(window_known)

    # ------------------------------------------------------------------
    # 4.6 – provider_claim_volume
    # ------------------------------------------------------------------

    def compute_provider_claim_volume(
        self,
        npi: str,
        all_claims: List[WindowClaim],
        reference_date: Optional[date] = None,
    ) -> int:
        """
        Count of ALL claims (regardless of adjudication status) submitted by
        *npi* in the trailing 90-day window.

        Requirement 4.6
        ---------------
        Adjudication status is irrelevant; pending, paid, and denied claims
        all count.

        Parameters
        ----------
        npi :
            Billing provider NPI.
        all_claims :
            All historical claims (any adjudication status).
        reference_date :
            Anchor date for the 90-day window.  Defaults to ``date.today()``.

        Returns
        -------
        int
            Non-negative integer count.
        """
        ref = reference_date or date.today()
        return sum(
            1
            for c in all_claims
            if c["npi"] == npi and _in_trailing_window(c["submission_date"], ref)
        )

    # ------------------------------------------------------------------
    # 4.7 / 4.8 – facility_bed_size
    # ------------------------------------------------------------------

    def lookup_facility_bed_size(
        self,
        npi: str,
        facility_master: Dict[str, int],
    ) -> int:
        """
        Bed size for *npi* from the facility master reference table.

        Requirements 4.7 / 4.8
        -----------------------
        * Returns the bed-size value keyed by NPI in *facility_master*.
        * When the NPI is absent, returns the **integer mean** (rounded to
          nearest whole number) of all values in the table.
        * When the table is empty, returns 0.

        Parameters
        ----------
        npi :
            Billing provider NPI.
        facility_master :
            ``{npi: bed_size_int}`` reference dict loaded from the facility
            master reference table.

        Returns
        -------
        int
            Non-negative integer bed size.
        """
        if npi in facility_master:
            return facility_master[npi]

        if not facility_master:
            logger.warning(
                "Facility master is empty; cannot compute mean bed size. Returning 0."
            )
            return 0

        all_values = list(facility_master.values())
        mean_value = mean(all_values)
        return round(mean_value)

    # ------------------------------------------------------------------
    # Feature Store upsert
    # ------------------------------------------------------------------

    def engineer_features(
        self,
        claim_id: str,
        npi: str,
        window_claims: List[WindowClaim],
        all_claims: List[WindowClaim],
        facility_master: Dict[str, int],
        reference_date: Optional[date] = None,
    ) -> Dict:
        """
        Compute all provider/facility features for *claim_id* and upsert
        them into the Feature Store.

        This is the main entry point for the pipeline.  It calls each of the
        five feature methods and writes the results as a single upsert keyed
        by *claim_id*.

        Parameters
        ----------
        claim_id :
            Unique claim identifier (Feature Store row key).
        npi :
            Billing provider NPI for the claim being scored.
        window_claims :
            Historical claims with known-outcome fields (used for LOO/rate
            computations and population-mean fallback).
        all_claims :
            All historical claims including pending/unknown outcomes (used
            for volume computation).
        facility_master :
            ``{npi: bed_size}`` reference table.
        reference_date :
            Anchor date for the 90-day look-back windows.

        Returns
        -------
        dict
            The provider/facility feature columns that were upserted.
        """
        billing_provider_id_encoded = self.compute_billing_provider_id_encoded(
            npi, window_claims, reference_date
        )
        performing_physician_specialty = self.lookup_physician_specialty(npi)
        provider_historical_denial_rate = self.compute_provider_historical_denial_rate(
            npi, window_claims, reference_date
        )
        provider_claim_volume = self.compute_provider_claim_volume(
            npi, all_claims, reference_date
        )
        facility_bed_size = self.lookup_facility_bed_size(npi, facility_master)

        columns = {
            "billing_provider_id_encoded": billing_provider_id_encoded,
            "performing_physician_specialty": performing_physician_specialty,
            "provider_historical_denial_rate": provider_historical_denial_rate,
            "provider_claim_volume": provider_claim_volume,
            "facility_bed_size": facility_bed_size,
        }

        self._store.upsert(claim_id, columns)
        logger.debug("Upserted provider/facility features for claim %s: %s", claim_id, columns)
        return columns

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_population_mean_denial_rate(
        window_known: List[WindowClaim],
    ) -> float:
        """
        Population-mean denial rate across all providers/entities with ≥1
        known-outcome claim in *window_known*.

        Per Requirements 4.2 and 4.9 (and design Property 4):
        * Only entities with at least one known adjudication outcome contribute.
        * Returns 0.0 when *window_known* is empty.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.
        """
        if not window_known:
            return 0.0

        denied_count = sum(1 for c in window_known if _is_denied(c))
        return denied_count / len(window_known)
