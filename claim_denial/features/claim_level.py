"""
Claim-level feature engineering for the Claim Denial Prediction system.

Computes all claim-level features as defined in Requirements 3.1–3.10:
  - Temporal features (day-of-year, month)
  - Patient demographics (age, gender)
  - One-hot encodings (insurance_type, claim_type, admission_type, admission_source)
  - Writes a denormalized row to the Feature Store keyed by Claim ID (upsert)
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from claim_denial.constants import (
    ADMISSION_SOURCES,
    ADMISSION_TYPES,
    CLAIM_TYPES,
    INSURANCE_TYPES,
    SENTINEL_AGE_UNKNOWN,
    SENTINEL_GENDER_UNKNOWN,
)
from claim_denial.models import ClaimFeatureRecord, ClaimRecord

logger = logging.getLogger(__name__)

# Current schema version (mirrors Avro schema in claim_denial/schemas/)
SCHEMA_VERSION = "1.0.0"


class ClaimLevelFeatureEngineer:
    """
    Computes claim-level features from a :class:`ClaimRecord` and writes the
    resulting denormalized row to a Feature Store.

    Parameters
    ----------
    feature_store:
        A dict-like object used as the Feature Store backend.  Keys are
        ``claim_id`` strings; values are plain dicts of feature columns.
        If *None*, a fresh in-memory dict is created (useful for unit tests).
        Pass a :class:`DiskBackedFeatureStore` instance to persist to JSON on
        the local filesystem.
    """

    def __init__(self, feature_store: Optional[Dict[str, Any]] = None) -> None:
        self._store: Dict[str, Any] = feature_store if feature_store is not None else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_and_store(self, claim: ClaimRecord) -> ClaimFeatureRecord:
        """Compute all claim-level features and upsert them to the Feature Store.

        Parameters
        ----------
        claim:
            A parsed and normalised :class:`ClaimRecord`.

        Returns
        -------
        ClaimFeatureRecord
            The (potentially partial) feature record containing only the
            claim-level columns populated by this engineer.  Other feature
            groups remain at their default values until filled by subsequent
            engineers.
        """
        record = self._build_feature_record(claim)
        self._upsert(claim.claim_id, record)
        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feature_record(self, claim: ClaimRecord) -> ClaimFeatureRecord:
        """Compute all claim-level features and return a :class:`ClaimFeatureRecord`."""
        submission_date = claim.statement_period_start

        # ── Temporal features (Requirement 3.1) ──────────────────────────
        doy, month = self._compute_date_features(submission_date)

        # ── Patient age (Requirements 3.2, 3.9) ──────────────────────────
        age = self._compute_patient_age(claim.date_of_birth, submission_date)

        # ── Gender encoding (Requirements 3.3, 3.4) ──────────────────────
        gender = self._encode_gender(claim.gender)

        # ── Insurance type one-hot (Requirement 3.5) ─────────────────────
        ins = self._one_hot_insurance_type(claim.insurance_type)

        # ── Claim type one-hot (Requirement 3.6) ─────────────────────────
        ct = self._one_hot_claim_type(claim.claim_type)

        # ── Admission type one-hot (Requirement 3.7) ─────────────────────
        at = self._one_hot_admission_type(claim.admission_type)

        # ── Admission source one-hot (Requirement 3.8) ───────────────────
        as_ = self._one_hot_admission_source(claim.admission_source)

        return ClaimFeatureRecord(
            schema_version=SCHEMA_VERSION,
            claim_id=claim.claim_id,
            claim_status=claim.claim_status,
            # Temporal
            claim_submission_date_doy=doy,
            claim_submission_date_month=month,
            # Demographics
            patient_age=age,
            gender=gender,
            # Insurance type
            ins_type_medicare=ins["Medicare"],
            ins_type_medicaid=ins["Medicaid"],
            ins_type_commercial=ins["Commercial"],
            ins_type_tricare=ins["TriCare"],
            ins_type_champva=ins["ChampVA"],
            ins_type_other=ins["Other"],
            # Claim type
            claim_type_professional=ct["Professional"],
            claim_type_institutional=ct["Institutional"],
            # Admission type
            adm_type_emergency=at["Emergency"],
            adm_type_elective=at["Elective"],
            adm_type_urgent=at["Urgent"],
            adm_type_trauma=at["Trauma"],
            # Admission source
            adm_src_physician_referral=as_["Physician Referral"],
            adm_src_transfer_hospital=as_["Transfer from Hospital"],
            adm_src_transfer_snf=as_["Transfer from SNF"],
            adm_src_er=as_["Emergency Room"],
            adm_src_court_law=as_["Court/Law Enforcement"],
            adm_src_not_available=as_["Not Available"],
            # EHR linkage flags (pass-through from claim)
            partial_ehr_flag=claim.partial_ehr_flag,
            ehr_null_match_flag=claim.ehr_null_match_flag,
        )

    # ── Individual feature computations ──────────────────────────────────

    @staticmethod
    def _compute_date_features(submission_date: Optional[date]) -> tuple[int, int]:
        """Return (day_of_year, month) for *submission_date*.

        If the date is absent the values default to 0 (sentinel for "unknown").
        """
        if submission_date is None:
            return 0, 0
        return submission_date.timetuple().tm_yday, submission_date.month

    @staticmethod
    def _compute_patient_age(
        dob: Optional[date],
        reference_date: Optional[date],
    ) -> int:
        """Compute whole-year age from *dob* to *reference_date*.

        Returns ``SENTINEL_AGE_UNKNOWN`` (−1) when either date is absent.
        """
        if dob is None or reference_date is None:
            return SENTINEL_AGE_UNKNOWN

        age = reference_date.year - dob.year
        # Subtract 1 if the birthday has not yet occurred this year
        if (reference_date.month, reference_date.day) < (dob.month, dob.day):
            age -= 1
        return age

    @staticmethod
    def _encode_gender(gender: Optional[str]) -> int:
        """Encode gender as 0 (F), 1 (M), or −1 (unknown/absent).

        Matching is case-insensitive.  Any value that is not ``"F"`` or ``"M"``
        (including ``None``, ``"U"``, empty string) returns −1.
        """
        if gender is None:
            return SENTINEL_GENDER_UNKNOWN
        g = gender.strip().upper()
        if g == "F":
            return 0
        if g == "M":
            return 1
        return SENTINEL_GENDER_UNKNOWN

    @staticmethod
    def _one_hot_insurance_type(insurance_type: Optional[str]) -> Dict[str, int]:
        """One-hot encode *insurance_type* over :data:`INSURANCE_TYPES`.

        Case-insensitive match.  Unrecognised or absent values → all zeros.
        """
        result = {cat: 0 for cat in INSURANCE_TYPES}
        if insurance_type is None:
            return result
        normalized = insurance_type.strip()
        for cat in INSURANCE_TYPES:
            if normalized.lower() == cat.lower():
                result[cat] = 1
                return result
        return result  # unrecognised → all zeros

    @staticmethod
    def _one_hot_claim_type(claim_type: Optional[str]) -> Dict[str, int]:
        """One-hot encode *claim_type* into Professional / Institutional columns.

        Case-insensitive match.  Absent or unrecognised → all zeros.
        """
        result = {cat: 0 for cat in CLAIM_TYPES}
        if claim_type is None:
            return result
        normalized = claim_type.strip()
        for cat in CLAIM_TYPES:
            if normalized.lower() == cat.lower():
                result[cat] = 1
                return result
        return result

    @staticmethod
    def _one_hot_admission_type(admission_type: Optional[str]) -> Dict[str, int]:
        """One-hot encode *admission_type* over :data:`ADMISSION_TYPES`.

        Absent or unrecognised values → all zeros (Requirement 3.7).
        """
        result = {cat: 0 for cat in ADMISSION_TYPES}
        if admission_type is None:
            return result
        normalized = admission_type.strip()
        for cat in ADMISSION_TYPES:
            if normalized.lower() == cat.lower():
                result[cat] = 1
                return result
        return result

    @staticmethod
    def _one_hot_admission_source(admission_source: Optional[str]) -> Dict[str, int]:
        """One-hot encode *admission_source* over :data:`ADMISSION_SOURCES`.

        Unrecognised values → all zeros (Requirement 3.8).
        """
        result = {cat: 0 for cat in ADMISSION_SOURCES}
        if admission_source is None:
            return result
        normalized = admission_source.strip()
        for cat in ADMISSION_SOURCES:
            if normalized.lower() == cat.lower():
                result[cat] = 1
                return result
        return result

    # ── Feature Store upsert (Requirement 3.10) ──────────────────────────

    def _upsert(self, claim_id: str, record: ClaimFeatureRecord) -> None:
        """Write the claim-level feature columns to the Feature Store.

        The upsert is keyed by *claim_id*.  If a row already exists it is
        merged, with the new claim-level values overwriting any prior values
        for the same keys.
        """
        # Serialise only the claim-level fields (all scalar fields from the
        # record's __dataclass_fields__, excluding expensive NLP arrays and
        # scoring outputs that are populated by other engineers).
        row: Dict[str, Any] = {
            "schema_version": record.schema_version,
            "claim_id": record.claim_id,
            "claim_status": record.claim_status,
            "claim_submission_date_doy": record.claim_submission_date_doy,
            "claim_submission_date_month": record.claim_submission_date_month,
            "patient_age": record.patient_age,
            "gender": record.gender,
            "ins_type_medicare": record.ins_type_medicare,
            "ins_type_medicaid": record.ins_type_medicaid,
            "ins_type_commercial": record.ins_type_commercial,
            "ins_type_tricare": record.ins_type_tricare,
            "ins_type_champva": record.ins_type_champva,
            "ins_type_other": record.ins_type_other,
            "claim_type_professional": record.claim_type_professional,
            "claim_type_institutional": record.claim_type_institutional,
            "adm_type_emergency": record.adm_type_emergency,
            "adm_type_elective": record.adm_type_elective,
            "adm_type_urgent": record.adm_type_urgent,
            "adm_type_trauma": record.adm_type_trauma,
            "adm_src_physician_referral": record.adm_src_physician_referral,
            "adm_src_transfer_hospital": record.adm_src_transfer_hospital,
            "adm_src_transfer_snf": record.adm_src_transfer_snf,
            "adm_src_er": record.adm_src_er,
            "adm_src_court_law": record.adm_src_court_law,
            "adm_src_not_available": record.adm_src_not_available,
            "partial_ehr_flag": record.partial_ehr_flag,
            "ehr_null_match_flag": record.ehr_null_match_flag,
        }

        if claim_id in self._store:
            # Merge: preserve existing keys not set by this engineer
            self._store[claim_id].update(row)
        else:
            self._store[claim_id] = row

        logger.debug("Feature Store upsert: claim_id=%s", claim_id)


# ---------------------------------------------------------------------------
# Optional disk-backed Feature Store mock
# ---------------------------------------------------------------------------

class DiskBackedFeatureStore(dict):
    """A simple JSON-file-backed dict that can be passed as *feature_store*.

    On every write the entire store is flushed to *file_path*.  This is not
    efficient for large volumes but is sufficient for local development and
    integration tests.

    Parameters
    ----------
    file_path:
        Path to the JSON file.  The file is created if it does not exist; if
        it already exists its contents are loaded into memory on construction.
    """

    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self._path = Path(file_path)
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.update(data)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._flush()

    def _flush(self) -> None:
        """Persist the entire store to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(dict(self), fh, indent=2, default=str)
