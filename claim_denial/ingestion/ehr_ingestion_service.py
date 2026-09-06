"""
EHR_Ingestion_Service — retrieves and links EHR encounter data to claims.

Responsibilities (Requirements 2.1 – 2.7):
    - Fetch EHR encounters for a Patient Account Number via an injectable
      EHRClient interface (FHIR R4 / HL7 v2 in production; stub in tests).
    - Select the encounter whose date most closely precedes the claim
      statement period start date; log the selection when multiple records
      exist.
    - Extract all structured fields (Requirement 2.3) from the matched
      encounter; set absent fields to null and populate ``partial_ehr_flag``.
    - Extract clinical notes (H&P, Discharge Summary, Progress Notes)
      scoped to the matched encounter.
    - Set ``ehr_null_match_flag`` when no EHR record is found, allowing
      downstream processing with claim-only features.
    - Write the extracted EHR JSON to the Data Lake partitioned by
      ``{patient_account_number}/{encounter_date}/ehr_record.json``.

All PHI fields in tests must use synthetic data only.
"""

from __future__ import annotations

import abc
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from claim_denial.models import (
    ClinicalNote,
    EHRRecord,
    LOINCLabResult,
    Medication,
    NoteType,
    SNOMEDProblem,
    VitalSigns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EHREncounter — lightweight intermediate representation
# ---------------------------------------------------------------------------

@dataclass
class EHREncounter:
    """
    Intermediate representation of a single EHR encounter returned by an
    EHRClient.

    This dataclass is designed to be hydrated from either a FHIR R4 Bundle
    or an HL7 v2 ADT message.  All nested clinical sub-records (vitals,
    labs, problems, medications, notes) are stored as plain Python
    dicts/lists so that the EHRClient implementations do not need to
    import the full domain model hierarchy.

    The EHRIngestionService converts an EHREncounter to the canonical
    EHRRecord domain model during structured-field extraction.
    """

    # Core identifiers
    patient_account_number: str
    encounter_id: str
    encounter_date: date  # The visit / admission date used for selection

    # Patient demographics (Requirement 2.3)
    age: Optional[int] = None
    sex: Optional[str] = None           # "M", "F", or "U"
    race: Optional[str] = None
    ethnicity: Optional[str] = None

    # Vitals (dict matching VitalSigns field names)
    vitals: Optional[Dict[str, Any]] = None

    # LOINC labs (list of dicts matching LOINCLabResult field names)
    lab_results: Optional[List[Dict[str, Any]]] = None

    # SNOMED / ICD-10 problem list (list of dicts)
    problem_list: Optional[List[Dict[str, Any]]] = None

    # Active medication list (list of dicts)
    medications: Optional[List[Dict[str, Any]]] = None

    # Administrative
    admitting_department: Optional[str] = None
    admitting_physician_id: Optional[str] = None
    length_of_stay_days: Optional[int] = None

    # Clinical notes scoped to this encounter
    clinical_notes: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# EHRClient — abstract interface (injectable / stub-friendly)
# ---------------------------------------------------------------------------

class EHRClient(abc.ABC):
    """
    Abstract EHR client interface.

    Concrete implementations integrate with:
      - FHIR R4 RESTful APIs (``FHIRClient``)
      - HL7 v2 message sources
      - Stub / in-memory implementations for unit tests

    All implementors must be constructable without side-effects (no network
    calls in ``__init__``) so that test stubs can be instantiated freely.
    """

    @abc.abstractmethod
    def get_encounters(self, patient_account_number: str) -> List[EHREncounter]:
        """
        Return all EHR encounters for the given Patient Account Number.

        Parameters
        ----------
        patient_account_number:
            The trimmed alphanumeric patient account number from the claim.

        Returns
        -------
        List[EHREncounter]
            Zero or more encounter records.  An empty list means no EHR
            record was found (triggers ``ehr_null_match_flag``).

        Raises
        ------
        EHRClientError
            On unrecoverable communication or parsing failures.
        """


class EHRClientError(Exception):
    """Raised when an EHRClient implementation fails to retrieve records."""


# ---------------------------------------------------------------------------
# FHIRClient — concrete FHIR R4 implementation (stub-friendly)
# ---------------------------------------------------------------------------

class FHIRClient(EHRClient):
    """
    Concrete FHIR R4 client.

    In a real deployment this class would call the FHIR endpoint at
    ``base_url``.  In unit tests, pass ``base_url=None`` and subclass or
    monkey-patch ``_http_get`` to return fixture data without network I/O.

    Parameters
    ----------
    base_url:
        FHIR server base URL, e.g. ``https://ehr.example.com/fhir/r4``.
        May be ``None`` in test / offline mode.
    auth_token:
        Bearer token for FHIR server authentication.  May be ``None``.
    timeout_seconds:
        HTTP request timeout.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Internal HTTP helper — overridable in tests
    # ------------------------------------------------------------------

    def _http_get(self, url: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform a GET request against the FHIR server and return the
        parsed JSON response body.

        Override this method in test stubs to inject fixture data.

        Raises
        ------
        EHRClientError
            On HTTP errors, timeouts, or JSON parse failures.
        """
        try:
            import urllib.request
            import urllib.parse

            headers = {"Accept": "application/fhir+json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            full_url = url
            if params:
                full_url = f"{url}?{urllib.parse.urlencode(params)}"

            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise EHRClientError(f"FHIR GET failed for {url}: {exc}") from exc

    def get_encounters(self, patient_account_number: str) -> List[EHREncounter]:
        """
        Query FHIR R4 for all Encounter resources linked to the patient.

        The patient is located by ``account`` identifier.  Each FHIR
        Encounter Bundle entry is parsed into an ``EHREncounter``.

        If ``base_url`` is None (offline / test mode), returns an empty
        list so that the null-match code path is exercised.
        """
        if not self.base_url:
            logger.warning(
                "FHIRClient.base_url is None — returning empty encounter list "
                "(offline / test mode). patient_account_number=%s",
                patient_account_number,
            )
            return []

        try:
            bundle = self._http_get(
                f"{self.base_url}/Encounter",
                params={
                    "patient.identifier": patient_account_number,
                    "_include": "Encounter:subject",
                    "_count": "50",
                },
            )
        except EHRClientError:
            logger.exception(
                "FHIR encounter lookup failed for patient_account_number=%s",
                patient_account_number,
            )
            return []

        encounters: List[EHREncounter] = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Encounter":
                continue
            enc = self._parse_fhir_encounter(resource, patient_account_number)
            if enc is not None:
                encounters.append(enc)

        return encounters

    def _parse_fhir_encounter(
        self, resource: Dict[str, Any], patient_account_number: str
    ) -> Optional[EHREncounter]:
        """
        Parse a single FHIR Encounter resource into an ``EHREncounter``.

        Returns ``None`` if the resource lacks the minimum required fields
        (encounter ID and period start date).
        """
        encounter_id = resource.get("id")
        period = resource.get("period", {})
        start_str = period.get("start")

        if not encounter_id or not start_str:
            logger.warning(
                "Skipping FHIR Encounter with missing id or period.start: %s",
                resource,
            )
            return None

        try:
            # FHIR dates may include time component; take date portion only
            encounter_date = date.fromisoformat(start_str[:10])
        except ValueError:
            logger.warning(
                "Cannot parse FHIR Encounter period.start=%r; skipping.", start_str
            )
            return None

        return EHREncounter(
            patient_account_number=patient_account_number,
            encounter_id=encounter_id,
            encounter_date=encounter_date,
            # Demographics and clinical data require additional FHIR resource
            # lookups (Patient, Observation, etc.) — left as None here and
            # populated by a richer implementation.  In tests, inject a
            # fully-populated EHREncounter via a stub client.
        )


# ---------------------------------------------------------------------------
# InMemoryEHRClient — stub implementation for tests
# ---------------------------------------------------------------------------

class InMemoryEHRClient(EHRClient):
    """
    In-memory EHR client that returns pre-loaded encounters.

    Intended for unit and integration tests.  Callers register encounters
    by Patient Account Number via ``register_encounters``; the client
    returns them from ``get_encounters`` without any I/O.

    Example
    -------
    >>> client = InMemoryEHRClient()
    >>> client.register_encounters("ACC001", [enc1, enc2])
    >>> assert client.get_encounters("ACC001") == [enc1, enc2]
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[EHREncounter]] = {}

    def register_encounters(
        self, patient_account_number: str, encounters: List[EHREncounter]
    ) -> None:
        """Register a list of encounters for a Patient Account Number."""
        self._store[patient_account_number] = list(encounters)

    def get_encounters(self, patient_account_number: str) -> List[EHREncounter]:
        """Return registered encounters, or an empty list if none registered."""
        return list(self._store.get(patient_account_number, []))


# ---------------------------------------------------------------------------
# EHRIngestionService
# ---------------------------------------------------------------------------

class EHRIngestionService:
    """
    Orchestrates EHR data retrieval and extraction for a single claim.

    Parameters
    ----------
    ehr_client:
        An ``EHRClient`` implementation.  Pass an ``InMemoryEHRClient``
        in tests, a ``FHIRClient`` in production.
    data_lake_base_path:
        Root directory of the (simulated) Data Lake.  EHR records are
        written to::

            {data_lake_base_path}/ehr/{patient_account_number}/{encounter_date}/ehr_record.json

        Defaults to ``./data_lake``.
    """

    # Structured field names extracted from an encounter (Requirement 2.3).
    # Used to determine which fields are absent when populating
    # ``partial_ehr_flag``.
    STRUCTURED_FIELDS: List[str] = [
        "age",
        "sex",
        "race",
        "ethnicity",
        "vitals",
        "lab_results",
        "problem_list",
        "medications",
        "admitting_department",
        "admitting_physician_id",
        "length_of_stay_days",
    ]

    def __init__(
        self,
        ehr_client: EHRClient,
        data_lake_base_path: Union[str, Path] = "./data_lake",
    ) -> None:
        self.ehr_client = ehr_client
        self.data_lake_base_path = Path(data_lake_base_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ehr_records(
        self, patient_account_number: str
    ) -> List[EHREncounter]:
        """
        Retrieve all EHR encounters for the given Patient Account Number.

        Delegates to the injected ``EHRClient``.  Returns an empty list
        when no records exist; the caller should then set
        ``ehr_null_match_flag``.

        Parameters
        ----------
        patient_account_number:
            Trimmed alphanumeric patient account number from the claim.

        Returns
        -------
        List[EHREncounter]
            Zero or more encounter records sorted by encounter_date
            ascending.
        """
        encounters = self.ehr_client.get_encounters(patient_account_number)
        # Stable sort by encounter date ascending
        encounters.sort(key=lambda e: e.encounter_date)
        return encounters

    def select_encounter(
        self,
        encounters: List[EHREncounter],
        claim_statement_start_date: date,
        claim_id: Optional[str] = None,
    ) -> Optional[EHREncounter]:
        """
        Select the encounter whose date most closely precedes
        ``claim_statement_start_date`` (Requirement 2.2).

        Selection rule:
        - Only encounters whose ``encounter_date < claim_statement_start_date``
          are eligible.
        - Among eligible encounters, select the one with the largest
          (most recent) ``encounter_date``.
        - If no encounter precedes the start date, return ``None``.

        When multiple encounters exist (regardless of whether they are
        eligible), the selection is logged at INFO level with the encounter
        date and claim ID.

        Parameters
        ----------
        encounters:
            List of EHREncounter objects for the patient.
        claim_statement_start_date:
            The claim statement period start date (from ClaimRecord).
        claim_id:
            Optional claim identifier used in log messages.

        Returns
        -------
        Optional[EHREncounter]
            The best-matching encounter, or ``None`` when no eligible
            encounter exists.
        """
        if not encounters:
            return None

        eligible = [
            e for e in encounters if e.encounter_date < claim_statement_start_date
        ]

        if not eligible:
            logger.info(
                "No eligible EHR encounters precede claim start date %s "
                "(claim_id=%s, total_encounters=%d)",
                claim_statement_start_date,
                claim_id or "unknown",
                len(encounters),
            )
            return None

        selected = max(eligible, key=lambda e: e.encounter_date)

        if len(encounters) > 1:
            logger.info(
                "Selected EHR encounter: encounter_id=%s encounter_date=%s "
                "claim_id=%s (from %d total encounters, %d eligible)",
                selected.encounter_id,
                selected.encounter_date,
                claim_id or "unknown",
                len(encounters),
                len(eligible),
            )

        return selected

    def extract_structured_fields(
        self, encounter: EHREncounter
    ) -> Dict[str, Any]:
        """
        Extract all structured EHR fields from the encounter (Requirement 2.3).

        Absent fields are set to ``None``.  The ``partial_ehr_flag`` key
        in the returned dict is a comma-separated string of the names of
        absent fields, or ``None`` when all fields are present
        (Requirement 2.5).

        Parameters
        ----------
        encounter:
            A matched ``EHREncounter`` object.

        Returns
        -------
        Dict[str, Any]
            Structured fields dict with the following keys:
            ``patient_account_number``, ``encounter_id``,
            ``encounter_date``, ``age``, ``sex``, ``race``, ``ethnicity``,
            ``vitals``, ``lab_results``, ``problem_list``, ``medications``,
            ``admitting_department``, ``admitting_physician_id``,
            ``length_of_stay_days``, ``partial_ehr_flag``.
        """
        absent_fields: List[str] = []

        def _get(field_name: str) -> Any:
            value = getattr(encounter, field_name, None)
            if value is None:
                absent_fields.append(field_name)
            return value

        structured: Dict[str, Any] = {
            "patient_account_number": encounter.patient_account_number,
            "encounter_id": encounter.encounter_id,
            "encounter_date": encounter.encounter_date.isoformat()
            if isinstance(encounter.encounter_date, date)
            else encounter.encounter_date,
        }

        for field_name in self.STRUCTURED_FIELDS:
            structured[field_name] = _get(field_name)

        structured["partial_ehr_flag"] = (
            ",".join(absent_fields) if absent_fields else None
        )

        return structured

    def extract_clinical_notes(
        self, encounter: EHREncounter
    ) -> List[ClinicalNote]:
        """
        Extract H&P, Discharge Summary, and Progress Notes scoped to the
        matched encounter (Requirement 2.4).

        Notes stored in ``encounter.clinical_notes`` (as dicts) are
        converted to ``ClinicalNote`` domain objects.  Only notes whose
        ``note_type`` is one of the three required types are included.
        Notes that are already ``ClinicalNote`` instances are returned
        as-is.

        Parameters
        ----------
        encounter:
            A matched ``EHREncounter`` object.

        Returns
        -------
        List[ClinicalNote]
            Chronologically ordered list (by ``note_datetime``) of
            clinical notes scoped to the encounter.
        """
        if not encounter.clinical_notes:
            return []

        accepted_types = {
            NoteType.HISTORY_AND_PHYSICAL,
            NoteType.DISCHARGE_SUMMARY,
            NoteType.PROGRESS_NOTE,
        }

        notes: List[ClinicalNote] = []
        for raw in encounter.clinical_notes:
            # Support both ClinicalNote instances and plain dicts
            if isinstance(raw, ClinicalNote):
                note = raw
            else:
                note = self._dict_to_clinical_note(raw, encounter)

            if note is None:
                continue

            if note.note_type in accepted_types:
                notes.append(note)

        notes.sort(key=lambda n: n.note_datetime)
        return notes

    def process_claim(
        self,
        patient_account_number: str,
        claim_statement_start_date: date,
        claim_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full EHR ingestion workflow for a single claim (Requirements 2.1 – 2.7).

        Steps:
        1. Fetch all EHR encounters for the patient.
        2. Select the best-matching encounter.
        3. If no encounter found, set ``ehr_null_match_flag`` and return
           early (claim-only features allowed downstream).
        4. Extract structured fields and clinical notes.
        5. Write EHR JSON to Data Lake.

        Parameters
        ----------
        patient_account_number:
            Patient account number from the claim record.
        claim_statement_start_date:
            Claim statement period start date.
        claim_id:
            Optional claim identifier for logging.

        Returns
        -------
        Dict[str, Any]
            A dict with the following keys:

            - ``ehr_null_match_flag`` (int): 1 if no EHR was matched, 0 otherwise.
            - ``partial_ehr_flag`` (Optional[str]): comma-separated absent fields, or None.
            - ``structured_fields`` (Optional[Dict]): extracted structured EHR data.
            - ``clinical_notes`` (Optional[List[ClinicalNote]]): extracted notes.
            - ``data_lake_path`` (Optional[str]): path where EHR JSON was written.
        """
        encounters = self.fetch_ehr_records(patient_account_number)
        selected = self.select_encounter(
            encounters, claim_statement_start_date, claim_id=claim_id
        )

        # ── No EHR record found (Requirement 2.6) ──────────────────────
        if selected is None:
            logger.info(
                "No EHR record found for patient_account_number=%s claim_id=%s. "
                "Setting ehr_null_match_flag=1.",
                patient_account_number,
                claim_id or "unknown",
            )
            return {
                "ehr_null_match_flag": 1,
                "partial_ehr_flag": None,
                "structured_fields": None,
                "clinical_notes": None,
                "data_lake_path": None,
            }

        # ── Extract structured fields (Requirements 2.3, 2.5) ──────────
        structured_fields = self.extract_structured_fields(selected)

        # ── Extract clinical notes (Requirement 2.4) ───────────────────
        clinical_notes = self.extract_clinical_notes(selected)

        # ── Write to Data Lake (Requirement 2.7) ───────────────────────
        data_lake_path = self._write_to_data_lake(
            structured_fields=structured_fields,
            clinical_notes=clinical_notes,
            patient_account_number=patient_account_number,
            encounter_date=selected.encounter_date,
        )

        partial_ehr_flag = structured_fields.get("partial_ehr_flag")

        return {
            "ehr_null_match_flag": 0,
            "partial_ehr_flag": partial_ehr_flag,
            "structured_fields": structured_fields,
            "clinical_notes": clinical_notes,
            "data_lake_path": data_lake_path,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_to_data_lake(
        self,
        structured_fields: Dict[str, Any],
        clinical_notes: List[ClinicalNote],
        patient_account_number: str,
        encounter_date: date,
    ) -> str:
        """
        Serialise the EHR record to JSON and write it to the Data Lake.

        Partition layout (Requirement 2.7)::

            {data_lake_base_path}/ehr/{patient_account_number}/{encounter_date}/ehr_record.json

        Parameters
        ----------
        structured_fields:
            Dict returned by ``extract_structured_fields``.
        clinical_notes:
            List of ``ClinicalNote`` objects.
        patient_account_number:
            Used as the first partition key.
        encounter_date:
            Used as the second partition key (ISO-8601 string).

        Returns
        -------
        str
            Absolute path of the written file.
        """
        enc_date_str = (
            encounter_date.isoformat()
            if isinstance(encounter_date, date)
            else str(encounter_date)
        )

        partition_dir = (
            self.data_lake_base_path
            / "ehr"
            / patient_account_number
            / enc_date_str
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / "ehr_record.json"

        # Serialise notes to plain dicts
        serialised_notes = [
            self._clinical_note_to_dict(note) for note in clinical_notes
        ]

        payload: Dict[str, Any] = {
            **structured_fields,
            "clinical_notes": serialised_notes,
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)

        logger.info(
            "EHR record written to Data Lake: %s (patient_account_number=%s)",
            output_path,
            patient_account_number,
        )

        return str(output_path)

    @staticmethod
    def _clinical_note_to_dict(note: ClinicalNote) -> Dict[str, Any]:
        """Convert a ``ClinicalNote`` instance to a JSON-serialisable dict."""
        return {
            "note_type": note.note_type.value,
            "note_date": note.note_date.isoformat(),
            "note_datetime": note.note_datetime.isoformat(),
            "author_id": note.author_id,
            "patient_account_number": note.patient_account_number,
            "encounter_id": note.encounter_id,
            "text": note.text,
        }

    @staticmethod
    def _dict_to_clinical_note(
        raw: Dict[str, Any], encounter: EHREncounter
    ) -> Optional[ClinicalNote]:
        """
        Convert a plain dict (from ``encounter.clinical_notes``) to a
        ``ClinicalNote`` instance.

        Returns ``None`` when the dict is missing required fields or
        contains an unrecognised note type.
        """
        try:
            note_type_val = raw.get("note_type", "")
            # Accept both enum value ("H&P") and enum name ("HISTORY_AND_PHYSICAL")
            try:
                note_type = NoteType(note_type_val)
            except ValueError:
                # Try matching by enum name
                note_type_by_name = {
                    member.name: member for member in NoteType
                }
                if note_type_val in note_type_by_name:
                    note_type = note_type_by_name[note_type_val]
                else:
                    logger.debug(
                        "Unknown note_type=%r in encounter %s; skipping.",
                        note_type_val,
                        encounter.encounter_id,
                    )
                    return None

            note_date_raw = raw.get("note_date")
            if isinstance(note_date_raw, date):
                note_date = note_date_raw
            elif isinstance(note_date_raw, str):
                note_date = date.fromisoformat(note_date_raw[:10])
            else:
                note_date = encounter.encounter_date

            note_datetime_raw = raw.get("note_datetime")
            if isinstance(note_datetime_raw, datetime):
                note_datetime = note_datetime_raw
            elif isinstance(note_datetime_raw, str):
                note_datetime = datetime.fromisoformat(note_datetime_raw)
            else:
                note_datetime = datetime(
                    note_date.year, note_date.month, note_date.day, 0, 0, 0
                )

            return ClinicalNote(
                note_type=note_type,
                note_date=note_date,
                note_datetime=note_datetime,
                author_id=raw.get("author_id", "UNKNOWN"),
                patient_account_number=raw.get(
                    "patient_account_number",
                    encounter.patient_account_number,
                ),
                encounter_id=raw.get("encounter_id", encounter.encounter_id),
                text=raw.get("text", ""),
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse clinical note dict in encounter %s: %s",
                encounter.encounter_id,
                exc,
            )
            return None


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Custom JSON encoder for types not natively serialisable."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
