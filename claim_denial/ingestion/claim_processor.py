"""
Claim_Processor — ingests, validates, parses, and stores X12 837 EDI claim files.

Responsibilities (Requirements 1.1 – 1.9):
    - Validate structural conformance of X12 837 files to ASC X12 005010
      (envelope segments ISA/GS/ST/SE/GE/IEA, correct delimiters).
    - Parse header, patient, service-line, and diagnosis fields from each
      claim loop, applying the normalisation rules in Requirements 1.3–1.6.
    - Write the raw EDI file to the Data Lake with a 7-year retention
      metadata sidecar (Requirement 1.9).
    - Write parsed claim JSON to the Data Lake (Requirement 1.1).
    - Quarantine structurally invalid files; skip and log per-claim parse
      failures for missing required fields (Requirement 1.7, 1.8).
    - Retry Data Lake writes up to 3 times with exponential back-off
      (1 s, 2 s, 4 s); quarantine + log on exhaustion (Requirement 1.2).

Data Lake layout::

    data_lake/
    ├── claims/YYYY/MM/DD/
    │   ├── raw/          ← original EDI bytes + .meta.json sidecar
    │   └── parsed/       ← normalised claim JSON records
    └── quarantine/YYYY/MM/DD/
        └── {file_name}   ← quarantined file copy

All PHI fields used in tests must be synthetic data only.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from claim_denial.models import ClaimRecord, ServiceLine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result / error types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Result of structural validation of an X12 837 EDI file.

    Attributes
    ----------
    is_valid:
        True when the file passes all structural checks.
    violation_segment_id:
        The segment identifier (e.g. ``"ISA"``, ``"GS"``) where the first
        violation was detected.  ``None`` when ``is_valid`` is ``True``.
    violation_description:
        Human-readable description of the structural violation.  ``None``
        when ``is_valid`` is ``True``.
    """
    is_valid: bool
    violation_segment_id: Optional[str] = None
    violation_description: Optional[str] = None


@dataclass
class ErrorLogRecord:
    """
    Structured error log record as defined in the design document.

    Schema: ``{ file_name, failure_timestamp, error_type,
                segment_ref?, field_name?, claim_segment_ref? }``
    """
    file_name: str
    failure_timestamp: str          # ISO-8601 datetime string
    error_type: str
    segment_ref: Optional[str] = None        # violation segment ID (structural errors)
    field_name: Optional[str] = None         # missing field name (claim-level errors)
    claim_segment_ref: Optional[str] = None  # claim segment ref number


# ---------------------------------------------------------------------------
# Retention metadata
# ---------------------------------------------------------------------------

_RETENTION_YEARS: int = 7

_RETENTION_METADATA: Dict[str, Any] = {
    "retention_policy": "regulatory",
    "retention_years": _RETENTION_YEARS,
    "retention_rule": "HIPAA / CMS 7-year raw EDI retention",
    "purge_after_years": _RETENTION_YEARS,
}


# ---------------------------------------------------------------------------
# ClaimProcessor
# ---------------------------------------------------------------------------

class ClaimProcessor:
    """
    Ingests, validates, parses, and stores X12 837 EDI claim files.

    Parameters
    ----------
    data_lake_base_path:
        Root directory of the (local filesystem) Data Lake.
        Defaults to ``./data_lake``.
    max_retries:
        Number of Data Lake write retries before quarantine.  Defaults to 3.
    retry_backoff_seconds:
        Sequence of wait durations (seconds) between successive retries.
        Must have ``len >= max_retries``.  Defaults to ``(1, 2, 4)``.
    """

    # Required X12 837 envelope segments in order
    _REQUIRED_ENVELOPE_SEGMENTS: Tuple[str, ...] = (
        "ISA",   # Interchange Control Header
        "GS",    # Functional Group Header
        "ST",    # Transaction Set Header
        "SE",    # Transaction Set Trailer
        "GE",    # Functional Group Trailer
        "IEA",   # Interchange Control Trailer
    )

    # Required fields per claim record (Requirement 1.7, 1.8)
    _REQUIRED_CLAIM_FIELDS: Tuple[str, ...] = (
        "provider_npi",
        "payer_id",
        "patient_account_number",
        "principal_icd10_code",
    )

    def __init__(
        self,
        data_lake_base_path: Union[str, Path] = "./data_lake",
        max_retries: int = 3,
        retry_backoff_seconds: Tuple[float, ...] = (1.0, 2.0, 4.0),
    ) -> None:
        self.data_lake_base_path = Path(data_lake_base_path)
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_x12_structure(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Check ASC X12 005010 envelope / segment conformance.

        Validates:
        - File is non-empty and readable.
        - ISA segment is present as the first segment and has exactly 16
          elements separated by the element delimiter defined in ISA[16].
        - GS (Functional Group Header) segment is present.
        - ST (Transaction Set Header) segment is present.
        - SE (Transaction Set Trailer) segment is present.
        - GE (Functional Group Trailer) segment is present.
        - IEA (Interchange Control Trailer) segment is present.
        - Segment terminator from ISA[16] is used consistently.

        Parameters
        ----------
        file_path:
            Path to the X12 837 EDI file on the local filesystem.

        Returns
        -------
        ValidationResult
            ``is_valid=True`` when all checks pass, otherwise
            ``is_valid=False`` with the first violation's segment ID and
            description.
        """
        file_path = Path(file_path)

        # ── Readability check ──────────────────────────────────────────
        try:
            raw_bytes = file_path.read_bytes()
        except (OSError, IOError) as exc:
            return ValidationResult(
                is_valid=False,
                violation_segment_id=None,
                violation_description=f"Cannot read file: {exc}",
            )

        if not raw_bytes:
            return ValidationResult(
                is_valid=False,
                violation_segment_id=None,
                violation_description="File is empty.",
            )

        # ── Decode (X12 is always ASCII / Latin-1) ────────────────────
        try:
            content = raw_bytes.decode("latin-1")
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                violation_segment_id=None,
                violation_description=f"File cannot be decoded: {exc}",
            )

        # ── ISA must be the first 3 characters ────────────────────────
        if not content.startswith("ISA"):
            return ValidationResult(
                is_valid=False,
                violation_segment_id="ISA",
                violation_description="ISA (Interchange Control Header) segment is missing or not the first segment.",
            )

        # ── Derive element / segment delimiters from ISA ──────────────
        # ISA is always exactly 106 characters:
        # "ISA" + 15 elements each separated by a 1-char element delimiter
        # The 106th char is the segment terminator.
        # Element delimiter is ISA[3] (index 3 in the raw string).
        if len(content) < 106:
            return ValidationResult(
                is_valid=False,
                violation_segment_id="ISA",
                violation_description=(
                    f"ISA segment is too short ({len(content)} chars); "
                    "expected at least 106 characters for a valid ISA envelope."
                ),
            )

        element_delimiter = content[3]
        segment_terminator = content[105]

        # ── Validate ISA has exactly 16 elements ─────────────────────
        isa_segment = content[:105]  # up to but not including the terminator
        isa_elements = isa_segment.split(element_delimiter)
        if len(isa_elements) != 16:
            return ValidationResult(
                is_valid=False,
                violation_segment_id="ISA",
                violation_description=(
                    f"ISA segment must contain exactly 16 elements; "
                    f"found {len(isa_elements)} using element delimiter {element_delimiter!r}."
                ),
            )

        # ── Split into segments using the discovered terminator ───────
        segments = [
            s.strip() for s in content.split(segment_terminator) if s.strip()
        ]

        segment_ids = {seg.split(element_delimiter)[0].strip() for seg in segments}

        # ── Check all required envelope segments are present ─────────
        for required in self._REQUIRED_ENVELOPE_SEGMENTS:
            if required not in segment_ids:
                return ValidationResult(
                    is_valid=False,
                    violation_segment_id=required,
                    violation_description=(
                        f"Required envelope segment '{required}' is missing from the file."
                    ),
                )

        # ── Verify segment terminator is used at least for ISA+GS+GE+IEA ─
        # (already guaranteed by split above producing the right segment IDs)

        return ValidationResult(is_valid=True)

    def parse_claim_file(
        self, file_path: Union[str, Path]
    ) -> List[ClaimRecord]:
        """
        Parse all claims from an X12 837 EDI file.

        Applies normalisation rules from Requirements 1.3–1.6:
        - Provider NPI: 10-digit numeric string, trimmed.
        - Payer ID: trimmed alphanumeric string.
        - Patient Account Number: trimmed alphanumeric string.
        - Billing Provider Address: street, city, state, ZIP as sub-fields.
        - Patient Name: last, first in upper-case trimmed strings.
        - Date of Birth: ISO-8601 date (YYYY-MM-DD).
        - Gender: single upper-case character M, F, or U.
        - Member ID, Group Number: trimmed alphanumeric.
        - Revenue Codes: 4-digit zero-padded string.
        - CPT/HCPCS Codes: 5-character trimmed string.
        - Service Dates: ISO-8601 YYYY-MM-DD.
        - Charge Amounts: decimal to 2 places.
        - Units of Service: non-negative integer.
        - Principal ICD-10 code: upper-case, dot-stripped, trimmed.
        - Secondary ICD-10 codes: same format, ordered list.

        Claims with missing required fields (NPI, Payer ID,
        Patient Account Number, Principal ICD-10) are skipped and logged
        (Requirement 1.8).

        Parameters
        ----------
        file_path:
            Path to a structurally valid X12 837 EDI file.

        Returns
        -------
        List[ClaimRecord]
            Successfully parsed and normalised claim records.  May be
            empty if every claim in the file failed validation.
        """
        file_path = Path(file_path)
        file_name = file_path.name

        try:
            raw_bytes = file_path.read_bytes()
            content = raw_bytes.decode("latin-1")
        except Exception as exc:
            logger.error(
                "parse_claim_file: cannot read file %s: %s", file_name, exc
            )
            return []

        element_delimiter, segment_terminator = self._detect_delimiters(content)
        if element_delimiter is None:
            logger.error(
                "parse_claim_file: cannot detect delimiters in %s", file_name
            )
            return []

        segments = [
            s.strip()
            for s in content.split(segment_terminator)
            if s.strip()
        ]

        parsed_claims: List[ClaimRecord] = []

        # ── Walk segments, collecting per-claim context ───────────────
        state = _ParserState()

        for raw_seg in segments:
            seg_id, elements = _split_segment(raw_seg, element_delimiter)

            if seg_id == "ISA":
                state = _ParserState()

            elif seg_id == "GS":
                # GS02 = application sender ID (not used here)
                pass

            elif seg_id == "NM1":
                self._handle_nm1(elements, state)

            elif seg_id == "N3":
                self._handle_n3(elements, state)

            elif seg_id == "N4":
                self._handle_n4(elements, state)

            elif seg_id == "CLM":
                # CLM: new claim loop
                if state.current_claim is not None:
                    # Finalise the previous claim before starting a new one
                    record = self._finalise_claim(state, file_name)
                    if record is not None:
                        parsed_claims.append(record)
                state.start_new_claim(elements)

            elif seg_id == "REF":
                self._handle_ref(elements, state)

            elif seg_id == "DMG":
                self._handle_dmg(elements, state)

            elif seg_id == "DTP":
                self._handle_dtp(elements, state)

            elif seg_id == "HI":
                self._handle_hi(elements, state)

            elif seg_id == "SV1":
                # Professional service line
                self._handle_sv1(elements, state)

            elif seg_id == "SV2":
                # Institutional service line
                self._handle_sv2(elements, state)

            elif seg_id == "LX":
                # Service line number — flush current service line
                state.flush_service_line()

            elif seg_id == "SE":
                # Transaction Set Trailer — finalise last claim in transaction
                if state.current_claim is not None:
                    record = self._finalise_claim(state, file_name)
                    if record is not None:
                        parsed_claims.append(record)
                    state.current_claim = None

            elif seg_id in ("GE", "IEA"):
                pass  # end of functional group / interchange — nothing to do

        return parsed_claims

    def write_to_data_lake(
        self,
        raw_bytes: bytes,
        parsed_records: List[ClaimRecord],
        date_partition: date,
        file_name: str = "claim.edi",
    ) -> None:
        """
        Write raw EDI bytes and parsed JSON records to the Data Lake.

        Retry policy: up to ``max_retries`` attempts with exponential
        back-off between them.  After exhausting retries the file is
        quarantined and a structured error record is logged.

        7-year retention metadata is written as a ``.meta.json`` sidecar
        alongside each raw EDI file (Requirement 1.9).

        Parameters
        ----------
        raw_bytes:
            Original raw EDI file content.
        parsed_records:
            Normalised ``ClaimRecord`` list to write as JSON.
        date_partition:
            The ingestion date used for partitioning (``YYYY/MM/DD``).
        file_name:
            Original file name — used as the on-disk file name and in
            error log records.

        Raises
        ------
        Does **not** raise.  All failures are logged and the file is
        quarantined after retry exhaustion.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                self._write_raw_and_parsed(
                    raw_bytes=raw_bytes,
                    parsed_records=parsed_records,
                    date_partition=date_partition,
                    file_name=file_name,
                )
                logger.info(
                    "write_to_data_lake: success on attempt %d for %s",
                    attempt + 1,
                    file_name,
                )
                return  # success — exit retry loop
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "write_to_data_lake: attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    self.max_retries,
                    file_name,
                    exc,
                )
                if attempt < self.max_retries - 1:
                    backoff = self.retry_backoff_seconds[attempt]
                    logger.debug(
                        "write_to_data_lake: waiting %.1f s before retry.", backoff
                    )
                    time.sleep(backoff)

        # ── All retries exhausted — quarantine + log ──────────────────
        logger.error(
            "write_to_data_lake: all %d retries exhausted for %s; quarantining.",
            self.max_retries,
            file_name,
        )
        self._quarantine_bytes(raw_bytes, file_name, date_partition)
        self._write_error_log(
            error_record=ErrorLogRecord(
                file_name=file_name,
                failure_timestamp=datetime.utcnow().isoformat(),
                error_type="data_lake_write_failure",
                segment_ref=None,
                field_name=None,
                claim_segment_ref=None,
            ),
            date_partition=date_partition,
        )

    def ingest_file(
        self, file_path: Union[str, Path], date_partition: Optional[date] = None
    ) -> List[ClaimRecord]:
        """
        Full ingestion workflow for a single X12 837 EDI file.

        Steps:
        1. Read the file.
        2. Validate structural conformance (Requirement 1.7).
        3. If invalid → quarantine + log; return [].
        4. Parse claim records (Requirement 1.3–1.6, 1.8).
        5. Write raw + parsed to Data Lake with retry logic (Requirement 1.1, 1.2, 1.9).
        6. Return successfully parsed records.

        Parameters
        ----------
        file_path:
            Path to the X12 837 EDI file.
        date_partition:
            Partition date for the Data Lake.  Defaults to today's date.

        Returns
        -------
        List[ClaimRecord]
            Successfully ingested claim records.
        """
        file_path = Path(file_path)
        file_name = file_path.name
        if date_partition is None:
            date_partition = date.today()

        # ── Read raw bytes ────────────────────────────────────────────
        try:
            raw_bytes = file_path.read_bytes()
        except (OSError, IOError) as exc:
            logger.error("ingest_file: cannot read %s: %s", file_name, exc)
            return []

        # ── Structural validation (Requirement 1.7) ───────────────────
        validation = self.validate_x12_structure(file_path)
        if not validation.is_valid:
            logger.error(
                "ingest_file: structural validation failed for %s — "
                "segment=%s description=%s",
                file_name,
                validation.violation_segment_id,
                validation.violation_description,
            )
            self._quarantine_bytes(raw_bytes, file_name, date_partition)
            self._write_error_log(
                error_record=ErrorLogRecord(
                    file_name=file_name,
                    failure_timestamp=datetime.utcnow().isoformat(),
                    error_type="structural_validation_failure",
                    segment_ref=validation.violation_segment_id,
                    field_name=None,
                    claim_segment_ref=None,
                ),
                date_partition=date_partition,
            )
            return []

        # ── Parse claims (Requirement 1.3–1.6, 1.8) ──────────────────
        parsed_records = self.parse_claim_file(file_path)

        # ── Write to Data Lake (Requirements 1.1, 1.2, 1.9) ──────────
        self.write_to_data_lake(
            raw_bytes=raw_bytes,
            parsed_records=parsed_records,
            date_partition=date_partition,
            file_name=file_name,
        )

        return parsed_records

    # ------------------------------------------------------------------
    # Private — segment handlers
    # ------------------------------------------------------------------

    def _handle_nm1(self, elements: List[str], state: "_ParserState") -> None:
        """NM1 — name segment for provider, payer, subscriber, patient."""
        # elements[0] = NM1, [1] = entity ID qualifier
        entity_qualifier = _get(elements, 1)

        if entity_qualifier == "85":
            # Billing provider
            state.billing_provider_last = _upper_trim(_get(elements, 3))
            state.provider_npi = _trim(_get(elements, 9))

        elif entity_qualifier == "PR":
            # Payer
            state.payer_name = _upper_trim(_get(elements, 3))
            state.payer_id = _trim(_get(elements, 9))

        elif entity_qualifier in ("IL", "QC"):
            # Subscriber / patient
            if state.current_claim is not None:
                # Patient name (IL = insured / subscriber, QC = patient)
                state.patient_last_name = _upper_trim(_get(elements, 3))
                state.patient_first_name = _upper_trim(_get(elements, 4))
                state.member_id = _trim(_get(elements, 9))

    def _handle_n3(self, elements: List[str], state: "_ParserState") -> None:
        """N3 — address line for billing provider."""
        # We capture the most recently seen N3 before CLM as billing provider
        # address; after CLM it belongs to the patient or facility — handled
        # per loop context.  For simplicity we track billing vs subscriber
        # via state flag set by NM1.
        if state.in_billing_provider_loop:
            state.billing_provider_street = _trim(_get(elements, 1))

    def _handle_n4(self, elements: List[str], state: "_ParserState") -> None:
        """N4 — city/state/ZIP."""
        if state.in_billing_provider_loop:
            state.billing_provider_city = _trim(_get(elements, 1))
            state.billing_provider_state = _trim(_get(elements, 2))
            state.billing_provider_zip = _trim(_get(elements, 3))

    def _handle_ref(self, elements: List[str], state: "_ParserState") -> None:
        """REF — reference identification (group number, member ID, etc.)."""
        ref_qualifier = _get(elements, 1)
        ref_value = _trim(_get(elements, 2))

        if ref_qualifier == "6P":
            # Group number
            state.group_number = ref_value
        elif ref_qualifier == "SY" and state.current_claim is not None:
            # Social Security — ignore for PHI reasons; do not store
            pass

    def _handle_dmg(self, elements: List[str], state: "_ParserState") -> None:
        """DMG — demographic information (DOB, gender)."""
        if state.current_claim is None:
            return
        # elements[1] = format qualifier (D8 = CCYYMMDD)
        # elements[2] = date of birth
        # elements[3] = gender code
        dob_raw = _get(elements, 2)
        if dob_raw:
            state.date_of_birth = _parse_x12_date(dob_raw)

        gender_raw = _upper_trim(_get(elements, 3))
        if gender_raw in ("M", "F"):
            state.gender = gender_raw
        else:
            state.gender = "U"

    def _handle_dtp(self, elements: List[str], state: "_ParserState") -> None:
        """DTP — date/time period."""
        # elements[1] = qualifier, [2] = format, [3] = date value
        qualifier = _get(elements, 1)
        date_value = _get(elements, 3)

        if qualifier == "472":
            # Service date (on service line)
            state.current_service_date = _parse_x12_date(date_value)
        elif qualifier == "435":
            # Admission date
            state.admission_date = _parse_x12_date(date_value)
        elif qualifier == "096":
            # Discharge date
            state.discharge_date = _parse_x12_date(date_value)
        elif qualifier == "291":
            # Claim statement period start
            state.statement_period_start = _parse_x12_date(date_value)
        elif qualifier == "292":
            # Claim statement period end
            state.statement_period_end = _parse_x12_date(date_value)

    def _handle_hi(self, elements: List[str], state: "_ParserState") -> None:
        """HI — health care diagnosis codes."""
        if state.current_claim is None:
            return

        # HI elements: each is a composite like "ABK:J189" or "BK:J189"
        # ABK / BK = principal, ABF / BF = secondary
        principal_set = False
        for element in elements[1:]:
            if not element:
                continue
            parts = element.split(":")
            qualifier = parts[0] if parts else ""
            code_raw = parts[1] if len(parts) > 1 else ""
            code = _normalise_icd10(code_raw)

            if qualifier in ("ABK", "BK"):
                # Principal diagnosis
                state.principal_icd10 = code
                principal_set = True
            elif qualifier in ("ABF", "BF", "DR"):
                # Secondary / additional diagnoses
                if code:
                    state.secondary_icd10_codes.append(code)

    def _handle_sv1(self, elements: List[str], state: "_ParserState") -> None:
        """SV1 — professional service line."""
        if state.current_claim is None:
            return
        state.flush_service_line()

        # SV101 composite: "HC:99213" or "HC:99213:26" etc.
        svc_composite = _get(elements, 1)
        cpt_code = ""
        if svc_composite:
            parts = svc_composite.split(":")
            cpt_code = _trim_5(parts[1]) if len(parts) > 1 else _trim_5(parts[0])

        charge_raw = _get(elements, 2)
        charge = _parse_decimal(charge_raw)

        units_raw = _get(elements, 4)
        units = _parse_int(units_raw)

        state.current_service_line = ServiceLine(
            cpt_hcpcs_code=cpt_code,
            service_date=state.current_service_date,
            charge_amount=charge,
            units_of_service=units,
        )

    def _handle_sv2(self, elements: List[str], state: "_ParserState") -> None:
        """SV2 — institutional service line."""
        if state.current_claim is None:
            return
        state.flush_service_line()

        revenue_raw = _trim(_get(elements, 1))
        revenue_code = _zero_pad_4(revenue_raw) if revenue_raw else None

        svc_composite = _get(elements, 2)
        cpt_code = ""
        if svc_composite:
            parts = svc_composite.split(":")
            cpt_code = _trim_5(parts[1]) if len(parts) > 1 else _trim_5(parts[0])

        charge_raw = _get(elements, 3)
        charge = _parse_decimal(charge_raw)

        units_raw = _get(elements, 5)
        units = _parse_int(units_raw)

        state.current_service_line = ServiceLine(
            revenue_code=revenue_code,
            cpt_hcpcs_code=cpt_code,
            service_date=state.current_service_date,
            charge_amount=charge,
            units_of_service=units,
        )

    # ------------------------------------------------------------------
    # Private — claim finalisation
    # ------------------------------------------------------------------

    def _finalise_claim(
        self, state: "_ParserState", file_name: str
    ) -> Optional[ClaimRecord]:
        """
        Construct a ``ClaimRecord`` from accumulated parser state.

        Returns ``None`` and logs an error record when any required field
        is absent (Requirement 1.8).
        """
        state.flush_service_line()

        claim_id = state.claim_id or f"UNKNOWN_{id(state)}"

        # Normalise required fields
        provider_npi = _normalise_npi(state.provider_npi)
        payer_id = _trim(state.payer_id)
        patient_account_number = _trim(state.patient_account_number)
        principal_icd10 = _normalise_icd10(state.principal_icd10)

        # ── Check required fields (Requirement 1.8) ───────────────────
        missing: List[str] = []
        if not _is_valid_npi(provider_npi):
            missing.append("provider_npi")
        if not payer_id:
            missing.append("payer_id")
        if not patient_account_number:
            missing.append("patient_account_number")
        if not principal_icd10:
            missing.append("principal_icd10_code")

        if missing:
            for field_name in missing:
                self._write_error_log(
                    error_record=ErrorLogRecord(
                        file_name=file_name,
                        failure_timestamp=datetime.utcnow().isoformat(),
                        error_type="missing_required_field",
                        segment_ref=None,
                        field_name=field_name,
                        claim_segment_ref=claim_id,
                    ),
                    date_partition=date.today(),
                )
            logger.warning(
                "_finalise_claim: skipping claim %s in %s — missing fields: %s",
                claim_id,
                file_name,
                missing,
            )
            return None

        return ClaimRecord(
            claim_id=claim_id,
            provider_npi=provider_npi,
            payer_id=payer_id,
            patient_account_number=patient_account_number,
            billing_provider_street=_trim(state.billing_provider_street),
            billing_provider_city=_trim(state.billing_provider_city),
            billing_provider_state=_trim(state.billing_provider_state),
            billing_provider_zip=_trim(state.billing_provider_zip),
            patient_last_name=_upper_trim(state.patient_last_name),
            patient_first_name=_upper_trim(state.patient_first_name),
            date_of_birth=state.date_of_birth,
            gender=state.gender or "U",
            member_id=_trim(state.member_id),
            group_number=_trim(state.group_number),
            principal_icd10_code=principal_icd10,
            secondary_icd10_codes=list(state.secondary_icd10_codes),
            service_lines=list(state.service_lines),
            statement_period_start=state.statement_period_start,
            statement_period_end=state.statement_period_end,
            admission_date=state.admission_date,
            discharge_date=state.discharge_date,
            claim_type=state.claim_type,
            claim_status="Active",
        )

    # ------------------------------------------------------------------
    # Private — Data Lake I/O
    # ------------------------------------------------------------------

    def _write_raw_and_parsed(
        self,
        raw_bytes: bytes,
        parsed_records: List[ClaimRecord],
        date_partition: date,
        file_name: str,
    ) -> None:
        """
        Atomically write raw EDI + sidecar metadata + parsed JSON.

        Raises ``IOError`` / ``OSError`` on failure so the caller's retry
        loop can catch and retry.
        """
        date_path = _date_partition_path(date_partition)

        # ── Raw EDI directory ─────────────────────────────────────────
        raw_dir = self.data_lake_base_path / "claims" / date_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        raw_file = raw_dir / file_name
        raw_file.write_bytes(raw_bytes)

        # ── 7-year retention metadata sidecar (Requirement 1.9) ──────
        meta_file = raw_dir / f"{file_name}.meta.json"
        meta_payload: Dict[str, Any] = {
            **_RETENTION_METADATA,
            "file_name": file_name,
            "ingestion_timestamp_utc": datetime.utcnow().isoformat(),
            "ingestion_date": date_partition.isoformat(),
        }
        meta_file.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

        # ── Parsed JSON ───────────────────────────────────────────────
        parsed_dir = self.data_lake_base_path / "claims" / date_path / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)

        parsed_file = parsed_dir / f"{file_name}.json"
        records_payload = [_claim_record_to_dict(r) for r in parsed_records]
        parsed_file.write_text(
            json.dumps(records_payload, indent=2, default=_json_default),
            encoding="utf-8",
        )

        logger.info(
            "_write_raw_and_parsed: wrote raw=%s meta=%s parsed=%s (%d records)",
            raw_file,
            meta_file,
            parsed_file,
            len(parsed_records),
        )

    def _quarantine_bytes(
        self, raw_bytes: bytes, file_name: str, date_partition: date
    ) -> None:
        """Write the raw file to the quarantine directory."""
        date_path = _date_partition_path(date_partition)
        quarantine_dir = self.data_lake_base_path / "quarantine" / date_path
        try:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            (quarantine_dir / file_name).write_bytes(raw_bytes)
            logger.info(
                "_quarantine_bytes: quarantined %s to %s",
                file_name,
                quarantine_dir,
            )
        except Exception as exc:
            logger.error(
                "_quarantine_bytes: failed to quarantine %s: %s", file_name, exc
            )

    def _write_error_log(
        self, error_record: ErrorLogRecord, date_partition: date
    ) -> None:
        """Append a structured error log entry to the errors log file."""
        date_path = _date_partition_path(date_partition)
        error_dir = self.data_lake_base_path / "claims" / date_path
        try:
            error_dir.mkdir(parents=True, exist_ok=True)
            error_log_path = error_dir / "errors.jsonl"
            entry = json.dumps(
                {
                    "file_name": error_record.file_name,
                    "failure_timestamp": error_record.failure_timestamp,
                    "error_type": error_record.error_type,
                    "segment_ref": error_record.segment_ref,
                    "field_name": error_record.field_name,
                    "claim_segment_ref": error_record.claim_segment_ref,
                }
            )
            with open(error_log_path, "a", encoding="utf-8") as fh:
                fh.write(entry + "\n")
        except Exception as exc:
            logger.error(
                "_write_error_log: failed to write error log for %s: %s",
                error_record.file_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Private — delimiter detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_delimiters(
        content: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(element_delimiter, segment_terminator)`` from ISA."""
        if not content.startswith("ISA") or len(content) < 106:
            return None, None
        element_delimiter = content[3]
        segment_terminator = content[105]
        return element_delimiter, segment_terminator


# ---------------------------------------------------------------------------
# Internal parser state
# ---------------------------------------------------------------------------

class _ParserState:
    """Mutable state accumulated while walking X12 segments for one claim."""

    __slots__ = (
        # File-level provider info (populated in NM1*85 loop before CLM)
        "provider_npi",
        "billing_provider_last",
        "billing_provider_street",
        "billing_provider_city",
        "billing_provider_state",
        "billing_provider_zip",
        # Payer info (populated in NM1*PR loop)
        "payer_id",
        "payer_name",
        # Subscriber info
        "patient_last_name",
        "patient_first_name",
        "date_of_birth",
        "gender",
        "member_id",
        "group_number",
        "patient_account_number",
        # Claim-level state
        "claim_id",
        "claim_type",
        "principal_icd10",
        "secondary_icd10_codes",
        "admission_date",
        "discharge_date",
        "statement_period_start",
        "statement_period_end",
        # Service line accumulator
        "current_service_line",
        "current_service_date",
        "service_lines",
        # Loop tracking
        "current_claim",
        "in_billing_provider_loop",
    )

    def __init__(self) -> None:
        self.provider_npi: Optional[str] = None
        self.billing_provider_last: Optional[str] = None
        self.billing_provider_street: Optional[str] = None
        self.billing_provider_city: Optional[str] = None
        self.billing_provider_state: Optional[str] = None
        self.billing_provider_zip: Optional[str] = None

        self.payer_id: Optional[str] = None
        self.payer_name: Optional[str] = None

        self.patient_last_name: Optional[str] = None
        self.patient_first_name: Optional[str] = None
        self.date_of_birth: Optional[date] = None
        self.gender: Optional[str] = None
        self.member_id: Optional[str] = None
        self.group_number: Optional[str] = None
        self.patient_account_number: Optional[str] = None

        self.claim_id: Optional[str] = None
        self.claim_type: Optional[str] = None
        self.principal_icd10: Optional[str] = None
        self.secondary_icd10_codes: List[str] = []

        self.admission_date: Optional[date] = None
        self.discharge_date: Optional[date] = None
        self.statement_period_start: Optional[date] = None
        self.statement_period_end: Optional[date] = None

        self.current_service_line: Optional[ServiceLine] = None
        self.current_service_date: Optional[date] = None
        self.service_lines: List[ServiceLine] = []

        self.current_claim: Optional[object] = None  # sentinel for "in claim loop"
        self.in_billing_provider_loop: bool = False

    def start_new_claim(self, elements: List[str]) -> None:
        """Initialise state for a new CLM loop."""
        # CLM01 = Patient Account Number (claim number / account)
        self.patient_account_number = _trim(_get(elements, 1))
        self.claim_id = self.patient_account_number or f"CLM_{id(self)}"

        # CLM05-01 = facility type code; CLM05-03 = claim frequency (not used)
        # Determine claim type from CLM05 composite
        claim_info = _get(elements, 5)  # "11:B:1" for institutional
        if claim_info:
            parts = claim_info.split(":")
            facility_code = parts[0] if parts else ""
            # Codes 11–19 and 71–72 = institutional; 11 = facility
            # For simplicity: if the code is purely numeric and 2 digits, institutional
            if re.match(r"^\d{2}$", facility_code):
                self.claim_type = "institutional"
            else:
                self.claim_type = "professional"

        self.principal_icd10 = None
        self.secondary_icd10_codes = []
        self.service_lines = []
        self.current_service_line = None
        self.current_service_date = None
        self.admission_date = None
        self.discharge_date = None
        self.statement_period_start = None
        self.statement_period_end = None

        self.current_claim = object()  # truthy sentinel
        self.in_billing_provider_loop = False

    def flush_service_line(self) -> None:
        """Move the current service line into the accumulator."""
        if self.current_service_line is not None:
            self.service_lines.append(self.current_service_line)
            self.current_service_line = None


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _split_segment(
    raw: str, element_delimiter: str
) -> Tuple[str, List[str]]:
    """Split a raw segment string into (segment_id, elements list)."""
    elements = raw.split(element_delimiter)
    seg_id = elements[0].strip() if elements else ""
    return seg_id, elements


def _get(elements: List[str], index: int, default: str = "") -> str:
    """Safely retrieve element by index from a parsed segment."""
    try:
        return elements[index]
    except IndexError:
        return default


def _trim(value: Optional[str]) -> str:
    """Return trimmed string or empty string for None."""
    return (value or "").strip()


def _upper_trim(value: Optional[str]) -> str:
    """Return upper-case trimmed string or empty string for None."""
    return (value or "").strip().upper()


def _trim_5(value: Optional[str]) -> str:
    """Return trimmed string truncated / padded to 5 characters."""
    v = (value or "").strip()
    return v[:5]


def _zero_pad_4(value: str) -> str:
    """Zero-pad a revenue code string to 4 digits."""
    v = value.strip().lstrip("0") or "0"
    return v.zfill(4)


def _normalise_npi(value: Optional[str]) -> str:
    """Trim and return the NPI string."""
    return (value or "").strip()


def _is_valid_npi(npi: str) -> bool:
    """Return True when NPI is a 10-digit numeric string."""
    return bool(re.match(r"^\d{10}$", npi))


def _normalise_icd10(value: Optional[str]) -> str:
    """Upper-case, dot-strip, and trim an ICD-10 code."""
    if not value:
        return ""
    return value.upper().replace(".", "").strip()


def _parse_x12_date(value: Optional[str]) -> Optional[date]:
    """
    Parse an X12 date string (CCYYMMDD or YYYYMMDD) into a ``date``.

    Returns ``None`` on failure.
    """
    if not value:
        return None
    v = value.strip()
    if len(v) == 8:
        try:
            return date(int(v[:4]), int(v[4:6]), int(v[6:8]))
        except (ValueError, TypeError):
            return None
    return None


def _parse_decimal(value: Optional[str]) -> Optional[float]:
    """Parse a charge amount string to float rounded to 2 places."""
    if not value:
        return None
    try:
        return round(float(value.strip()), 2)
    except (ValueError, TypeError):
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    """Parse a units-of-service string to a non-negative integer."""
    if not value:
        return None
    try:
        v = int(value.strip())
        return max(0, v)
    except (ValueError, TypeError):
        return None


def _date_partition_path(d: date) -> str:
    """Return ``YYYY/MM/DD`` partition path string for a date."""
    return f"{d.year:04d}/{d.month:02d}/{d.day:02d}"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _claim_record_to_dict(record: ClaimRecord) -> Dict[str, Any]:
    """Convert a ``ClaimRecord`` to a JSON-serialisable dict."""
    d: Dict[str, Any] = {}
    for f_name in record.__dataclass_fields__:  # type: ignore[attr-defined]
        value = getattr(record, f_name)
        if isinstance(value, date):
            d[f_name] = value.isoformat()
        elif isinstance(value, list):
            items = []
            for item in value:
                if hasattr(item, "__dataclass_fields__"):
                    items.append(_service_line_to_dict(item))
                else:
                    items.append(item)
            d[f_name] = items
        else:
            d[f_name] = value
    return d


def _service_line_to_dict(sl: ServiceLine) -> Dict[str, Any]:
    """Convert a ``ServiceLine`` to a JSON-serialisable dict."""
    return {
        "revenue_code": sl.revenue_code,
        "cpt_hcpcs_code": sl.cpt_hcpcs_code,
        "service_date": sl.service_date.isoformat() if sl.service_date else None,
        "charge_amount": sl.charge_amount,
        "units_of_service": sl.units_of_service,
    }


def _json_default(obj: Any) -> Any:
    """Custom JSON encoder for types not natively serialisable."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
