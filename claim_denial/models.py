"""
Shared domain models for the Claim Denial Prediction system.

All patient data in these models must be synthetic (no real PHI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NoteType(str, Enum):
    """Clinical note document types extracted from EHR encounters."""
    HISTORY_AND_PHYSICAL = "H&P"
    DISCHARGE_SUMMARY = "Discharge Summary"
    PROGRESS_NOTE = "Progress Note"


class AdjudicationStatus(str, Enum):
    """Final ERA/835 adjudication status values."""
    DENIED = "denied"
    PAID = "paid"
    PENDING = "pending"
    ADJUSTED = "adjusted"


class ModelStage(str, Enum):
    """Model Registry stage values."""
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    FAILED = "Failed"


# ---------------------------------------------------------------------------
# ClinicalNote
# ---------------------------------------------------------------------------

@dataclass
class ClinicalNote:
    """
    A single free-text clinical note scoped to a specific patient encounter.

    Attributes
    ----------
    note_type:
        The document type — H&P, Discharge Summary, or Progress Note.
    note_date:
        ISO-8601 date the note was authored.
    note_datetime:
        Full ISO-8601 datetime (with time) the note was authored; used for
        chronological ordering when multiple notes share the same date.
    author_id:
        Synthetic provider/author identifier (no real PHI).
    patient_account_number:
        Synthetic patient account number linking the note to a claim.
    encounter_id:
        Synthetic encounter identifier scoping the note to one visit.
    text:
        The free-text body of the clinical note (synthetic content only).
    """
    note_type: NoteType
    note_date: date
    note_datetime: datetime
    author_id: str
    patient_account_number: str
    encounter_id: str
    text: str


# ---------------------------------------------------------------------------
# EHRRecord
# ---------------------------------------------------------------------------

@dataclass
class VitalSigns:
    """Most-recent vital signs from an EHR encounter."""
    temperature_f: Optional[float] = None
    systolic_bp_mmhg: Optional[int] = None
    diastolic_bp_mmhg: Optional[int] = None
    heart_rate_bpm: Optional[int] = None
    respiratory_rate_rpm: Optional[int] = None
    oxygen_saturation_pct: Optional[float] = None


@dataclass
class LOINCLabResult:
    """A single LOINC-coded lab result."""
    loinc_code: str
    display_name: str
    value: float
    unit: str
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None
    abnormal_flag: Optional[str] = None  # "H", "L", "N", or None


@dataclass
class SNOMEDProblem:
    """A single SNOMED/ICD-coded problem list entry."""
    code: str
    code_system: str          # "SNOMED-CT" or "ICD-10"
    display_name: str
    onset_date: Optional[date] = None
    clinical_status: str = "active"  # active | resolved | inactive


@dataclass
class Medication:
    """A single entry from the active medication list."""
    medication_name: str
    rxnorm_code: Optional[str]
    dose: Optional[str]
    route: Optional[str]
    frequency: Optional[str]
    prescriber_id: Optional[str] = None


@dataclass
class EHRRecord:
    """
    Structured EHR encounter record as defined in Requirement 2.3.

    All fields match the structured extraction spec.  Absent fields are
    set to None; a partial_ehr_flag lists the names of absent fields.
    """
    # Patient demographics
    patient_account_number: str
    encounter_id: str
    encounter_date: date

    age: Optional[int] = None
    sex: Optional[str] = None              # "M", "F", "U"
    race: Optional[str] = None
    ethnicity: Optional[str] = None

    # Vital signs
    vitals: Optional[VitalSigns] = None

    # LOINC lab results
    lab_results: Optional[List[LOINCLabResult]] = None

    # SNOMED / ICD-coded problem list
    problem_list: Optional[List[SNOMEDProblem]] = None

    # Active medication list
    medications: Optional[List[Medication]] = None

    # Administrative
    admitting_department: Optional[str] = None
    admitting_physician_id: Optional[str] = None
    length_of_stay_days: Optional[int] = None

    # EHR quality flags (populated by EHR_Ingestion_Service)
    partial_ehr_flag: Optional[str] = None   # comma-separated absent field names
    ehr_null_match_flag: int = 0


# ---------------------------------------------------------------------------
# ClaimRecord
# ---------------------------------------------------------------------------

@dataclass
class ServiceLine:
    """A single service line from a claim."""
    revenue_code: Optional[str] = None
    cpt_hcpcs_code: str = ""
    service_date: Optional[date] = None
    charge_amount: Optional[float] = None
    units_of_service: Optional[int] = None


@dataclass
class ClaimRecord:
    """
    Normalized representation of a parsed X12 837 claim.

    Field names and formats follow Requirements 1.3–1.6.
    """
    # Header
    claim_id: str
    provider_npi: str                  # 10-digit numeric string
    payer_id: str                      # trimmed alphanumeric
    patient_account_number: str
    billing_provider_street: Optional[str] = None
    billing_provider_city: Optional[str] = None
    billing_provider_state: Optional[str] = None
    billing_provider_zip: Optional[str] = None

    # Patient demographics
    patient_last_name: Optional[str] = None
    patient_first_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None       # "M", "F", or "U"
    member_id: Optional[str] = None
    group_number: Optional[str] = None

    # Diagnosis codes
    principal_icd10_code: Optional[str] = None
    secondary_icd10_codes: List[str] = field(default_factory=list)

    # Service lines
    service_lines: List[ServiceLine] = field(default_factory=list)

    # Dates
    statement_period_start: Optional[date] = None
    statement_period_end: Optional[date] = None
    admission_date: Optional[date] = None
    discharge_date: Optional[date] = None

    # Claim metadata
    claim_type: Optional[str] = None           # "professional" | "institutional"
    insurance_type: Optional[str] = None
    admission_type: Optional[str] = None
    admission_source: Optional[str] = None
    claim_status: str = "Active"

    # EHR linkage flags
    partial_ehr_flag: Optional[str] = None
    ehr_null_match_flag: int = 0


# ---------------------------------------------------------------------------
# ClaimFeatureRecord  (mirrors the Avro schema in design.md)
# ---------------------------------------------------------------------------

@dataclass
class ClaimFeatureRecord:
    """
    Denormalized Feature Store row keyed by claim_id.

    Maps 1-to-1 with the Avro schema in claim_denial/schemas/claim_feature_record.avsc.
    """
    schema_version: str
    claim_id: str
    claim_status: str

    # Claim-Level Features
    claim_submission_date_doy: int = 0
    claim_submission_date_month: int = 0
    patient_age: int = -1
    gender: int = -1
    ins_type_medicare: int = 0
    ins_type_medicaid: int = 0
    ins_type_commercial: int = 0
    ins_type_tricare: int = 0
    ins_type_champva: int = 0
    ins_type_other: int = 0
    claim_type_professional: int = 0
    claim_type_institutional: int = 0
    adm_type_emergency: int = 0
    adm_type_elective: int = 0
    adm_type_urgent: int = 0
    adm_type_trauma: int = 0
    adm_src_physician_referral: int = 0
    adm_src_transfer_hospital: int = 0
    adm_src_transfer_snf: int = 0
    adm_src_er: int = 0
    adm_src_court_law: int = 0
    adm_src_not_available: int = 0
    partial_ehr_flag: Optional[str] = None
    ehr_null_match_flag: int = 0

    # Provider / Facility Features
    billing_provider_id_encoded: float = 0.0
    performing_physician_specialty: str = "UNKNOWN"
    provider_historical_denial_rate: float = 0.0
    provider_claim_volume: int = 0
    facility_bed_size: int = 0

    # Clinical Features
    principal_dx_ccs_category: int = 0
    secondary_dx_count: int = 0
    procedure_category_vector: List[int] = field(default_factory=list)
    length_of_stay: int = -2
    comorbidity_risk_score: int = 0
    diagnosis_procedure_mismatch: int = 0

    # Payer Behavior Features
    payer_historical_denial_rate: float = 0.0
    payer_historical_denial_rate_by_type: float = 0.0
    days_since_last_payment: int = -1
    payer_contract_stop_loss: int = 0

    # NLP Features
    documents_medical_necessity: float = 0.5
    mentions_lack_of_pre_auth: int = 0
    tx_plan_complexity_embedding: List[float] = field(default_factory=lambda: [0.0] * 768)
    nlp_notes_absent_flag: int = 0
    nlp_model_name: Optional[str] = None
    nlp_model_version: Optional[str] = None

    # Scoring Output
    predicted_denial_score: Optional[float] = None
    score_quality_flag: Optional[str] = None
    model_version_id: Optional[str] = None
    scoring_timestamp_utc: Optional[str] = None

    # Ground Truth (post-adjudication)
    adjudication_outcome: Optional[str] = None
    adjudication_timestamp_utc: Optional[str] = None


# ---------------------------------------------------------------------------
# MonitoringRecord
# ---------------------------------------------------------------------------

@dataclass
class MonitoringRecord:
    """Nightly monitoring dashboard entry."""
    computation_date: date
    rolling_precision: Optional[float]
    window_days: int
    total_scored_claims: int
    predicted_denials: int
    true_positive_denials: int
    false_positive_denials: int
    computation_timestamp_utc: datetime


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# HCUP CCS valid range  (0 = unknown, 1–285 = valid categories)
HCUP_CCS_MIN: int = 0
HCUP_CCS_MAX: int = 285

# Sentinel values
SENTINEL_UNKNOWN_INT: int = -1          # null/unknown categorical / missing age
SENTINEL_BOTH_DATES_ABSENT: int = -2   # both admission and discharge absent
SENTINEL_NLP_DEFAULT_NECESSITY: float = 0.5
SENTINEL_NLP_EMBEDDING_DIM: int = 768

# Insurance type one-hot categories (order matches feature-vector columns)
INSURANCE_TYPE_CATEGORIES: List[str] = [
    "Medicare", "Medicaid", "Commercial", "TriCare", "ChampVA", "Other"
]

# Admission type one-hot categories
ADMISSION_TYPE_CATEGORIES: List[str] = [
    "Emergency", "Elective", "Urgent", "Trauma"
]

# Admission source one-hot categories
ADMISSION_SOURCE_CATEGORIES: List[str] = [
    "Physician Referral",
    "Transfer from Hospital",
    "Transfer from SNF",
    "Emergency Room",
    "Court/Law Enforcement",
    "Not Available",
]
