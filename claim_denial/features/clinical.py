"""
Clinical feature engineering for the Claim Denial Prediction system.

Implements ClinicalFeatureEngineer, which computes all six clinical feature
columns (Requirements 5.1–5.6) and upserts them to the Feature Store.

The Feature Store is represented as an injectable dict-like object for
testability (``Dict[str, Dict[str, Any]]``).  In production this would be
replaced by a real HBase/Cassandra client.

Canonical lookup tables (CCS, procedure, Charlson, mismatch) are fully
injectable so every function can be exercised in isolation without touching
module-level state.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from claim_denial.constants import (
    HCUP_CCS_LOOKUP,
    HCUP_PROC_LOOKUP,
    HCUP_PROC_CATEGORIES,
    HCUP_CCS_UNKNOWN,
    SENTINEL_MISSING_INT,
    SENTINEL_BOTH_DATES_ABSENT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample / stub lookup tables
#
# These are module-level singletons suitable for use in tests and demos.
# All production code should inject its own validated reference data.
# ---------------------------------------------------------------------------

#: Stub HCUP CCS lookup: ICD-10 code (upper-case, dot-stripped) → category int.
#: Imported from constants — re-exported here for convenient test injection.
SAMPLE_CCS_LOOKUP: Dict[str, int] = dict(HCUP_CCS_LOOKUP)

#: Stub HCUP procedure lookup: CPT/HCPCS code → 0-based category index.
#: Imported from constants — re-exported here for convenient test injection.
SAMPLE_PROC_LOOKUP: Dict[str, int] = dict(HCUP_PROC_LOOKUP)

#: Number of procedure categories in the active reference version.
SAMPLE_PROC_VECTOR_LENGTH: int = len(HCUP_PROC_CATEGORIES)

# ---------------------------------------------------------------------------
# Charlson Comorbidity Index — ICD-10 mapping
#
# Standard Charlson ICD-10 mapping (Quan et al. 2005 / Deyo adaptation).
# Each key is a dot-stripped upper-case ICD-10 code; the value is the integer
# weight assigned by the scoring algorithm.
# Reference: https://doi.org/10.1097/01.mlr.0000182534.19832.83
# ---------------------------------------------------------------------------

SAMPLE_CHARLSON_MAPPING: Dict[str, int] = {
    # ── Weight 1 ──────────────────────────────────────────────────────────
    # Myocardial infarction
    "I21": 1, "I22": 1, "I25": 1,
    # Congestive heart failure
    "I50": 1,
    # Peripheral vascular disease
    "I70": 1, "I71": 1, "I73": 1, "I77": 1,
    # Cerebrovascular disease
    "I60": 1, "I61": 1, "I62": 1, "I63": 1, "I65": 1, "I66": 1, "I67": 1,
    "I69": 1, "G45": 1, "G46": 1,
    # Dementia
    "F00": 1, "F01": 1, "F02": 1, "F03": 1, "G30": 1,
    # Chronic pulmonary disease
    "J40": 1, "J41": 1, "J42": 1, "J43": 1, "J44": 1, "J45": 1, "J46": 1,
    "J47": 1, "J60": 1, "J61": 1, "J62": 1, "J63": 1, "J64": 1, "J65": 1,
    "J66": 1, "J67": 1,
    # Connective tissue / rheumatic disease
    "M05": 1, "M06": 1, "M32": 1, "M33": 1, "M34": 1, "M35": 1,
    # Peptic ulcer disease
    "K25": 1, "K26": 1, "K27": 1, "K28": 1,
    # Mild liver disease
    "B18": 1, "K70": 1, "K73": 1, "K74": 1,
    # Diabetes without end-organ damage
    "E10": 1, "E11": 1, "E12": 1, "E13": 1, "E14": 1,
    # ── Weight 2 ──────────────────────────────────────────────────────────
    # Hemiplegia or paraplegia
    "G81": 2, "G82": 2,
    # Renal disease
    "N18": 2, "N19": 2, "N25": 2,
    # Diabetes with end-organ damage
    "E102": 2, "E112": 2, "E122": 2, "E132": 2, "E142": 2,
    # Any malignancy (including leukaemia, lymphoma)
    "C00": 2, "C01": 2, "C02": 2, "C03": 2, "C04": 2, "C05": 2,
    "C06": 2, "C07": 2, "C08": 2, "C09": 2, "C10": 2, "C11": 2,
    "C12": 2, "C13": 2, "C14": 2, "C15": 2, "C16": 2, "C17": 2,
    "C18": 2, "C19": 2, "C20": 2, "C21": 2, "C22": 2, "C23": 2,
    "C24": 2, "C25": 2, "C26": 2, "C30": 2, "C31": 2, "C32": 2,
    "C33": 2, "C34": 2, "C37": 2, "C38": 2, "C39": 2, "C40": 2,
    "C41": 2, "C43": 2, "C45": 2, "C46": 2, "C47": 2, "C48": 2,
    "C49": 2, "C50": 2, "C51": 2, "C52": 2, "C53": 2, "C54": 2,
    "C55": 2, "C56": 2, "C57": 2, "C58": 2, "C60": 2, "C61": 2,
    "C62": 2, "C63": 2, "C64": 2, "C65": 2, "C66": 2, "C67": 2,
    "C68": 2, "C69": 2, "C70": 2, "C71": 2, "C72": 2, "C73": 2,
    "C74": 2, "C75": 2, "C76": 2, "C81": 2, "C82": 2, "C83": 2,
    "C84": 2, "C85": 2, "C88": 2, "C90": 2, "C91": 2, "C92": 2,
    "C93": 2, "C94": 2, "C95": 2, "C96": 2, "C97": 2,
    # ── Weight 3 ──────────────────────────────────────────────────────────
    # Moderate or severe liver disease
    "K721": 3, "K729": 3, "K766": 3, "K767": 3,
    # ── Weight 6 ──────────────────────────────────────────────────────────
    # Metastatic solid tumour
    "C77": 6, "C78": 6, "C79": 6, "C80": 6,
    # AIDS / HIV disease
    "B20": 6, "B21": 6, "B22": 6, "B24": 6,
}

# ---------------------------------------------------------------------------
# Diagnosis–procedure mismatch lookup
#
# Maps principal CCS category → set of expected (acceptable) HCUP procedure
# category indices.  A claim is flagged as a mismatch (1) when the
# procedure_vector has no overlap with the expected set.
# This is a representative stub for testing; production data would be loaded
# from a versioned reference file.
# ---------------------------------------------------------------------------

SAMPLE_MISMATCH_LOOKUP: Dict[int, List[int]] = {
    # CCS 100 — Acute MI: expect cardiac catheterisation (5) or coronary
    # arteriography (4) or bypass graft (6)
    100: [4, 5, 6],
    # CCS 108 — Congestive heart failure: expect ECG (3) or cardiac cath (5)
    108: [3, 5],
    # CCS 127 — COPD: expect respiratory therapy (2)
    127: [2],
    # CCS 128 — Asthma: expect respiratory therapy (2)
    128: [2],
    # CCS 142 — Hernia: expect appendectomy (10) — used as test placeholder
    142: [10],
    # CCS 203 — Hip fracture: expect hip replacement (8)
    203: [8],
    # CCS 226 — Femur fracture: expect hip replacement (8)
    226: [8],
    # CCS 138 — Peptic ulcer: expect upper GI endoscopy (15)
    138: [15],
    # CCS 14  — Colonoscopy / colon tumour: expect colonoscopy (14)
    14: [14],
    # CCS 17  — Hemodialysis: expect hemodialysis (16)
    16: [16],
}


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def map_principal_dx_to_ccs(
    icd10_code: Optional[str],
    ccs_lookup: Dict[str, int],
) -> int:
    """Return the HCUP CCS single-level category for *icd10_code*.

    The lookup key is normalised (upper-case, dot-stripped) before matching.
    Returns 0 (``HCUP_CCS_UNKNOWN``) when the code is absent from
    *ccs_lookup* or when *icd10_code* is ``None`` / empty.

    Parameters
    ----------
    icd10_code:
        Raw ICD-10-CM code string (dots and mixed case accepted).
    ccs_lookup:
        Mapping of normalised ICD-10 code → CCS category integer (1–285).

    Returns
    -------
    int
        CCS category in [1, 285], or 0 for unknown / absent.
    """
    if not icd10_code:
        return HCUP_CCS_UNKNOWN

    normalised = icd10_code.upper().replace(".", "").strip()
    return ccs_lookup.get(normalised, HCUP_CCS_UNKNOWN)


def compute_secondary_dx_count(secondary_codes: Optional[List[str]]) -> int:
    """Return the count of secondary ICD-10 diagnosis codes.

    Parameters
    ----------
    secondary_codes:
        List of secondary ICD-10 codes from the claim; may be ``None`` or
        empty.

    Returns
    -------
    int
        Non-negative integer count; 0 for ``None`` or empty list.
    """
    if not secondary_codes:
        return 0
    return len(secondary_codes)


def build_procedure_multihot_vector(
    cpt_hcpcs_codes: Optional[List[str]],
    hcup_proc_lookup: Dict[str, int],
    vector_length: Optional[int] = None,
) -> List[int]:
    """Build a fixed-length multi-hot binary vector over HCUP procedure categories.

    Each element corresponds to one HCUP procedure category.  An element is
    set to 1 when at least one CPT/HCPCS code on the claim maps to that
    category; unrecognised codes are silently ignored.

    Parameters
    ----------
    cpt_hcpcs_codes:
        Collection of CPT/HCPCS code strings from the claim's service lines.
    hcup_proc_lookup:
        Mapping of CPT/HCPCS code → 0-based HCUP procedure category index.
    vector_length:
        Expected length of the output vector.  When ``None``, falls back to
        the maximum index found in *hcup_proc_lookup* + 1, or
        ``SAMPLE_PROC_VECTOR_LENGTH`` when the lookup is empty.

    Returns
    -------
    List[int]
        Binary vector of length *vector_length* with values in {0, 1}.
    """
    # Determine the canonical vector length
    if vector_length is None:
        if hcup_proc_lookup:
            vector_length = max(hcup_proc_lookup.values()) + 1
        else:
            vector_length = SAMPLE_PROC_VECTOR_LENGTH

    vec: List[int] = [0] * vector_length

    if not cpt_hcpcs_codes:
        return vec

    for code in cpt_hcpcs_codes:
        if code is None:
            continue
        normalised = code.strip().upper()
        idx = hcup_proc_lookup.get(normalised)
        # Also try original case (some lookups use mixed-case keys)
        if idx is None:
            idx = hcup_proc_lookup.get(code.strip())
        if idx is not None and 0 <= idx < vector_length:
            vec[idx] = 1

    return vec


def compute_length_of_stay(
    admission_date: Optional[date],
    discharge_date: Optional[date],
) -> int:
    """Compute whole-day length of stay from admission and discharge dates.

    Sentinel values (per Requirement 5.4 / Property 11):
    * **−1** — only the discharge date is absent.
    * **−2** — both admission and discharge dates are absent.
    * Non-negative integer — whole days (``discharge_date − admission_date``).

    Parameters
    ----------
    admission_date:
        Date the patient was admitted; may be ``None``.
    discharge_date:
        Date the patient was discharged; may be ``None``.

    Returns
    -------
    int
        Length of stay in whole days, or a sentinel value.
    """
    if admission_date is None and discharge_date is None:
        return SENTINEL_BOTH_DATES_ABSENT  # −2

    if discharge_date is None:
        return SENTINEL_MISSING_INT  # −1

    if admission_date is None:
        # Discharge present but admission absent — treat same as only discharge
        # absent (one date missing) → return −1 sentinel.
        return SENTINEL_MISSING_INT  # −1

    # Both dates present — compute whole days (non-negative by construction
    # when discharge >= admission, but we clamp at 0 defensively).
    delta = (discharge_date - admission_date).days
    return max(0, delta)


def compute_charlson_comorbidity_index(
    secondary_icd10_codes: Optional[List[str]],
    charlson_mapping: Dict[str, int],
) -> int:
    """Compute the Charlson Comorbidity Index from secondary ICD-10 codes.

    The algorithm matches each secondary code against *charlson_mapping* using
    a hierarchical prefix search: the full normalised code is tried first, then
    progressively shorter prefixes (down to 3 characters) until a match is
    found.  Each comorbidity category is counted at most once, avoiding
    double-counting when multiple codes map to the same category.

    Parameters
    ----------
    secondary_icd10_codes:
        List of secondary ICD-10-CM codes from the claim; may be ``None`` or
        empty.
    charlson_mapping:
        Mapping of normalised ICD-10 prefix → Charlson weight.

    Returns
    -------
    int
        Non-negative integer Charlson score; 0 for absent / empty code list.
    """
    if not secondary_icd10_codes:
        return 0

    total_score = 0
    # Track matched prefixes to avoid double-counting the same comorbidity
    matched_prefixes: set[str] = set()

    for raw_code in secondary_icd10_codes:
        if not raw_code:
            continue
        normalised = raw_code.upper().replace(".", "").strip()

        # Try the full code, then progressively shorter prefixes (min 3 chars)
        matched = False
        for length in range(len(normalised), 2, -1):
            prefix = normalised[:length]
            if prefix in charlson_mapping and prefix not in matched_prefixes:
                total_score += charlson_mapping[prefix]
                matched_prefixes.add(prefix)
                matched = True
                break
            # If the prefix was already matched, skip it (no double-counting)
            if prefix in matched_prefixes:
                matched = True
                break

        if not matched:
            # No entry found in mapping — code contributes 0
            pass

    return total_score


def compute_diagnosis_procedure_mismatch(
    principal_ccs: int,
    procedure_vector: List[int],
    mismatch_lookup: Dict[int, List[int]],
) -> int:
    """Return a binary mismatch flag for a principal diagnosis / procedure pair.

    A mismatch (1) is returned when the principal CCS category has an entry in
    *mismatch_lookup* AND none of the expected procedure category bits are set
    in *procedure_vector*.

    Returns 0 (no mismatch) in the following cases:
    * The principal CCS category has no entry in *mismatch_lookup*.
    * The procedure vector has at least one expected bit set.

    Parameters
    ----------
    principal_ccs:
        HCUP CCS category of the principal diagnosis (0 = unknown).
    procedure_vector:
        Multi-hot binary vector over HCUP procedure categories.
    mismatch_lookup:
        Mapping of principal CCS category → list of acceptable procedure
        category indices.

    Returns
    -------
    int
        1 if a mismatch is detected, 0 otherwise.
    """
    expected_indices = mismatch_lookup.get(principal_ccs)
    if expected_indices is None:
        # No rule for this CCS category — default to no mismatch
        return 0

    # Check whether any expected procedure category is active in the vector
    for idx in expected_indices:
        if 0 <= idx < len(procedure_vector) and procedure_vector[idx] == 1:
            return 0  # At least one expected procedure is present → no mismatch

    return 1  # None of the expected procedures are present → mismatch


# ---------------------------------------------------------------------------
# Feature Store type alias
# ---------------------------------------------------------------------------

# The Feature Store is modelled as a plain dict for testability.
# Keys are claim_id strings; values are dicts of feature column → value.
FeatureStore = Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# ClinicalFeatureEngineer
# ---------------------------------------------------------------------------


class ClinicalFeatureEngineer:
    """Compute clinical features and upsert them to the Feature Store.

    All lookup tables are constructor-injected so the class can be used in
    tests without touching module-level state.  When not supplied, the module-
    level sample/stub tables are used.

    Parameters
    ----------
    feature_store:
        Injectable dict-like Feature Store.  If ``None``, an internal empty
        dict is created (useful for ephemeral testing).
    ccs_lookup:
        ICD-10 → CCS category mapping.  Defaults to ``SAMPLE_CCS_LOOKUP``.
    proc_lookup:
        CPT/HCPCS → HCUP procedure category index mapping.  Defaults to
        ``SAMPLE_PROC_LOOKUP``.
    proc_vector_length:
        Fixed length of the procedure multi-hot vector.  Defaults to
        ``SAMPLE_PROC_VECTOR_LENGTH``.
    charlson_mapping:
        ICD-10 prefix → Charlson weight mapping.  Defaults to
        ``SAMPLE_CHARLSON_MAPPING``.
    mismatch_lookup:
        Principal CCS → list of acceptable procedure category indices.
        Defaults to ``SAMPLE_MISMATCH_LOOKUP``.
    """

    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        ccs_lookup: Optional[Dict[str, int]] = None,
        proc_lookup: Optional[Dict[str, int]] = None,
        proc_vector_length: Optional[int] = None,
        charlson_mapping: Optional[Dict[str, int]] = None,
        mismatch_lookup: Optional[Dict[int, List[int]]] = None,
    ) -> None:
        self._feature_store: FeatureStore = feature_store if feature_store is not None else {}
        self._ccs_lookup: Dict[str, int] = ccs_lookup if ccs_lookup is not None else SAMPLE_CCS_LOOKUP
        self._proc_lookup: Dict[str, int] = proc_lookup if proc_lookup is not None else SAMPLE_PROC_LOOKUP
        self._proc_vector_length: int = (
            proc_vector_length
            if proc_vector_length is not None
            else SAMPLE_PROC_VECTOR_LENGTH
        )
        self._charlson_mapping: Dict[str, int] = (
            charlson_mapping if charlson_mapping is not None else SAMPLE_CHARLSON_MAPPING
        )
        self._mismatch_lookup: Dict[int, List[int]] = (
            mismatch_lookup if mismatch_lookup is not None else SAMPLE_MISMATCH_LOOKUP
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_and_upsert(
        self,
        claim_id: str,
        principal_icd10: Optional[str],
        secondary_icd10_codes: Optional[List[str]],
        cpt_hcpcs_codes: Optional[List[str]],
        admission_date: Optional[date],
        discharge_date: Optional[date],
    ) -> Dict[str, Any]:
        """Compute all six clinical features and upsert them to the Feature Store.

        Parameters
        ----------
        claim_id:
            Unique claim identifier used as the Feature Store row key.
        principal_icd10:
            Principal diagnosis ICD-10-CM code.
        secondary_icd10_codes:
            List of secondary ICD-10-CM codes (may be empty or ``None``).
        cpt_hcpcs_codes:
            List of CPT/HCPCS procedure codes from service lines.
        admission_date:
            Patient admission date; ``None`` if absent.
        discharge_date:
            Patient discharge date; ``None`` if absent.

        Returns
        -------
        Dict[str, Any]
            The clinical feature columns that were upserted.
        """
        # ── Step 1: compute all six features ──────────────────────────
        principal_ccs = map_principal_dx_to_ccs(principal_icd10, self._ccs_lookup)

        secondary_count = compute_secondary_dx_count(secondary_icd10_codes)

        proc_vector = build_procedure_multihot_vector(
            cpt_hcpcs_codes,
            self._proc_lookup,
            self._proc_vector_length,
        )

        los = compute_length_of_stay(admission_date, discharge_date)

        charlson = compute_charlson_comorbidity_index(
            secondary_icd10_codes, self._charlson_mapping
        )

        mismatch = compute_diagnosis_procedure_mismatch(
            principal_ccs, proc_vector, self._mismatch_lookup
        )

        # ── Step 2: assemble feature dict ─────────────────────────────
        clinical_features: Dict[str, Any] = {
            "principal_dx_ccs_category": principal_ccs,
            "secondary_dx_count": secondary_count,
            "procedure_category_vector": proc_vector,
            "length_of_stay": los,
            "comorbidity_risk_score": charlson,
            "diagnosis_procedure_mismatch": mismatch,
        }

        # ── Step 3: upsert to Feature Store ───────────────────────────
        self._upsert(claim_id, clinical_features)

        logger.debug(
            "ClinicalFeatureEngineer upserted %d clinical columns for claim_id=%s",
            len(clinical_features),
            claim_id,
        )

        return clinical_features

    # ------------------------------------------------------------------
    # Feature Store access helpers
    # ------------------------------------------------------------------

    def _upsert(self, claim_id: str, columns: Dict[str, Any]) -> None:
        """Write *columns* to the Feature Store row for *claim_id*.

        Creates the row if it does not yet exist; merges (overwrites) only the
        supplied columns when the row already exists.
        """
        if claim_id not in self._feature_store:
            self._feature_store[claim_id] = {}
        self._feature_store[claim_id].update(columns)

    def get_feature_store_row(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Return the full Feature Store row for *claim_id*, or ``None``."""
        return self._feature_store.get(claim_id)

    @property
    def feature_store(self) -> FeatureStore:
        """Direct access to the underlying Feature Store dict (for testing)."""
        return self._feature_store
