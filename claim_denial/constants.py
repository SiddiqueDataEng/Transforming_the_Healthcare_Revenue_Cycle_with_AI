"""
Shared constants for the Claim Denial Prediction system.

Includes HCUP CCS/procedure lookup stubs, sentinel values, and
one-hot category lists used throughout feature engineering and
synthetic data generation.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Sentinel values
# ──────────────────────────────────────────────────────────────────────────────

SENTINEL_MISSING_INT: int = -1          # single absent date / missing field
SENTINEL_BOTH_DATES_ABSENT: int = -2   # both admission and discharge dates absent
SENTINEL_GENDER_UNKNOWN: int = -1
SENTINEL_AGE_UNKNOWN: int = -1
SENTINEL_DENIAL_RATE_DEFAULT: float = 0.5
NLP_DEFAULT_MEDICAL_NECESSITY: float = 0.5
NLP_EMBEDDING_DIM: int = 768

# ──────────────────────────────────────────────────────────────────────────────
# One-hot category lists
# ──────────────────────────────────────────────────────────────────────────────

INSURANCE_TYPES: list[str] = [
    "Medicare",
    "Medicaid",
    "Commercial",
    "TriCare",
    "ChampVA",
    "Other",
]

CLAIM_TYPES: list[str] = [
    "Professional",
    "Institutional",
]

ADMISSION_TYPES: list[str] = [
    "Emergency",
    "Elective",
    "Urgent",
    "Trauma",
]

ADMISSION_SOURCES: list[str] = [
    "Physician Referral",
    "Transfer from Hospital",
    "Transfer from SNF",
    "Emergency Room",
    "Court/Law Enforcement",
    "Not Available",
]

# ──────────────────────────────────────────────────────────────────────────────
# HCUP CCS — single-level categories
#
# The full HCUP CCS mapping covers ICD-10-CM codes → integer categories 1–285.
# Below is a representative subset used for synthetic data generation and
# property-based tests.  The real lookup would be loaded from an external
# reference file in production.
# ──────────────────────────────────────────────────────────────────────────────

HCUP_CCS_RANGE_MIN: int = 1
HCUP_CCS_RANGE_MAX: int = 285
HCUP_CCS_UNKNOWN: int = 0  # sentinel for codes absent from the lookup

# Mapping: ICD-10-CM code prefix → CCS category (representative sample).
# Keys are dot-stripped, upper-cased ICD-10 codes as stored after normalisation.
HCUP_CCS_LOOKUP: dict[str, int] = {
    # Infectious and parasitic diseases (CCS 1–8)
    "A000": 1, "A001": 1, "A009": 1,
    "A040": 2, "A041": 2, "A049": 2,
    "B180": 3, "B181": 3, "B182": 3,
    # Neoplasms (CCS 11–45)
    "C000": 11, "C001": 11, "C189": 11,
    "C220": 12, "C221": 12,
    "C500": 13, "C501": 13, "C509": 13,
    "C180": 14, "C181": 14,
    "D050": 45, "D051": 45,
    # Endocrine / nutritional (CCS 48–58)
    "E100": 49, "E101": 49, "E109": 49,
    "E110": 50, "E111": 50, "E119": 50,
    "E780": 53, "E785": 53,
    # Mental disorders (CCS single-level ICD-10: schizophrenia=192, mood=191)
    "F200": 192, "F201": 192,
    "F319": 191, "F320": 191,
    # Diseases of the nervous system (CCS 76–95)
    "G300": 76, "G309": 76,
    "G400": 83, "G409": 83,
    # Circulatory (CCS 96–121)
    "I100": 98, "I109": 98,
    "I200": 100, "I210": 100, "I219": 100,
    "I500": 108, "I501": 108, "I509": 108,
    "I630": 109, "I639": 109,
    # Respiratory (CCS 122–134)
    "J180": 122, "J189": 122,
    "J440": 127, "J441": 127, "J449": 127,
    "J450": 128, "J451": 128, "J459": 128,
    # Digestive (CCS 135–155)
    "K250": 138, "K259": 138,
    "K400": 142, "K409": 142,
    "K920": 154, "K921": 154,
    # Musculoskeletal (CCS 200–212)
    "M0600": 200, "M0690": 200,
    "M1600": 203, "M1611": 203,
    "M8000": 210, "M8001": 210,
    # Injury / poisoning (CCS 225–244)
    "S0200": 225, "S0201": 225,
    "S7200": 226, "S7201": 226,
    # Symptoms (CCS 245–259)
    "R000": 245, "R001": 245,
    "R1000": 251, "R109": 251,
}

# Active ICD-10 codes for draw during synthetic data generation
HCUP_CCS_ACTIVE_ICD10: list[str] = sorted(HCUP_CCS_LOOKUP.keys())

# ──────────────────────────────────────────────────────────────────────────────
# HCUP Procedure categories — representative sample
#
# Maps CPT/HCPCS → HCUP procedure category index (0-based within the vector).
# The vector length equals len(HCUP_PROC_CATEGORIES).
# ──────────────────────────────────────────────────────────────────────────────

HCUP_PROC_CATEGORIES: list[str] = [
    "Diagnostic radiology",          # 0
    "Diagnostic ultrasound",         # 1
    "Respiratory therapy",           # 2
    "Electrocardiogram",             # 3
    "Coronary arteriography",        # 4
    "Cardiac catheterization",       # 5
    "Bypass graft",                  # 6
    "Spinal fusion",                 # 7
    "Hip replacement",               # 8
    "Knee replacement",              # 9
    "Appendectomy",                  # 10
    "Cholecystectomy",               # 11
    "Hysterectomy",                  # 12
    "Cesarean section",              # 13
    "Colonoscopy",                   # 14
    "Upper GI endoscopy",            # 15
    "Hemodialysis",                  # 16
    "Chemotherapy",                  # 17
    "Radiation therapy",             # 18
    "Blood transfusion",             # 19
    "Physical therapy",              # 20
    "Occupational therapy",          # 21
    "Lab tests",                     # 22
    "Pathology",                     # 23
    "Other diagnostic",              # 24
]

HCUP_PROC_VECTOR_LENGTH: int = len(HCUP_PROC_CATEGORIES)

# CPT/HCPCS → category index  (representative subset of active codes)
HCUP_PROC_LOOKUP: dict[str, int] = {
    # Diagnostic radiology
    "71046": 0, "71048": 0, "72141": 0, "72148": 0,
    # Diagnostic ultrasound
    "93306": 1, "76700": 1, "76705": 1,
    # Respiratory therapy
    "94640": 2, "94002": 2,
    # Electrocardiogram
    "93000": 3, "93005": 3, "93010": 3,
    # Coronary arteriography
    "93454": 4, "93455": 4, "93456": 4,
    # Cardiac catheterization
    "93460": 5, "93461": 5,
    # Bypass graft
    "33510": 6, "33511": 6, "33533": 6,
    # Spinal fusion
    "22612": 7, "22630": 7, "22633": 7,
    # Hip replacement
    "27130": 8, "27132": 8,
    # Knee replacement
    "27447": 9, "27446": 9,
    # Appendectomy
    "44950": 10, "44960": 10,
    # Cholecystectomy
    "47562": 11, "47563": 11,
    # Hysterectomy
    "58150": 12, "58180": 12,
    # Cesarean section
    "59510": 13, "59514": 13,
    # Colonoscopy
    "45378": 14, "45380": 14, "45385": 14,
    # Upper GI endoscopy
    "43239": 15, "43239": 15,
    # Hemodialysis
    "90935": 16, "90937": 16,
    # Chemotherapy
    "96413": 17, "96415": 17,
    # Radiation therapy
    "77385": 18, "77386": 18,
    # Blood transfusion
    "36430": 19, "36440": 19,
    # Physical therapy
    "97110": 20, "97010": 20,
    # Occupational therapy
    "97165": 21, "97166": 21,
    # Lab tests
    "80053": 22, "80048": 22, "85025": 22,
    # Pathology
    "88305": 23, "88307": 23,
    # Other diagnostic
    "99213": 24, "99214": 24, "99215": 24, "99232": 24,
}

# Active CPT/HCPCS codes available for draw during synthetic data generation
HCUP_PROC_ACTIVE_CODES: list[str] = sorted(set(HCUP_PROC_LOOKUP.keys()))
