"""
SyntheticDataGenerator — produces synthetic (PHI-free) test data for the
Claim Denial Prediction pipeline.

All generated patient identifiers, names, dates, and clinical values are
purely synthetic.  No real Protected Health Information (PHI) is used or
embedded in this module.
"""

from __future__ import annotations

import random
import string
import textwrap
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any

from claim_denial.models import (
    ClinicalNote,
    EHRRecord,
    LOINCLabResult,
    Medication,
    NoteType,
    SNOMEDProblem,
    VitalSigns,
)
from claim_denial.constants import (
    HCUP_CCS_ACTIVE_ICD10,
    HCUP_PROC_ACTIVE_CODES,
)


# ---------------------------------------------------------------------------
# Reference data  (representative synthetic samples only)
# ---------------------------------------------------------------------------

# A small representative sample of HCUP CCS-mapped ICD-10 codes
_HCUP_ICD10_SAMPLE: List[str] = [
    "I10",    # Essential hypertension  (CCS 98)
    "E119",   # Type 2 DM without complications  (CCS 49)
    "J189",   # Pneumonia, unspecified  (CCS 122)
    "N390",   # UTI, unspecified  (CCS 159)
    "M5450",  # Low back pain  (CCS 203)
    "K5900",  # Constipation, unspecified  (CCS 145)
    "F329",   # Major depressive disorder  (CCS 657)
    "G4300",  # Migraine  (CCS 84)
    "Z8673",  # Personal hx of TIA  (CCS 113)
    "S0990XA",# Head injury  (CCS 228)
    "I2510",  # CAD  (CCS 101)
    "J449",   # COPD, unspecified  (CCS 127)
    "E785",   # Hyperlipidemia  (CCS 53)
    "K219",   # GERD  (CCS 138)
    "I48",    # Atrial fibrillation  (CCS 105)
    "C189",   # Colon cancer  (CCS 15)
    "N183",   # CKD stage 3  (CCS 158)
    "D649",   # Anemia, unspecified  (CCS 59)
    "J069",   # URI, unspecified  (CCS 126)
    "K922",   # GI bleed  (CCS 154)
]

# A small representative sample of CPT/HCPCS codes
_CPT_SAMPLE: List[str] = [
    "99213", "99214", "99215",   # Office visits
    "99232", "99233",            # Subsequent hospital care
    "93306",                      # Echo
    "71046",                      # Chest X-ray 2 views
    "85025",                      # CBC with auto differential
    "80053",                      # Comprehensive metabolic panel
    "36415",                      # Venipuncture
    "43239",                      # EGD with biopsy
    "45378",                      # Colonoscopy
    "27447",                      # Total knee replacement
    "33533",                      # CABG arterial
    "70553",                      # Brain MRI w/ contrast
    "93000",                      # ECG
    "J0696",                      # Ceftriaxone injection
]

# LOINC codes for synthetic lab results
_LOINC_LABS: List[Dict[str, Any]] = [
    {"loinc": "2160-0", "name": "Creatinine [Mass/volume] in Serum",
     "low": 0.6, "high": 1.2, "unit": "mg/dL"},
    {"loinc": "2823-3", "name": "Potassium [Moles/volume] in Serum",
     "low": 3.5, "high": 5.0, "unit": "mEq/L"},
    {"loinc": "2947-0", "name": "Sodium [Moles/volume] in Blood",
     "low": 136.0, "high": 145.0, "unit": "mEq/L"},
    {"loinc": "4548-4", "name": "Hemoglobin A1c/Hemoglobin.total in Blood",
     "low": 4.0, "high": 5.6, "unit": "%"},
    {"loinc": "2093-3", "name": "Cholesterol [Mass/volume] in Serum",
     "low": 0.0, "high": 199.0, "unit": "mg/dL"},
    {"loinc": "6299-2", "name": "Urea nitrogen [Mass/volume] in Blood",
     "low": 7.0, "high": 20.0, "unit": "mg/dL"},
    {"loinc": "718-7",  "name": "Hemoglobin [Mass/volume] in Blood",
     "low": 12.0, "high": 17.5, "unit": "g/dL"},
    {"loinc": "26515-7","name": "Platelets [#/volume] in Blood",
     "low": 150.0, "high": 400.0, "unit": "10*3/uL"},
    {"loinc": "6690-2", "name": "Leukocytes [#/volume] in Blood",
     "low": 4.5, "high": 11.0, "unit": "10*3/uL"},
    {"loinc": "2345-7", "name": "Glucose [Mass/volume] in Serum",
     "low": 70.0, "high": 100.0, "unit": "mg/dL"},
]

# SNOMED-CT problem codes (synthetic representative samples)
_SNOMED_PROBLEMS: List[Dict[str, str]] = [
    {"code": "44054006",  "name": "Type 2 diabetes mellitus"},
    {"code": "73211009",  "name": "Diabetes mellitus"},
    {"code": "38341003",  "name": "Hypertensive disorder"},
    {"code": "13645005",  "name": "Chronic obstructive lung disease"},
    {"code": "84114007",  "name": "Heart failure"},
    {"code": "49601007",  "name": "Disorder of cardiovascular system"},
    {"code": "195967001", "name": "Asthma"},
    {"code": "40055000",  "name": "Chronic sinusitis"},
    {"code": "399068003", "name": "Prostate cancer"},
    {"code": "363346000", "name": "Malignant neoplastic disease"},
    {"code": "56265001",  "name": "Heart disease"},
    {"code": "73430006",  "name": "Sleep apnea"},
    {"code": "267425008", "name": "Peripheral neuropathy"},
    {"code": "60234000",  "name": "Abdominal pain"},
    {"code": "36971009",  "name": "Sinusitis"},
]

# Synthetic medication names
_MEDICATIONS: List[Dict[str, str]] = [
    {"name": "Metformin 500 mg", "rxnorm": "860975",  "dose": "500 mg",  "route": "PO", "freq": "BID"},
    {"name": "Lisinopril 10 mg",  "rxnorm": "104375",  "dose": "10 mg",   "route": "PO", "freq": "QD"},
    {"name": "Atorvastatin 40 mg","rxnorm": "617311",  "dose": "40 mg",   "route": "PO", "freq": "QHS"},
    {"name": "Amlodipine 5 mg",   "rxnorm": "197361",  "dose": "5 mg",    "route": "PO", "freq": "QD"},
    {"name": "Omeprazole 20 mg",  "rxnorm": "40790",   "dose": "20 mg",   "route": "PO", "freq": "QD"},
    {"name": "Aspirin 81 mg",     "rxnorm": "243670",  "dose": "81 mg",   "route": "PO", "freq": "QD"},
    {"name": "Levothyroxine 50 mcg","rxnorm":"686924", "dose": "50 mcg",  "route": "PO", "freq": "QD"},
    {"name": "Albuterol 90 mcg",  "rxnorm": "308047",  "dose": "90 mcg",  "route": "INH","freq": "PRN"},
    {"name": "Losartan 50 mg",    "rxnorm": "203644",  "dose": "50 mg",   "route": "PO", "freq": "QD"},
    {"name": "Gabapentin 300 mg", "rxnorm": "283921",  "dose": "300 mg",  "route": "PO", "freq": "TID"},
    {"name": "Sertraline 50 mg",  "rxnorm": "312941",  "dose": "50 mg",   "route": "PO", "freq": "QD"},
    {"name": "Furosemide 40 mg",  "rxnorm": "202991",  "dose": "40 mg",   "route": "PO", "freq": "QD"},
]

# Race / ethnicity values (HL7 value set)
_RACES: List[str] = [
    "White", "Black or African American", "Asian",
    "American Indian or Alaska Native",
    "Native Hawaiian or Other Pacific Islander",
    "Other Race", "Unknown",
]
_ETHNICITIES: List[str] = [
    "Hispanic or Latino", "Not Hispanic or Latino", "Unknown",
]
_SEXES: List[str] = ["M", "F", "U"]

_DEPARTMENTS: List[str] = [
    "Internal Medicine", "Cardiology", "Pulmonology",
    "General Surgery", "Orthopedics", "Neurology",
    "Gastroenterology", "Oncology", "Emergency Medicine",
    "Nephrology",
]

# Keywords that detect_prior_auth_gap classifier must match (requirement 7.2)
_PRE_AUTH_GAP_PHRASES: List[str] = [
    "prior authorization was not obtained before the procedure",
    "no prior authorization on file for this service",
    "pre-authorization request was denied by the payer",
    "authorization number is missing from the claim",
    "service rendered without required prior authorization",
]

# Note templates (synthetic only)
_HP_TEMPLATE = textwrap.dedent("""\
    HISTORY AND PHYSICAL

    Patient Account: {pan}
    Encounter Date:  {enc_date}
    Author:          {author}

    CHIEF COMPLAINT
    {complaint}

    HISTORY OF PRESENT ILLNESS
    The patient is a {age}-year-old {sex_desc} who presents to {dept} with a chief
    complaint of {complaint}.  Symptoms began approximately {onset} days ago and
    have been {course}.  The patient reports {symptom_detail}.  Past medical
    history is significant for {pmh}.

    MEDICATIONS
    {medications}

    REVIEW OF SYSTEMS
    Constitutional: {ros_constitutional}.
    Cardiovascular: {ros_cardio}.
    Respiratory: {ros_resp}.
    Gastrointestinal: {ros_gi}.

    PHYSICAL EXAMINATION
    Vitals: BP {sbp}/{dbp} mmHg, HR {hr} bpm, Temp {temp} F, O2 Sat {o2}%.
    General: {gen_exam}.
    Cardiovascular: {cv_exam}.
    Respiratory: {resp_exam}.
    Abdomen: {abd_exam}.

    ASSESSMENT AND PLAN
    {assessment}

    {pre_auth_section}
    """)

_DISCHARGE_TEMPLATE = textwrap.dedent("""\
    DISCHARGE SUMMARY

    Patient Account: {pan}
    Encounter Date:  {enc_date}
    Discharge Date:  {dc_date}
    Author:          {author}

    ADMISSION DIAGNOSIS
    {admission_dx}

    HOSPITAL COURSE
    The patient was admitted on {enc_date} for management of {admission_dx}.
    {hospital_course}

    DISCHARGE CONDITION
    {condition}

    DISCHARGE MEDICATIONS
    {medications}

    FOLLOW-UP
    The patient should follow up with {dept} in {fu_days} days.

    {pre_auth_section}
    """)

_PROGRESS_TEMPLATE = textwrap.dedent("""\
    PROGRESS NOTE

    Patient Account: {pan}
    Note Date:       {note_date}
    Author:          {author}

    SUBJECTIVE
    {subjective}

    OBJECTIVE
    Vitals: BP {sbp}/{dbp} mmHg, HR {hr} bpm, Temp {temp} F, O2 Sat {o2}%.
    {objective}

    ASSESSMENT
    {assessment}

    PLAN
    {plan}

    {pre_auth_section}
    """)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rand_account_number(rng: random.Random) -> str:
    """Generate a synthetic 10-character alphanumeric patient account number."""
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=10))


def _rand_provider_id(rng: random.Random) -> str:
    """Generate a synthetic 6-character provider author ID."""
    return "DR" + "".join(rng.choices(string.digits, k=4))


def _rand_vitals(rng: random.Random) -> Dict[str, Any]:
    """Return a dict of synthetic vital-sign values."""
    return {
        "temperature_f": round(rng.uniform(97.0, 102.5), 1),
        "systolic_bp_mmhg": rng.randint(90, 180),
        "diastolic_bp_mmhg": rng.randint(55, 110),
        "heart_rate_bpm": rng.randint(50, 130),
        "o2_sat_pct": round(rng.uniform(90.0, 100.0), 1),
    }


def _sex_desc(sex: str) -> str:
    return {"M": "male", "F": "female", "U": "person"}.get(sex, "person")


def _complaint_for_code(icd10: str) -> str:
    _map = {
        "I10": "elevated blood pressure",
        "E119": "elevated blood glucose",
        "J189": "productive cough and fever",
        "N390": "dysuria and urinary frequency",
        "M5450": "low back pain",
        "K5900": "abdominal discomfort and constipation",
        "F329": "depressed mood and fatigue",
        "G4300": "severe headache",
        "I2510": "exertional chest pain",
        "J449": "shortness of breath and wheezing",
    }
    return _map.get(icd10.upper(), "generalized weakness and malaise")


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Produces fully synthetic, PHI-free test data for the Claim Denial
    Prediction pipeline.

    All generated values (account numbers, dates, clinical data) are
    computer-generated and do not represent any real patient, provider,
    or payer.
    """

    # ------------------------------------------------------------------
    # 1.5.2 — EHR record generator
    # ------------------------------------------------------------------

    def generate_ehr_record(
        self,
        patient_account_number: str,
        encounter_date: date,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Produce a JSON-serialisable EHR encounter record matching the full
        field list in Requirement 2.3.

        All values are synthetic.  The returned dict is directly
        JSON-serialisable (dates are returned as ISO-8601 strings).

        Parameters
        ----------
        patient_account_number:
            Synthetic Patient Account Number to embed in the record.
        encounter_date:
            ISO-8601 date of the encounter (``date`` object).
        seed:
            Optional integer seed for reproducible output.

        Returns
        -------
        dict
            A fully populated, JSON-serialisable EHR encounter dict.
            Missing-field semantics: no fields are null in the default
            output; all required fields are populated with synthetic values.
        """
        rng = random.Random(seed)

        # Demographics
        age = rng.randint(18, 89)
        sex = rng.choice(_SEXES)
        race = rng.choice(_RACES)
        ethnicity = rng.choice(_ETHNICITIES)

        # Vital signs
        vitals = _rand_vitals(rng)

        # LOINC labs  (3–6 panels)
        n_labs = rng.randint(3, min(6, len(_LOINC_LABS)))
        chosen_labs = rng.sample(_LOINC_LABS, n_labs)
        lab_results = []
        for lab in chosen_labs:
            value = round(rng.uniform(lab["low"] * 0.7, lab["high"] * 1.3), 2)
            flag = (
                "H" if value > lab["high"]
                else ("L" if value < lab["low"] else "N")
            )
            lab_results.append({
                "loinc_code": lab["loinc"],
                "display_name": lab["name"],
                "value": value,
                "unit": lab["unit"],
                "reference_range_low": lab["low"],
                "reference_range_high": lab["high"],
                "abnormal_flag": flag,
            })

        # SNOMED problem list (1–4 problems)
        n_problems = rng.randint(1, min(4, len(_SNOMED_PROBLEMS)))
        chosen_problems = rng.sample(_SNOMED_PROBLEMS, n_problems)
        onset_offset_days = rng.randint(30, 3650)
        problem_list = []
        for prob in chosen_problems:
            onset = encounter_date - timedelta(days=rng.randint(30, onset_offset_days))
            problem_list.append({
                "code": prob["code"],
                "code_system": "SNOMED-CT",
                "display_name": prob["name"],
                "onset_date": onset.isoformat(),
                "clinical_status": rng.choice(["active", "active", "active", "resolved"]),
            })

        # Medications (1–5)
        n_meds = rng.randint(1, min(5, len(_MEDICATIONS)))
        chosen_meds = rng.sample(_MEDICATIONS, n_meds)
        medications = []
        prescriber_id = _rand_provider_id(rng)
        for med in chosen_meds:
            medications.append({
                "medication_name": med["name"],
                "rxnorm_code": med["rxnorm"],
                "dose": med["dose"],
                "route": med["route"],
                "frequency": med["freq"],
                "prescriber_id": prescriber_id,
            })

        # Encounter administrative data
        admitting_dept = rng.choice(_DEPARTMENTS)
        admitting_physician_id = _rand_provider_id(rng)
        los = rng.randint(1, 14)
        encounter_id = "ENC-" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))

        return {
            "patient_account_number": patient_account_number,
            "encounter_id": encounter_id,
            "encounter_date": encounter_date.isoformat(),

            # Requirement 2.3 fields
            "age": age,
            "sex": sex,
            "race": race,
            "ethnicity": ethnicity,

            "vitals": {
                "temperature_f": vitals["temperature_f"],
                "systolic_bp_mmhg": vitals["systolic_bp_mmhg"],
                "diastolic_bp_mmhg": vitals["diastolic_bp_mmhg"],
                "heart_rate_bpm": vitals["heart_rate_bpm"],
                "oxygen_saturation_pct": vitals["o2_sat_pct"],
            },

            "lab_results": lab_results,
            "problem_list": problem_list,
            "medications": medications,

            "admitting_department": admitting_dept,
            "admitting_physician_id": admitting_physician_id,
            "length_of_stay_days": los,

            # EHR quality flags — none absent in the full record
            "partial_ehr_flag": None,
            "ehr_null_match_flag": 0,
        }

    def generate_partial_ehr_record(
        self,
        absent_fields: List[str],
        patient_account_number: Optional[str] = None,
        encounter_date: Optional[date] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Produce an EHR encounter record with the specified fields set to
        ``null`` and ``partial_ehr_flag`` populated.

        This is for testing the partial-EHR code path in
        ``EHRIngestionService.extract_structured_fields``.

        Parameters
        ----------
        absent_fields:
            List of field names that should be set to ``None`` in the
            returned dict.  Must be a subset of the fields produced by
            ``generate_ehr_record``.
        patient_account_number:
            Synthetic Patient Account Number.  A random value is used
            when ``None``.
        encounter_date:
            Encounter date.  Today is used when ``None``.
        seed:
            Optional seed for reproducibility.

        Returns
        -------
        dict
            A JSON-serialisable EHR encounter dict where every field in
            ``absent_fields`` is ``None`` and ``partial_ehr_flag`` is
            set to a comma-separated string of the absent field names.
        """
        rng = random.Random(seed)

        if patient_account_number is None:
            patient_account_number = _rand_account_number(rng)
        if encounter_date is None:
            encounter_date = date.today()

        record = self.generate_ehr_record(
            patient_account_number=patient_account_number,
            encounter_date=encounter_date,
            seed=seed,
        )

        # Null out each requested field
        for field_name in absent_fields:
            if field_name in record:
                record[field_name] = None

        # Populate partial_ehr_flag with the names of absent fields
        # (only track fields that are actually part of the record)
        actually_absent = [f for f in absent_fields if f in record]
        record["partial_ehr_flag"] = (
            ",".join(actually_absent) if actually_absent else None
        )

        return record

    def generate_clinical_notes(
        self,
        n_hp: int = 1,
        n_discharge: int = 1,
        n_progress: int = 2,
        include_pre_auth_gap: bool = False,
        patient_account_number: Optional[str] = None,
        encounter_date: Optional[date] = None,
        seed: Optional[int] = None,
    ) -> List[ClinicalNote]:
        """
        Produce synthetic clinical notes for an encounter.

        Notes are returned in chronological order (H&P first, then
        progress notes distributed across the LOS, then discharge summary).

        Parameters
        ----------
        n_hp:
            Number of History & Physical notes to generate (typically 1).
        n_discharge:
            Number of Discharge Summary notes to generate (typically 1).
        n_progress:
            Number of Progress Notes to generate.
        include_pre_auth_gap:
            When ``True``, embeds a phrase from ``_PRE_AUTH_GAP_PHRASES``
            in at least one note so that ``NLPProcessor.detect_prior_auth_gap``
            returns 1.
        patient_account_number:
            Synthetic Patient Account Number.  Random if ``None``.
        encounter_date:
            Date of the encounter (H&P date).  Today if ``None``.
        seed:
            Optional seed for reproducibility.

        Returns
        -------
        List[ClinicalNote]
            Notes sorted chronologically by ``note_datetime``.
        """
        rng = random.Random(seed)

        if patient_account_number is None:
            patient_account_number = _rand_account_number(rng)
        if encounter_date is None:
            encounter_date = date.today()

        encounter_id = "ENC-" + "".join(
            rng.choices(string.ascii_uppercase + string.digits, k=8)
        )
        author_id = _rand_provider_id(rng)

        # Pick a representative ICD-10 code for narrative coherence
        principal_icd10 = rng.choice(_HCUP_ICD10_SAMPLE)
        complaint = _complaint_for_code(principal_icd10)

        # EHR vitals for templates
        vitals = _rand_vitals(rng)
        sbp = vitals["systolic_bp_mmhg"]
        dbp = vitals["diastolic_bp_mmhg"]
        hr  = vitals["heart_rate_bpm"]
        temp = vitals["temperature_f"]
        o2  = vitals["o2_sat_pct"]

        # Medication list snippet
        n_meds = rng.randint(2, 4)
        chosen_meds = rng.sample(_MEDICATIONS, min(n_meds, len(_MEDICATIONS)))
        med_text = "\n".join(f"  - {m['name']} {m['freq']}" for m in chosen_meds)

        # LOS for discharge note
        los = n_progress + rng.randint(1, 3)
        dc_date = (encounter_date + timedelta(days=los)).isoformat()

        # Past medical history snippet
        pmh_entries = rng.sample(_SNOMED_PROBLEMS, k=min(3, len(_SNOMED_PROBLEMS)))
        pmh_text = ", ".join(p["name"] for p in pmh_entries)

        dept = rng.choice(_DEPARTMENTS)
        fu_days = rng.randint(7, 30)
        age = rng.randint(18, 89)
        sex = rng.choice(_SEXES)

        # Pre-auth gap phrase (may be embedded in any note)
        pre_auth_phrase = (
            rng.choice(_PRE_AUTH_GAP_PHRASES)
            if include_pre_auth_gap
            else ""
        )
        # Only insert in the FIRST note that needs it (H&P > Discharge > Progress)
        pre_auth_used = False

        def _pre_auth_section(force: bool = False) -> str:
            nonlocal pre_auth_used
            if (force or not pre_auth_used) and include_pre_auth_gap and not pre_auth_used:
                pre_auth_used = True
                return f"\nNOTE: {pre_auth_phrase}\n"
            return ""

        notes: List[ClinicalNote] = []

        # ── History & Physical notes ──────────────────────────────────────
        for i in range(n_hp):
            hp_hour = rng.randint(6, 10) + i
            hp_dt = datetime(
                encounter_date.year, encounter_date.month, encounter_date.day,
                min(hp_hour, 23), rng.randint(0, 59), 0
            )
            hp_text = _HP_TEMPLATE.format(
                pan=patient_account_number,
                enc_date=encounter_date.isoformat(),
                author=author_id,
                complaint=complaint,
                age=age,
                sex_desc=_sex_desc(sex),
                dept=dept,
                onset=rng.randint(1, 14),
                course=rng.choice(["gradually worsening", "stable", "acutely worsening"]),
                symptom_detail=rng.choice([
                    "associated nausea and mild diaphoresis",
                    "no associated fever or chills",
                    "mild shortness of breath on exertion",
                ]),
                pmh=pmh_text,
                medications=med_text,
                ros_constitutional=rng.choice(["Negative for fever", "Positive for fatigue"]),
                ros_cardio=rng.choice(["No chest pain", "Positive for palpitations"]),
                ros_resp=rng.choice(["No cough", "Positive for cough"]),
                ros_gi=rng.choice(["No nausea", "Positive for nausea"]),
                sbp=sbp, dbp=dbp, hr=hr, temp=temp, o2=o2,
                gen_exam=rng.choice(["Alert and oriented", "Appears uncomfortable"]),
                cv_exam=rng.choice(["Regular rate and rhythm", "Irregular rhythm"]),
                resp_exam=rng.choice(["Clear to auscultation bilaterally", "Decreased breath sounds at bases"]),
                abd_exam=rng.choice(["Soft, non-tender, non-distended", "Mild epigastric tenderness"]),
                assessment=f"1. {complaint.capitalize()} — see plan below.\n    Plan: Admit for workup and management.",
                pre_auth_section=_pre_auth_section(force=(i == 0)),
            )
            notes.append(ClinicalNote(
                note_type=NoteType.HISTORY_AND_PHYSICAL,
                note_date=encounter_date,
                note_datetime=hp_dt,
                author_id=author_id,
                patient_account_number=patient_account_number,
                encounter_id=encounter_id,
                text=hp_text,
            ))

        # ── Progress Notes ────────────────────────────────────────────────
        for i in range(n_progress):
            offset_days = i + 1
            pn_date = encounter_date + timedelta(days=offset_days)
            pn_dt = datetime(
                pn_date.year, pn_date.month, pn_date.day,
                rng.randint(7, 15), rng.randint(0, 59), 0
            )
            pn_vitals = _rand_vitals(rng)
            pn_text = _PROGRESS_TEMPLATE.format(
                pan=patient_account_number,
                note_date=pn_date.isoformat(),
                author=author_id,
                subjective=rng.choice([
                    f"Patient reports improvement in {complaint}.",
                    f"Patient continues to have {complaint}. No new complaints.",
                    "Patient is resting comfortably. No acute distress.",
                ]),
                sbp=pn_vitals["systolic_bp_mmhg"],
                dbp=pn_vitals["diastolic_bp_mmhg"],
                hr=pn_vitals["heart_rate_bpm"],
                temp=pn_vitals["temperature_f"],
                o2=pn_vitals["o2_sat_pct"],
                objective=rng.choice([
                    "Labs reviewed — see nursing notes.",
                    "Morning labs pending.",
                    "Labs within acceptable limits.",
                ]),
                assessment=f"1. {complaint.capitalize()} — {rng.choice(['improving', 'stable', 'ongoing management'])}.",
                plan=rng.choice([
                    "Continue current management. Monitor closely.",
                    "Adjust medications per pharmacy recommendations.",
                    "Consult specialty services as needed.",
                ]),
                pre_auth_section=_pre_auth_section(),
            )
            notes.append(ClinicalNote(
                note_type=NoteType.PROGRESS_NOTE,
                note_date=pn_date,
                note_datetime=pn_dt,
                author_id=author_id,
                patient_account_number=patient_account_number,
                encounter_id=encounter_id,
                text=pn_text,
            ))

        # ── Discharge Summary notes ───────────────────────────────────────
        for i in range(n_discharge):
            ds_date = encounter_date + timedelta(days=los + i)
            ds_dt = datetime(
                ds_date.year, ds_date.month, ds_date.day,
                rng.randint(10, 17), rng.randint(0, 59), 0
            )
            ds_vitals = _rand_vitals(rng)
            ds_text = _DISCHARGE_TEMPLATE.format(
                pan=patient_account_number,
                enc_date=encounter_date.isoformat(),
                dc_date=ds_date.isoformat(),
                author=author_id,
                admission_dx=complaint.capitalize(),
                hospital_course=rng.choice([
                    f"Patient was monitored and treated for {complaint}. "
                    f"Clinical condition improved with treatment.",
                    f"Patient underwent evaluation for {complaint}. "
                    f"Appropriate interventions were performed.",
                ]),
                condition=rng.choice(["Stable", "Improved", "Good"]),
                medications=med_text,
                dept=dept,
                fu_days=fu_days,
                pre_auth_section=_pre_auth_section(),
            )
            notes.append(ClinicalNote(
                note_type=NoteType.DISCHARGE_SUMMARY,
                note_date=ds_date,
                note_datetime=ds_dt,
                author_id=author_id,
                patient_account_number=patient_account_number,
                encounter_id=encounter_id,
                text=ds_text,
            ))

        # Sort chronologically
        notes.sort(key=lambda n: n.note_datetime)

        return notes

    # ------------------------------------------------------------------
    # 1.5.1 — X12 837 EDI file generator (previously completed)
    # ------------------------------------------------------------------

    def generate_x12_837(
        self,
        n_claims: int,
        seed: Optional[int] = None,
    ) -> bytes:
        """
        Produce a structurally valid ASC X12 005010X222A1 EDI file
        containing ``n_claims`` claim loops with randomised, spec-conformant
        synthetic data.

        Each claim includes a 10-digit NPI, trimmed alphanumeric Payer ID,
        Patient Account Number, ISO-8601 DOB, gender (M/F/U), at least one
        CPT/HCPCS code, a principal ICD-10 code, and 0–5 secondary ICD-10
        codes.

        Returns raw EDI bytes (segment terminator: ``~``, element
        delimiter: ``*``, component separator: ``:``, line ending: ``\\n``).
        """
        rng = random.Random(seed)
        seg_term = "~"
        elem_sep = "*"

        def npi() -> str:
            return "".join(rng.choices(string.digits, k=10))

        def payer_id() -> str:
            return "".join(
                rng.choices(string.ascii_uppercase + string.digits, k=rng.randint(5, 10))
            ).strip()

        def account_num() -> str:
            return "".join(
                rng.choices(string.ascii_uppercase + string.digits, k=10)
            )

        def dob() -> str:
            yr = rng.randint(1930, 2005)
            mo = rng.randint(1, 12)
            day = rng.randint(1, 28)
            return f"{yr:04d}{mo:02d}{day:02d}"

        def gender() -> str:
            return rng.choice(["M", "F", "U"])

        def service_date() -> str:
            base = date.today() - timedelta(days=rng.randint(1, 90))
            return base.strftime("%Y%m%d")

        # ── ISA / GS envelope ────────────────────────────────────────────
        isa_ctrl = f"{rng.randint(1, 999999999):09d}"
        gs_ctrl = f"{rng.randint(1, 99999):05d}"
        st_ctrl = "0001"
        today_d = date.today().strftime("%Y%m%d")
        today_t = datetime.now().strftime("%H%M")

        segments: List[str] = [
            elem_sep.join([
                "ISA", "00", " " * 10, "00", " " * 10,
                "ZZ", "SENDER" + " " * 9, "ZZ", "RECEIVER" + " " * 7,
                today_d[2:], today_t, "^", "00501",
                isa_ctrl, "0", "P", ":"
            ]),
            elem_sep.join(["GS", "HC", "SENDERGS", "RECEIVERGS",
                           today_d, today_t, gs_ctrl, "X", "005010X222A1"]),
            elem_sep.join(["ST", "837", st_ctrl, "005010X222A1"]),
            elem_sep.join(["BPR", "I", "0", "C", "ACH", "CTX", "", "", "", "",
                           "", "", "", "", "", "", today_d]),
            elem_sep.join(["NM1", "41", "2", "SYNTHETIC BILLING ORG", "",
                           "", "", "", "46", npi()]),
            elem_sep.join(["PER", "IC", "CONTACT NAME", "TE", "5555550000"]),
            elem_sep.join(["NM1", "40", "2", "SYNTHETIC PAYER INC", "",
                           "", "", "", "46", payer_id()]),
            elem_sep.join(["HL", "1", "", "20", "1"]),
            elem_sep.join(["NM1", "85", "2", "SYNTHETIC BILLING", "",
                           "", "", "", "XX", npi()]),
            elem_sep.join(["N3", "123 SYNTHETIC ST"]),
            elem_sep.join(["N4", "ANYTOWN", "CA", "90210"]),
        ]

        # ── Claim loops ──────────────────────────────────────────────────
        for claim_idx in range(n_claims):
            hl_idx = claim_idx + 2
            pan = account_num()
            patient_npi = npi()
            pid = payer_id()
            patient_dob = dob()
            patient_gender = gender()
            svc_date = service_date()

            principal_icd = rng.choice(HCUP_CCS_ACTIVE_ICD10).replace(".", "")
            n_secondary = rng.randint(0, 5)
            secondary_icds = [
                rng.choice(HCUP_CCS_ACTIVE_ICD10).replace(".", "")
                for _ in range(n_secondary)
            ]

            n_cpt = rng.randint(1, 3)
            cpt_codes = rng.sample(HCUP_PROC_ACTIVE_CODES, min(n_cpt, len(HCUP_PROC_ACTIVE_CODES)))

            charge = round(rng.uniform(50.0, 5000.0), 2)

            segments += [
                elem_sep.join(["HL", str(hl_idx), "1", "22", "0"]),
                elem_sep.join(["SBR", "P", "18", "", "", "", "", "", "", "MB"]),
                elem_sep.join(["NM1", "IL", "1", "DOE", "JOHN", "", "", "",
                               "MI", pan]),
                elem_sep.join(["N3", "456 PATIENT AVE"]),
                elem_sep.join(["N4", "SOMECITY", "TX", "75001"]),
                elem_sep.join(["DMG", "D8", patient_dob, patient_gender]),
                elem_sep.join(["NM1", "PR", "2", "SYNTHETIC PAYER INC", "",
                               "", "", "", "PI", pid]),
                elem_sep.join(["CLM", pan, f"{charge:.2f}", "", "",
                               "11:B:1", "Y", "A", "Y", "I"]),
                elem_sep.join(["DTP", "435", "D8", svc_date]),
                # Diagnosis codes
                elem_sep.join(
                    ["HI", "ABK:" + principal_icd]
                    + [f"ABF:{c}" for c in secondary_icds]
                ),
                # Rendering provider
                elem_sep.join(["NM1", "82", "1", "SMITH", "JANE", "", "", "",
                               "XX", patient_npi]),
            ]
            # Service lines
            for cpt in cpt_codes:
                line_charge = round(rng.uniform(50.0, 2000.0), 2)
                segments.append(
                    elem_sep.join([
                        "SV1",
                        f"HC:{cpt}",
                        f"{line_charge:.2f}",
                        "UN",
                        str(rng.randint(1, 5)),
                        "", "", "1",
                    ])
                )
                segments.append(
                    elem_sep.join(["DTP", "472", "D8", svc_date])
                )

        # ── Trailer ──────────────────────────────────────────────────────
        total_segments = len(segments) + 3  # SE, GE, IEA
        segments += [
            elem_sep.join(["SE", str(total_segments), st_ctrl]),
            elem_sep.join(["GE", "1", gs_ctrl]),
            elem_sep.join(["IEA", "1", isa_ctrl]),
        ]

        edi_text = (seg_term + "\n").join(segments) + seg_term + "\n"
        return edi_text.encode("ascii")

    def generate_invalid_x12_837(self, violation: str) -> bytes:
        """
        Produce a syntactically broken X12 837 file with a deliberate
        structural violation for testing ``validate_x12_structure``.

        Parameters
        ----------
        violation:
            A string key identifying the violation type:

            - ``"missing_isa"``   — no ISA envelope at all
            - ``"bad_terminator"``— ISA uses wrong segment terminator (``|``)
            - ``"missing_gs"``    — ISA present but GS is absent
            - ``"truncated_isa"`` — ISA segment has fewer than 16 elements
            - ``"missing_iea"``   — IEA trailer absent
        """
        if violation == "missing_isa":
            return b"GS*HC*SENDER*RECEIVER*20240101*1200*1*X*005010X222A1~\nST*837*0001~\n"

        if violation == "bad_terminator":
            # Uses | instead of ~ as segment terminator
            return (
                b"ISA|00|          |00|          |ZZ|SENDER         |ZZ|RECEIVER       "
                b"|240101|1200|^|00501|000000001|0|P|:|"
                b"GS|HC|SENDER|RECEIVER|20240101|1200|1|X|005010X222A1|"
            )

        if violation == "missing_gs":
            return (
                b"ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       "
                b"*240101*1200*^*00501*000000001*0*P*:~\n"
                b"ST*837*0001*005010X222A1~\n"
            )

        if violation == "truncated_isa":
            # ISA with only 5 elements instead of 16
            return b"ISA*00*          *00*          *ZZ~\n"

        if violation == "missing_iea":
            return (
                b"ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       "
                b"*240101*1200*^*00501*000000001*0*P*:~\n"
                b"GS*HC*SENDER*RECEIVER*20240101*1200*1*X*005010X222A1~\n"
                b"ST*837*0001*005010X222A1~\n"
                b"SE*3*0001~\n"
                b"GE*1*1~\n"
                # IEA intentionally omitted
            )

        raise ValueError(f"Unknown violation type: {violation!r}")

    def generate_x12_837_missing_fields(
        self, missing: List[str]
    ) -> bytes:
        """
        Produce a structurally valid X12 837 envelope with one claim loop
        that intentionally omits the specified required fields.

        Parameters
        ----------
        missing:
            List of required field names to omit.  Valid values:
            ``"npi"``, ``"payer_id"``, ``"patient_account_number"``,
            ``"principal_icd10"``.

        Returns
        -------
        bytes
            EDI bytes — the envelope is valid; the claim loop is missing
            the requested fields so the claim parser should skip that claim.
        """
        rng = random.Random(42)

        def _npi() -> str:
            return "" if "npi" in missing else "".join(rng.choices(string.digits, k=10))

        def _pid() -> str:
            return "" if "payer_id" in missing else "".join(
                rng.choices(string.ascii_uppercase + string.digits, k=8)
            )

        def _pan() -> str:
            return "" if "patient_account_number" in missing else "".join(
                rng.choices(string.ascii_uppercase + string.digits, k=10)
            )

        def _icd() -> str:
            return "" if "principal_icd10" in missing else "I10"

        elem_sep = "*"
        seg_term = "~"
        today_d = date.today().strftime("%Y%m%d")
        today_t = datetime.now().strftime("%H%M")
        isa_ctrl = "000000042"
        gs_ctrl = "00042"

        segments = [
            elem_sep.join([
                "ISA", "00", " " * 10, "00", " " * 10,
                "ZZ", "SENDER" + " " * 9, "ZZ", "RECEIVER" + " " * 7,
                today_d[2:], today_t, "^", "00501",
                isa_ctrl, "0", "P", ":",
            ]),
            elem_sep.join(["GS", "HC", "SENDERGS", "RECEIVERGS",
                           today_d, today_t, gs_ctrl, "X", "005010X222A1"]),
            elem_sep.join(["ST", "837", "0001", "005010X222A1"]),
            elem_sep.join(["BPR", "I", "0", "C", "ACH", "CTX", "", "", "", "",
                           "", "", "", "", "", "", today_d]),
            elem_sep.join(["NM1", "41", "2", "SYNTHETIC BILLING ORG", "",
                           "", "", "", "46", _npi()]),
            elem_sep.join(["PER", "IC", "CONTACT NAME", "TE", "5555550000"]),
            elem_sep.join(["NM1", "40", "2", "SYNTHETIC PAYER INC", "",
                           "", "", "", "46", _pid()]),
            elem_sep.join(["HL", "1", "", "20", "1"]),
            elem_sep.join(["HL", "2", "1", "22", "0"]),
            elem_sep.join(["SBR", "P", "18", "", "", "", "", "", "", "MB"]),
            elem_sep.join(["NM1", "IL", "1", "DOE", "JOHN", "", "", "",
                           "MI", _pan()]),
            elem_sep.join(["NM1", "PR", "2", "SYNTHETIC PAYER", "",
                           "", "", "", "PI", _pid()]),
            elem_sep.join(["CLM", _pan(), "500.00", "", "", "11:B:1",
                           "Y", "A", "Y", "I"]),
        ]

        icd = _icd()
        if icd:
            segments.append(elem_sep.join(["HI", f"ABK:{icd}"]))
        # If principal ICD is missing, omit the HI segment entirely

        segments.append(
            elem_sep.join(["SV1", "HC:99213", "250.00", "UN", "1", "", "", "1"])
        )
        segments.append(elem_sep.join(["DTP", "472", "D8", today_d]))

        total_segments = len(segments) + 3
        segments += [
            elem_sep.join(["SE", str(total_segments), "0001"]),
            elem_sep.join(["GE", "1", gs_ctrl]),
            elem_sep.join(["IEA", "1", isa_ctrl]),
        ]

        edi_text = (seg_term + "\n").join(segments) + seg_term + "\n"
        return edi_text.encode("ascii")

    # ------------------------------------------------------------------
    # 1.5.3 — ERA/835 adjudication outcome generator
    # ------------------------------------------------------------------

    # Denial reason codes mapped to human-readable descriptions
    _DENIAL_REASONS: List[str] = [
        "CO-4: The procedure code is inconsistent with the modifier used or a required modifier is missing.",
        "CO-11: The diagnosis is inconsistent with the procedure.",
        "CO-15: The authorization number is missing, invalid, or does not apply to the billed services.",
        "CO-22: This care may be covered by another payer per coordination of benefits.",
        "CO-29: The time limit for filing has expired.",
        "CO-50: These are non-covered services because this is not deemed a medical necessity by the payer.",
        "CO-97: The benefit for this service is included in the payment/allowance for another service/procedure.",
        "CO-109: Claim not covered by this payer/contractor. You must send the claim to the correct payer/contractor.",
        "PR-1: Deductible Amount.",
        "PR-2: Coinsurance Amount.",
    ]

    def generate_era_835(
        self,
        claim_ids: List[str],
        denial_rate: float = 0.20,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Produce a JSON-serialisable ERA/835 record assigning each claim ID
        a final adjudication status of ``"denied"`` (with probability
        ``denial_rate``) or ``"paid"``.

        The returned dict follows the Data Lake ``era/YYYY/MM/DD/era_835.json``
        schema:

        .. code-block:: json

            {
                "transaction_date": "2024-06-15",
                "payer_id": "XYZ123",
                "claims": [
                    {
                        "claim_id": "CLM-001",
                        "adjudication_status": "paid",
                        "paid_amount": 1234.56,
                        "denial_reason": null
                    },
                    ...
                ]
            }

        All generated values are synthetic (no real PHI).

        Parameters
        ----------
        claim_ids:
            List of claim IDs to adjudicate.  Each ID appears exactly once
            in the returned ``claims`` list.
        denial_rate:
            Probability in [0.0, 1.0] that any given claim is denied.
            Default is 0.20 (20 %).
        seed:
            Optional integer seed for reproducible output.

        Returns
        -------
        dict
            JSON-serialisable ERA/835 dict with keys:
            ``transaction_date``, ``payer_id``, ``claims``.

        Raises
        ------
        ValueError
            If ``denial_rate`` is not in [0.0, 1.0].
        """
        if not 0.0 <= denial_rate <= 1.0:
            raise ValueError(
                f"denial_rate must be in [0.0, 1.0]; got {denial_rate!r}"
            )

        rng = random.Random(seed)

        # Synthetic payer ID
        payer_id = "PAY" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))

        # Transaction date: today or a recent past date (simulate receipt lag)
        transaction_date = (date.today() - timedelta(days=rng.randint(0, 3))).isoformat()

        claim_records: List[dict] = []
        for claim_id in claim_ids:
            is_denied = rng.random() < denial_rate
            if is_denied:
                adjudication_status = "denied"
                paid_amount = 0.0
                denial_reason = rng.choice(self._DENIAL_REASONS)
            else:
                adjudication_status = "paid"
                paid_amount = round(rng.uniform(50.0, 8000.0), 2)
                denial_reason = None

            claim_records.append({
                "claim_id": claim_id,
                "adjudication_status": adjudication_status,
                "paid_amount": paid_amount,
                "denial_reason": denial_reason,
            })

        return {
            "transaction_date": transaction_date,
            "payer_id": payer_id,
            "claims": claim_records,
        }

    def generate_labeled_claim_dataset(
        self,
        n_records: int,
        denial_rate: float = 0.20,
        date_range_days: int = 90,
        seed: Optional[int] = None,
    ) -> List[dict]:
        """
        Produce a temporally ordered list of claim records with known
        adjudication outcomes, suitable for passing directly to
        ``TrainingDatasetBuilder.build()``.

        Each record in the returned list contains:

        - ``claim_id``          — unique synthetic claim identifier
        - ``submission_date``   — ISO-8601 date within the specified window
        - ``billing_provider_npi`` — synthetic 10-digit NPI
        - ``payer_id``          — synthetic payer identifier
        - ``adjudication_outcome`` — ``"denied"`` or ``"paid"``

        Records are sorted by ``submission_date`` (ascending).

        The ``denial_rate`` is configurable to support the <5 % class-imbalance
        test scenarios described in Requirement 8.5.

        Parameters
        ----------
        n_records:
            Number of claim records to generate.  Must be ≥ 1.
        denial_rate:
            Probability in [0.0, 1.0] that any given claim is denied.
            Default is 0.20.
        date_range_days:
            Width of the submission-date window in calendar days.  The
            window ends at today and extends ``date_range_days`` days
            into the past.  Default is 90 (Requirement 8.1).
        seed:
            Optional integer seed for reproducible output.

        Returns
        -------
        List[dict]
            Claim records sorted ascending by ``submission_date``.

        Raises
        ------
        ValueError
            If ``denial_rate`` is not in [0.0, 1.0] or ``n_records`` < 1.
        """
        if not 0.0 <= denial_rate <= 1.0:
            raise ValueError(
                f"denial_rate must be in [0.0, 1.0]; got {denial_rate!r}"
            )
        if n_records < 1:
            raise ValueError(f"n_records must be ≥ 1; got {n_records!r}")

        rng = random.Random(seed)

        # Date window: from (today - date_range_days) to today inclusive
        end_date = date.today()
        start_date = end_date - timedelta(days=max(0, date_range_days - 1))

        # Pre-generate a small pool of synthetic NPIs and payer IDs to produce
        # realistic provider / payer diversity (≈10 % of n_records, min 5, max 50)
        n_providers = max(5, min(50, n_records // 10))
        n_payers = max(3, min(20, n_records // 20))

        provider_npis = [
            "".join(rng.choices(string.digits, k=10))
            for _ in range(n_providers)
        ]
        payer_ids = [
            "PAY" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
            for _ in range(n_payers)
        ]

        records: List[dict] = []
        total_days = (end_date - start_date).days  # inclusive range: 0..total_days

        for i in range(n_records):
            claim_id = "CLM-" + "".join(
                rng.choices(string.ascii_uppercase + string.digits, k=10)
            )
            # Random date within the window
            offset = rng.randint(0, total_days)
            submission_date = (start_date + timedelta(days=offset)).isoformat()

            npi = rng.choice(provider_npis)
            payer = rng.choice(payer_ids)

            # Adjudication outcome (Requirement 8.2: "denied"=1, "paid"=0)
            outcome = "denied" if rng.random() < denial_rate else "paid"

            records.append({
                "claim_id": claim_id,
                "submission_date": submission_date,
                "billing_provider_npi": npi,
                "payer_id": payer,
                "adjudication_outcome": outcome,
            })

        # Sort by submission_date ascending (Requirement 8.3 temporal ordering)
        records.sort(key=lambda r: r["submission_date"])

        return records

    def generate_provider_window(
        self,
        npi: str,
        n_claims: int,
        denial_rate: float = 0.20,
        window_days: int = 90,
        seed: Optional[int] = None,
    ) -> List[dict]:
        """
        Produce a window of historical claims for a single NPI with
        configurable volume and denial rate.

        This method is specifically designed to test the LOO-encoding /
        population-mean boundary at exactly 30 and 31 claims (Requirements
        4.1 and 4.2): callers can pass ``n_claims=30`` to exercise the
        population-mean fallback path and ``n_claims=31`` to exercise the
        leave-one-out encoding path.

        Each record contains:

        - ``claim_id``              — unique synthetic claim identifier
        - ``submission_date``       — ISO-8601 date within the window
        - ``billing_provider_npi``  — the supplied ``npi``
        - ``payer_id``              — synthetic payer identifier
        - ``adjudication_outcome``  — ``"denied"`` or ``"paid"``

        Records are sorted by ``submission_date`` ascending.

        Parameters
        ----------
        npi:
            The billing provider NPI to embed in all generated records.
            Must be a 10-digit numeric string.
        n_claims:
            Number of claims to generate for this provider.  Must be ≥ 1.
        denial_rate:
            Probability in [0.0, 1.0] that any given claim is denied.
            Default is 0.20.
        window_days:
            Width of the trailing window in calendar days.  Default is 90
            (Requirement 8.1).
        seed:
            Optional integer seed for reproducible output.

        Returns
        -------
        List[dict]
            Claim records for the single NPI, sorted ascending by
            ``submission_date``.

        Raises
        ------
        ValueError
            If ``denial_rate`` is not in [0.0, 1.0], ``n_claims`` < 1, or
            ``npi`` is not a 10-digit numeric string.
        """
        if not 0.0 <= denial_rate <= 1.0:
            raise ValueError(
                f"denial_rate must be in [0.0, 1.0]; got {denial_rate!r}"
            )
        if n_claims < 1:
            raise ValueError(f"n_claims must be ≥ 1; got {n_claims!r}")
        if not npi.isdigit() or len(npi) != 10:
            raise ValueError(
                f"npi must be a 10-digit numeric string; got {npi!r}"
            )

        rng = random.Random(seed)

        # Date window: (today - window_days) to today
        end_date = date.today()
        start_date = end_date - timedelta(days=max(0, window_days - 1))
        total_days = (end_date - start_date).days

        # Small pool of payer IDs for this provider's history
        n_payers = max(2, min(5, n_claims // 5))
        payer_ids = [
            "PAY" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
            for _ in range(n_payers)
        ]

        records: List[dict] = []
        for i in range(n_claims):
            claim_id = "CLM-" + "".join(
                rng.choices(string.ascii_uppercase + string.digits, k=10)
            )
            offset = rng.randint(0, total_days)
            submission_date = (start_date + timedelta(days=offset)).isoformat()
            payer = rng.choice(payer_ids)
            outcome = "denied" if rng.random() < denial_rate else "paid"

            records.append({
                "claim_id": claim_id,
                "submission_date": submission_date,
                "billing_provider_npi": npi,
                "payer_id": payer,
                "adjudication_outcome": outcome,
            })

        # Sort by submission_date ascending
        records.sort(key=lambda r: r["submission_date"])

        return records
