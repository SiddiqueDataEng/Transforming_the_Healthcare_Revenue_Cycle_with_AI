"""
Unit tests for ClaimLevelFeatureEngineer.

Coverage:
  - Temporal features (DOY / month): Dec 31, Feb 29 leap year, Jan 1
  - Patient age: same-day birthday, DOB in future, null DOB
  - One-hot encoding: each category for insurance_type, claim_type,
    admission_type, admission_source, plus the all-zeros fallback for
    unrecognized / absent values

Requirements: 3.1, 3.2, 3.3, 3.9
"""

from __future__ import annotations

from datetime import date

import pytest

from claim_denial.constants import (
    ADMISSION_SOURCES,
    ADMISSION_TYPES,
    CLAIM_TYPES,
    INSURANCE_TYPES,
    SENTINEL_AGE_UNKNOWN,
)
from claim_denial.features.claim_level import ClaimLevelFeatureEngineer
from claim_denial.models import ClaimRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(
    *,
    statement_period_start: date | None = date(2024, 6, 15),
    date_of_birth: date | None = date(1980, 3, 10),
    gender: str | None = "M",
    insurance_type: str | None = None,
    claim_type: str | None = None,
    admission_type: str | None = None,
    admission_source: str | None = None,
) -> ClaimRecord:
    """Return a minimal ClaimRecord with only the fields needed for claim-level FE."""
    return ClaimRecord(
        claim_id="CLM-001",
        provider_npi="1234567890",
        payer_id="PAYER-01",
        patient_account_number="PAT-001",
        statement_period_start=statement_period_start,
        date_of_birth=date_of_birth,
        gender=gender,
        insurance_type=insurance_type,
        claim_type=claim_type,
        admission_type=admission_type,
        admission_source=admission_source,
    )


@pytest.fixture
def engineer() -> ClaimLevelFeatureEngineer:
    return ClaimLevelFeatureEngineer()


# ===========================================================================
# Requirement 3.1 — Temporal features: DOY and month
# ===========================================================================

class TestTemporalFeatures:
    """DOY and month are computed correctly for boundary dates."""

    def test_jan_1_doy_and_month(self, engineer):
        """Jan 1 is DOY=1, month=1."""
        claim = _make_claim(statement_period_start=date(2024, 1, 1))
        rec = engineer.compute_and_store(claim)
        assert rec.claim_submission_date_doy == 1
        assert rec.claim_submission_date_month == 1

    def test_dec_31_non_leap_year(self, engineer):
        """Dec 31 in a non-leap year is DOY=365, month=12."""
        claim = _make_claim(statement_period_start=date(2023, 12, 31))
        rec = engineer.compute_and_store(claim)
        assert rec.claim_submission_date_doy == 365
        assert rec.claim_submission_date_month == 12

    def test_dec_31_leap_year(self, engineer):
        """Dec 31 in a leap year is DOY=366, month=12."""
        claim = _make_claim(statement_period_start=date(2024, 12, 31))
        rec = engineer.compute_and_store(claim)
        assert rec.claim_submission_date_doy == 366
        assert rec.claim_submission_date_month == 12

    def test_feb_29_leap_year(self, engineer):
        """Feb 29 exists only in a leap year: DOY=60, month=2."""
        claim = _make_claim(statement_period_start=date(2024, 2, 29))
        rec = engineer.compute_and_store(claim)
        assert rec.claim_submission_date_doy == 60
        assert rec.claim_submission_date_month == 2

    def test_none_submission_date_returns_sentinels(self, engineer):
        """Absent submission date → DOY=0, month=0."""
        claim = _make_claim(statement_period_start=None)
        rec = engineer.compute_and_store(claim)
        assert rec.claim_submission_date_doy == 0
        assert rec.claim_submission_date_month == 0

    def test_mid_year_date_doy(self, engineer):
        """Spot-check a mid-year date: 2024-07-04 is the 186th day."""
        claim = _make_claim(statement_period_start=date(2024, 7, 4))
        rec = engineer.compute_and_store(claim)
        assert rec.claim_submission_date_doy == 186
        assert rec.claim_submission_date_month == 7


# ===========================================================================
# Requirement 3.2 / 3.9 — Patient age
# ===========================================================================

class TestPatientAge:
    """Age is computed as whole years; sentinel −1 for absent or invalid DOB."""

    def test_normal_age_calculation(self, engineer):
        """Patient born 1980-03-10; claim submitted 2024-06-15 → age 44."""
        claim = _make_claim(
            date_of_birth=date(1980, 3, 10),
            statement_period_start=date(2024, 6, 15),
        )
        rec = engineer.compute_and_store(claim)
        assert rec.patient_age == 44

    def test_birthday_not_yet_occurred_this_year(self, engineer):
        """DOB 1980-09-01; claim date 2024-06-15: birthday hasn't occurred → age 43."""
        claim = _make_claim(
            date_of_birth=date(1980, 9, 1),
            statement_period_start=date(2024, 6, 15),
        )
        rec = engineer.compute_and_store(claim)
        assert rec.patient_age == 43

    def test_same_day_birthday(self, engineer):
        """Claim submitted exactly on the patient's birthday: birthday has occurred."""
        dob = date(1990, 5, 20)
        claim_date = date(2024, 5, 20)
        claim = _make_claim(date_of_birth=dob, statement_period_start=claim_date)
        rec = engineer.compute_and_store(claim)
        assert rec.patient_age == 34  # 2024 - 1990 = 34

    def test_dob_in_future_returns_negative_age(self, engineer):
        """DOB in the future: age calculation yields a negative integer, not −1 sentinel.

        The implementation computes year difference and adjusts if birthday not yet
        passed. A DOB strictly in the future (same year, later month) still returns
        (ref_year - dob_year) adjusted by -1 if birthday hasn't occurred.
        This edge case: we just verify no exception is raised and the value is ≤ 0.
        """
        claim = _make_claim(
            date_of_birth=date(2030, 1, 1),
            statement_period_start=date(2024, 6, 15),
        )
        rec = engineer.compute_and_store(claim)
        # 2024 - 2030 = -6; birthday hasn't occurred yet in 2024 → -6 - 1 = -7
        assert rec.patient_age < 0

    def test_null_dob_returns_sentinel(self, engineer):
        """Absent DOB → SENTINEL_AGE_UNKNOWN (−1). Requirement 3.9."""
        claim = _make_claim(
            date_of_birth=None,
            statement_period_start=date(2024, 6, 15),
        )
        rec = engineer.compute_and_store(claim)
        assert rec.patient_age == SENTINEL_AGE_UNKNOWN

    def test_null_submission_date_returns_sentinel(self, engineer):
        """Absent submission date → SENTINEL_AGE_UNKNOWN because reference date is needed."""
        claim = _make_claim(
            date_of_birth=date(1980, 3, 10),
            statement_period_start=None,
        )
        rec = engineer.compute_and_store(claim)
        assert rec.patient_age == SENTINEL_AGE_UNKNOWN

    def test_both_dates_null_returns_sentinel(self, engineer):
        """Both DOB and submission date absent → SENTINEL_AGE_UNKNOWN."""
        claim = _make_claim(date_of_birth=None, statement_period_start=None)
        rec = engineer.compute_and_store(claim)
        assert rec.patient_age == SENTINEL_AGE_UNKNOWN

    def test_leap_year_dob_feb_29_non_leap_reference(self, engineer):
        """DOB on Feb 29 (leap year); reference date is Mar 1 of a non-leap year.
        Birthday is treated as Feb 28/Mar 1 — age increments by reference date.
        2000-02-29 DOB; reference 2023-03-01.
        (2023 - 2000) = 23; (3,1) >= (2,29) in Python tuple comparison → birthday passed → age 23.
        """
        claim = _make_claim(
            date_of_birth=date(2000, 2, 29),
            statement_period_start=date(2023, 3, 1),
        )
        rec = engineer.compute_and_store(claim)
        assert rec.patient_age == 23

    def test_leap_year_dob_feb_29_before_birthday_in_non_leap_year(self, engineer):
        """DOB on Feb 29; reference date is Feb 28 of a non-leap year → birthday not yet reached."""
        claim = _make_claim(
            date_of_birth=date(2000, 2, 29),
            statement_period_start=date(2023, 2, 28),
        )
        rec = engineer.compute_and_store(claim)
        # (2,28) < (2,29) → subtract 1 → 22
        assert rec.patient_age == 22


# ===========================================================================
# Requirement 3.3 — Gender encoding
# ===========================================================================

class TestGenderEncoding:
    """Female=0, Male=1, unknown/absent=−1."""

    @pytest.mark.parametrize("gender_val", ["M", "m", " M ", "male"])
    def test_male_variants(self, engineer, gender_val):
        """Various casings/spellings that normalise to 'M' → 1."""
        # Note: the implementation only recognises stripped upper-case 'M'.
        # 'male' (not 'M') will fall through to sentinel.
        claim = _make_claim(gender=gender_val)
        rec = engineer.compute_and_store(claim)
        if gender_val.strip().upper() == "M":
            assert rec.gender == 1
        else:
            assert rec.gender == -1

    @pytest.mark.parametrize("gender_val", ["F", "f", " F "])
    def test_female_variants(self, engineer, gender_val):
        claim = _make_claim(gender=gender_val)
        rec = engineer.compute_and_store(claim)
        assert rec.gender == 0

    @pytest.mark.parametrize("gender_val", [None, "U", "unknown", "", " "])
    def test_unknown_and_null_gender(self, engineer, gender_val):
        claim = _make_claim(gender=gender_val)
        rec = engineer.compute_and_store(claim)
        assert rec.gender == -1


# ===========================================================================
# Requirement 3.5 — Insurance type one-hot encoding
# ===========================================================================

class TestInsuranceTypeOneHot:
    """Each recognized category sets exactly its own flag; unrecognized → all zeros."""

    CATEGORY_TO_FIELD = {
        "Medicare":   "ins_type_medicare",
        "Medicaid":   "ins_type_medicaid",
        "Commercial": "ins_type_commercial",
        "TriCare":    "ins_type_tricare",
        "ChampVA":    "ins_type_champva",
        "Other":      "ins_type_other",
    }

    def _all_ins_fields(self, rec) -> dict[str, int]:
        return {
            "ins_type_medicare":   rec.ins_type_medicare,
            "ins_type_medicaid":   rec.ins_type_medicaid,
            "ins_type_commercial": rec.ins_type_commercial,
            "ins_type_tricare":    rec.ins_type_tricare,
            "ins_type_champva":    rec.ins_type_champva,
            "ins_type_other":      rec.ins_type_other,
        }

    @pytest.mark.parametrize("category", INSURANCE_TYPES)
    def test_each_category_sets_exactly_one_flag(self, engineer, category):
        claim = _make_claim(insurance_type=category)
        rec = engineer.compute_and_store(claim)
        fields = self._all_ins_fields(rec)
        expected_field = self.CATEGORY_TO_FIELD[category]
        assert fields[expected_field] == 1, f"Expected {expected_field}=1 for '{category}'"
        for fname, val in fields.items():
            if fname != expected_field:
                assert val == 0, f"Expected {fname}=0 when '{category}' is set"

    @pytest.mark.parametrize("category", INSURANCE_TYPES)
    def test_case_insensitive_match(self, engineer, category):
        """Lowercase input should still map to the correct column."""
        claim = _make_claim(insurance_type=category.lower())
        rec = engineer.compute_and_store(claim)
        fields = self._all_ins_fields(rec)
        expected_field = self.CATEGORY_TO_FIELD[category]
        assert fields[expected_field] == 1

    @pytest.mark.parametrize("bad_val", [None, "Unknown", "PPO", "HMO", "BlueCross", ""])
    def test_unrecognized_or_absent_is_all_zeros(self, engineer, bad_val):
        """Unrecognized / absent insurance type → all six columns = 0."""
        claim = _make_claim(insurance_type=bad_val)
        rec = engineer.compute_and_store(claim)
        fields = self._all_ins_fields(rec)
        for fname, val in fields.items():
            assert val == 0, f"Expected {fname}=0 for unrecognized value '{bad_val}'"


# ===========================================================================
# Requirement 3.6 — Claim type one-hot encoding
# ===========================================================================

class TestClaimTypeOneHot:
    """Professional / Institutional encoding."""

    def _fields(self, rec) -> dict[str, int]:
        return {
            "claim_type_professional":  rec.claim_type_professional,
            "claim_type_institutional": rec.claim_type_institutional,
        }

    @pytest.mark.parametrize("category, expected_field", [
        ("Professional",  "claim_type_professional"),
        ("Institutional", "claim_type_institutional"),
    ])
    def test_each_category(self, engineer, category, expected_field):
        claim = _make_claim(claim_type=category)
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        assert fields[expected_field] == 1
        for fname, val in fields.items():
            if fname != expected_field:
                assert val == 0

    @pytest.mark.parametrize("category", CLAIM_TYPES)
    def test_case_insensitive(self, engineer, category):
        claim = _make_claim(claim_type=category.upper())
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        # Sum should still be 1 for recognised category
        assert sum(fields.values()) == 1

    @pytest.mark.parametrize("bad_val", [None, "outpatient", "inpatient", ""])
    def test_unrecognized_is_all_zeros(self, engineer, bad_val):
        claim = _make_claim(claim_type=bad_val)
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        assert sum(fields.values()) == 0


# ===========================================================================
# Requirement 3.7 — Admission type one-hot encoding
# ===========================================================================

class TestAdmissionTypeOneHot:
    """Emergency / Elective / Urgent / Trauma encoding."""

    CATEGORY_TO_FIELD = {
        "Emergency": "adm_type_emergency",
        "Elective":  "adm_type_elective",
        "Urgent":    "adm_type_urgent",
        "Trauma":    "adm_type_trauma",
    }

    def _fields(self, rec) -> dict[str, int]:
        return {
            "adm_type_emergency": rec.adm_type_emergency,
            "adm_type_elective":  rec.adm_type_elective,
            "adm_type_urgent":    rec.adm_type_urgent,
            "adm_type_trauma":    rec.adm_type_trauma,
        }

    @pytest.mark.parametrize("category", ADMISSION_TYPES)
    def test_each_category(self, engineer, category):
        claim = _make_claim(admission_type=category)
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        expected_field = self.CATEGORY_TO_FIELD[category]
        assert fields[expected_field] == 1
        for fname, val in fields.items():
            if fname != expected_field:
                assert val == 0

    @pytest.mark.parametrize("category", ADMISSION_TYPES)
    def test_case_insensitive(self, engineer, category):
        claim = _make_claim(admission_type=category.lower())
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        assert sum(fields.values()) == 1

    @pytest.mark.parametrize("bad_val", [None, "scheduled", "walk-in", "unknown", ""])
    def test_unrecognized_or_absent_is_all_zeros(self, engineer, bad_val):
        claim = _make_claim(admission_type=bad_val)
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        assert sum(fields.values()) == 0, (
            f"Expected all zeros for admission_type='{bad_val}'"
        )


# ===========================================================================
# Requirement 3.8 — Admission source one-hot encoding
# ===========================================================================

class TestAdmissionSourceOneHot:
    """Six admission source categories; unrecognized → all zeros."""

    CATEGORY_TO_FIELD = {
        "Physician Referral":     "adm_src_physician_referral",
        "Transfer from Hospital": "adm_src_transfer_hospital",
        "Transfer from SNF":      "adm_src_transfer_snf",
        "Emergency Room":         "adm_src_er",
        "Court/Law Enforcement":  "adm_src_court_law",
        "Not Available":          "adm_src_not_available",
    }

    def _fields(self, rec) -> dict[str, int]:
        return {
            "adm_src_physician_referral": rec.adm_src_physician_referral,
            "adm_src_transfer_hospital":  rec.adm_src_transfer_hospital,
            "adm_src_transfer_snf":       rec.adm_src_transfer_snf,
            "adm_src_er":                 rec.adm_src_er,
            "adm_src_court_law":          rec.adm_src_court_law,
            "adm_src_not_available":      rec.adm_src_not_available,
        }

    @pytest.mark.parametrize("category", ADMISSION_SOURCES)
    def test_each_category(self, engineer, category):
        claim = _make_claim(admission_source=category)
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        expected_field = self.CATEGORY_TO_FIELD[category]
        assert fields[expected_field] == 1
        for fname, val in fields.items():
            if fname != expected_field:
                assert val == 0

    @pytest.mark.parametrize("category", ADMISSION_SOURCES)
    def test_case_insensitive(self, engineer, category):
        claim = _make_claim(admission_source=category.upper())
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        assert sum(fields.values()) == 1

    @pytest.mark.parametrize("bad_val", [None, "self-referral", "ambulance", "UNKNOWN", ""])
    def test_unrecognized_or_absent_is_all_zeros(self, engineer, bad_val):
        claim = _make_claim(admission_source=bad_val)
        rec = engineer.compute_and_store(claim)
        fields = self._fields(rec)
        assert sum(fields.values()) == 0, (
            f"Expected all zeros for admission_source='{bad_val}'"
        )


# ===========================================================================
# Requirement 3.10 — Feature Store upsert
# ===========================================================================

class TestFeatureStoreUpsert:
    """compute_and_store writes / merges rows keyed by claim_id."""

    def test_first_write_creates_row(self):
        store: dict = {}
        eng = ClaimLevelFeatureEngineer(feature_store=store)
        claim = _make_claim()
        eng.compute_and_store(claim)
        assert "CLM-001" in store

    def test_second_write_overwrites_claim_level_keys(self):
        """Upsert for the same claim_id updates existing values."""
        store: dict = {}
        eng = ClaimLevelFeatureEngineer(feature_store=store)

        claim_v1 = _make_claim(
            statement_period_start=date(2024, 1, 15),
            insurance_type="Medicare",
        )
        eng.compute_and_store(claim_v1)

        claim_v2 = _make_claim(
            statement_period_start=date(2024, 3, 20),
            insurance_type="Medicaid",
        )
        # Use same claim_id
        eng.compute_and_store(claim_v2)

        row = store["CLM-001"]
        assert row["claim_submission_date_month"] == 3
        assert row["ins_type_medicare"] == 0
        assert row["ins_type_medicaid"] == 1

    def test_upsert_preserves_non_claim_level_keys(self):
        """Pre-existing keys not written by this engineer must survive the upsert."""
        store: dict = {"CLM-001": {"provider_historical_denial_rate": 0.25}}
        eng = ClaimLevelFeatureEngineer(feature_store=store)
        eng.compute_and_store(_make_claim())
        assert store["CLM-001"]["provider_historical_denial_rate"] == 0.25
