"""
Unit tests for ClinicalFeatureEngineer.

**Validates: Requirements 5.1, 5.3, 5.5**

Covers:
- CCS mapping: known code → correct category, unknown code → 0
- Procedure multi-hot vector: empty codes → all zeros, unrecognized code → no
  change, recognized code → correct bit set
- Charlson Comorbidity Index: codes from mapping table accumulate correct
  score, codes not in table contribute 0, empty list → 0
"""

from __future__ import annotations

from datetime import date

import pytest

from claim_denial.constants import (
    HCUP_CCS_LOOKUP,
    HCUP_CCS_UNKNOWN,
    HCUP_PROC_LOOKUP,
)
from claim_denial.features.clinical import (
    SAMPLE_CHARLSON_MAPPING,
    SAMPLE_CCS_LOOKUP,
    SAMPLE_PROC_LOOKUP,
    SAMPLE_PROC_VECTOR_LENGTH,
    ClinicalFeatureEngineer,
    build_procedure_multihot_vector,
    compute_charlson_comorbidity_index,
    compute_length_of_stay,
    map_principal_dx_to_ccs,
)


# ─────────────────────────────────────────────────────────────────────────────
# CCS Mapping — Requirements 5.1
# ─────────────────────────────────────────────────────────────────────────────


class TestMapPrincipalDxToCcs:
    """Tests for ``map_principal_dx_to_ccs``."""

    def test_known_code_returns_correct_category(self) -> None:
        """A code present in the lookup returns its mapped category."""
        # I200 → CCS 100 (Acute MI) per constants.py
        assert map_principal_dx_to_ccs("I200", SAMPLE_CCS_LOOKUP) == 100

    def test_known_code_case_insensitive(self) -> None:
        """Lookup is case-insensitive — lower-case input still resolves."""
        assert map_principal_dx_to_ccs("i200", SAMPLE_CCS_LOOKUP) == 100

    def test_known_code_with_dot_stripped(self) -> None:
        """Dots in the ICD-10 code are stripped before lookup."""
        # "I20.0" normalises to "I200" → category 100
        assert map_principal_dx_to_ccs("I20.0", SAMPLE_CCS_LOOKUP) == 100

    def test_known_code_with_leading_trailing_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped before lookup."""
        assert map_principal_dx_to_ccs("  I200  ", SAMPLE_CCS_LOOKUP) == 100

    def test_unknown_code_returns_zero(self) -> None:
        """A code absent from the lookup returns the unknown sentinel (0)."""
        assert map_principal_dx_to_ccs("Z99999", SAMPLE_CCS_LOOKUP) == HCUP_CCS_UNKNOWN

    def test_none_returns_zero(self) -> None:
        """None input returns the unknown sentinel."""
        assert map_principal_dx_to_ccs(None, SAMPLE_CCS_LOOKUP) == HCUP_CCS_UNKNOWN

    def test_empty_string_returns_zero(self) -> None:
        """Empty string returns the unknown sentinel."""
        assert map_principal_dx_to_ccs("", SAMPLE_CCS_LOOKUP) == HCUP_CCS_UNKNOWN

    def test_multiple_known_codes_map_correctly(self) -> None:
        """Several known codes each map to their correct category."""
        # K400 → CCS 142 (Hernia)
        assert map_principal_dx_to_ccs("K400", SAMPLE_CCS_LOOKUP) == 142
        # J440 → CCS 127 (COPD)
        assert map_principal_dx_to_ccs("J440", SAMPLE_CCS_LOOKUP) == 127
        # I500 → CCS 108 (Congestive heart failure)
        assert map_principal_dx_to_ccs("I500", SAMPLE_CCS_LOOKUP) == 108

    def test_empty_lookup_always_returns_unknown(self) -> None:
        """With an empty lookup, any code returns the unknown sentinel."""
        assert map_principal_dx_to_ccs("I200", {}) == HCUP_CCS_UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Procedure Multi-Hot Vector — Requirements 5.3
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildProcedureMultihotVector:
    """Tests for ``build_procedure_multihot_vector``."""

    def test_empty_codes_returns_all_zeros(self) -> None:
        """An empty code list produces a zero vector of the correct length."""
        vec = build_procedure_multihot_vector([], SAMPLE_PROC_LOOKUP, SAMPLE_PROC_VECTOR_LENGTH)
        assert len(vec) == SAMPLE_PROC_VECTOR_LENGTH
        assert all(v == 0 for v in vec)

    def test_none_codes_returns_all_zeros(self) -> None:
        """A None code list produces a zero vector of the correct length."""
        vec = build_procedure_multihot_vector(None, SAMPLE_PROC_LOOKUP, SAMPLE_PROC_VECTOR_LENGTH)
        assert len(vec) == SAMPLE_PROC_VECTOR_LENGTH
        assert all(v == 0 for v in vec)

    def test_unrecognized_code_produces_no_change(self) -> None:
        """An unrecognized CPT code leaves the vector unchanged (all zeros)."""
        vec = build_procedure_multihot_vector(
            ["99999UNKNOWN"],
            SAMPLE_PROC_LOOKUP,
            SAMPLE_PROC_VECTOR_LENGTH,
        )
        assert all(v == 0 for v in vec), (
            f"Unrecognized code should not set any bit, got {vec}"
        )

    def test_recognized_code_sets_correct_bit(self) -> None:
        """A recognized CPT code sets the corresponding bit."""
        # "93000" → index 3 (Electrocardiogram) per HCUP_PROC_LOOKUP
        vec = build_procedure_multihot_vector(
            ["93000"],
            SAMPLE_PROC_LOOKUP,
            SAMPLE_PROC_VECTOR_LENGTH,
        )
        assert vec[3] == 1, f"Expected bit 3 set for code '93000', got {vec[3]}"
        # All other bits should be 0
        for i, v in enumerate(vec):
            if i != 3:
                assert v == 0, f"Unexpected bit {i} set to {v}"

    def test_multiple_recognized_codes_set_multiple_bits(self) -> None:
        """Multiple recognized codes set their respective bits."""
        # "71046" → index 0 (Diagnostic radiology)
        # "27130" → index 8 (Hip replacement)
        vec = build_procedure_multihot_vector(
            ["71046", "27130"],
            SAMPLE_PROC_LOOKUP,
            SAMPLE_PROC_VECTOR_LENGTH,
        )
        assert vec[0] == 1, "Expected bit 0 set for code '71046'"
        assert vec[8] == 1, "Expected bit 8 set for code '27130'"

    def test_duplicate_codes_same_category_sets_bit_once(self) -> None:
        """Duplicate codes mapping to the same category set the bit to 1 (not >1)."""
        # "93000" and "93005" both map to index 3
        vec = build_procedure_multihot_vector(
            ["93000", "93005"],
            SAMPLE_PROC_LOOKUP,
            SAMPLE_PROC_VECTOR_LENGTH,
        )
        assert vec[3] == 1, f"Expected bit 3 = 1 (not >1), got {vec[3]}"

    def test_vector_length_equals_proc_vector_length(self) -> None:
        """Returned vector always has exactly SAMPLE_PROC_VECTOR_LENGTH elements."""
        for codes in [[], ["93000"], ["UNKNOWN"], ["93000", "27130", "UNKNOWN"]]:
            vec = build_procedure_multihot_vector(codes, SAMPLE_PROC_LOOKUP, SAMPLE_PROC_VECTOR_LENGTH)
            assert len(vec) == SAMPLE_PROC_VECTOR_LENGTH, (
                f"Expected length {SAMPLE_PROC_VECTOR_LENGTH}, got {len(vec)} for codes={codes}"
            )

    def test_all_elements_are_binary(self) -> None:
        """All elements of the vector are 0 or 1."""
        vec = build_procedure_multihot_vector(
            list(SAMPLE_PROC_LOOKUP.keys()),
            SAMPLE_PROC_LOOKUP,
            SAMPLE_PROC_VECTOR_LENGTH,
        )
        for i, v in enumerate(vec):
            assert v in (0, 1), f"Element {i} is {v}, expected 0 or 1"

    def test_code_lookup_is_case_insensitive(self) -> None:
        """Lower-case CPT code normalises to upper-case for lookup."""
        vec_upper = build_procedure_multihot_vector(["93000"], SAMPLE_PROC_LOOKUP, SAMPLE_PROC_VECTOR_LENGTH)
        vec_lower = build_procedure_multihot_vector(["93000"], SAMPLE_PROC_LOOKUP, SAMPLE_PROC_VECTOR_LENGTH)
        assert vec_upper == vec_lower


# ─────────────────────────────────────────────────────────────────────────────
# Charlson Comorbidity Index — Requirements 5.5
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeCharlsonComorbidityIndex:
    """Tests for ``compute_charlson_comorbidity_index``."""

    def test_empty_list_returns_zero(self) -> None:
        """An empty secondary code list returns a score of 0."""
        assert compute_charlson_comorbidity_index([], SAMPLE_CHARLSON_MAPPING) == 0

    def test_none_returns_zero(self) -> None:
        """None secondary codes return a score of 0."""
        assert compute_charlson_comorbidity_index(None, SAMPLE_CHARLSON_MAPPING) == 0

    def test_single_weight_one_code(self) -> None:
        """A single weight-1 code contributes exactly 1 to the score."""
        # I50 (Congestive heart failure) has weight 1
        assert compute_charlson_comorbidity_index(["I50"], SAMPLE_CHARLSON_MAPPING) == 1

    def test_single_weight_two_code(self) -> None:
        """A single weight-2 code contributes exactly 2 to the score."""
        # G81 (Hemiplegia) has weight 2
        assert compute_charlson_comorbidity_index(["G81"], SAMPLE_CHARLSON_MAPPING) == 2

    def test_single_weight_six_code(self) -> None:
        """A single weight-6 code contributes exactly 6 to the score."""
        # B20 (AIDS/HIV) has weight 6
        assert compute_charlson_comorbidity_index(["B20"], SAMPLE_CHARLSON_MAPPING) == 6

    def test_multiple_codes_accumulate_score(self) -> None:
        """Multiple codes from the mapping table accumulate their weights."""
        # I50 (weight 1) + G81 (weight 2) = 3
        score = compute_charlson_comorbidity_index(
            ["I50", "G81"], SAMPLE_CHARLSON_MAPPING
        )
        assert score == 3

    def test_code_not_in_table_contributes_zero(self) -> None:
        """A code absent from the mapping table contributes 0."""
        # Z99 is not in SAMPLE_CHARLSON_MAPPING
        assert compute_charlson_comorbidity_index(["Z99"], SAMPLE_CHARLSON_MAPPING) == 0

    def test_mix_of_known_and_unknown_codes(self) -> None:
        """Known codes accumulate; unknown codes contribute 0."""
        # I50 (weight 1) + Z99 (not in table, 0) = 1
        score = compute_charlson_comorbidity_index(
            ["I50", "Z99999"], SAMPLE_CHARLSON_MAPPING
        )
        assert score == 1

    def test_prefix_matching_works(self) -> None:
        """ICD-10 codes matched by prefix contribute the prefix weight."""
        # I21 (weight 1) — "I210" should match prefix "I21"
        score = compute_charlson_comorbidity_index(
            ["I210"], SAMPLE_CHARLSON_MAPPING
        )
        assert score == 1

    def test_no_double_counting_for_same_prefix(self) -> None:
        """Two codes mapping to the same Charlson prefix are counted only once."""
        # I21 and I22 are separate weight-1 entries — should both contribute
        score = compute_charlson_comorbidity_index(
            ["I21", "I22"], SAMPLE_CHARLSON_MAPPING
        )
        assert score == 2

    def test_score_is_non_negative(self) -> None:
        """The Charlson score is always >= 0 for any input."""
        for codes in [[], None, ["Z99"], ["I50"], ["B20", "C50"]]:
            score = compute_charlson_comorbidity_index(codes, SAMPLE_CHARLSON_MAPPING)
            assert score >= 0, f"Negative score {score} for codes={codes}"

    def test_score_is_integer(self) -> None:
        """The returned score is always an integer."""
        score = compute_charlson_comorbidity_index(["I50", "G81"], SAMPLE_CHARLSON_MAPPING)
        assert isinstance(score, int)

    def test_empty_mapping_table_returns_zero(self) -> None:
        """With an empty mapping table, all codes contribute 0."""
        assert compute_charlson_comorbidity_index(["I50", "G81", "B20"], {}) == 0


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalFeatureEngineer integration — Requirements 5.1, 5.3, 5.5
# ─────────────────────────────────────────────────────────────────────────────


class TestClinicalFeatureEngineer:
    """Integration tests for ``ClinicalFeatureEngineer.compute_and_upsert``."""

    def setup_method(self) -> None:
        self.engineer = ClinicalFeatureEngineer()

    def test_known_principal_dx_maps_to_correct_ccs(self) -> None:
        """A known principal diagnosis code produces the correct CCS category."""
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM001",
            principal_icd10="I200",  # → CCS 100
            secondary_icd10_codes=[],
            cpt_hcpcs_codes=[],
            admission_date=None,
            discharge_date=None,
        )
        assert result["principal_dx_ccs_category"] == 100

    def test_unknown_principal_dx_maps_to_zero(self) -> None:
        """An unknown principal diagnosis code produces CCS category 0."""
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM002",
            principal_icd10="Z99999",  # not in lookup
            secondary_icd10_codes=[],
            cpt_hcpcs_codes=[],
            admission_date=None,
            discharge_date=None,
        )
        assert result["principal_dx_ccs_category"] == HCUP_CCS_UNKNOWN

    def test_empty_proc_codes_produce_zero_vector(self) -> None:
        """Empty procedure codes produce an all-zeros vector of correct length."""
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM003",
            principal_icd10=None,
            secondary_icd10_codes=[],
            cpt_hcpcs_codes=[],
            admission_date=None,
            discharge_date=None,
        )
        vec = result["procedure_category_vector"]
        assert len(vec) == SAMPLE_PROC_VECTOR_LENGTH
        assert all(v == 0 for v in vec)

    def test_unrecognized_proc_code_leaves_vector_unchanged(self) -> None:
        """An unrecognized procedure code does not change the vector from zeros."""
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM004",
            principal_icd10=None,
            secondary_icd10_codes=[],
            cpt_hcpcs_codes=["99999UNKNOWN"],
            admission_date=None,
            discharge_date=None,
        )
        vec = result["procedure_category_vector"]
        assert all(v == 0 for v in vec)

    def test_recognized_proc_code_sets_correct_bit(self) -> None:
        """A recognized procedure code sets the correct bit in the vector."""
        # "93000" → index 3 (Electrocardiogram)
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM005",
            principal_icd10=None,
            secondary_icd10_codes=[],
            cpt_hcpcs_codes=["93000"],
            admission_date=None,
            discharge_date=None,
        )
        vec = result["procedure_category_vector"]
        assert vec[3] == 1

    def test_charlson_accumulates_for_known_codes(self) -> None:
        """Charlson score accumulates correctly for codes in the mapping table."""
        # I50 (weight 1) + B20 (weight 6) = 7
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM006",
            principal_icd10=None,
            secondary_icd10_codes=["I50", "B20"],
            cpt_hcpcs_codes=[],
            admission_date=None,
            discharge_date=None,
        )
        assert result["comorbidity_risk_score"] == 7

    def test_charlson_zero_for_unknown_codes(self) -> None:
        """Charlson score is 0 when no codes are in the mapping table."""
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM007",
            principal_icd10=None,
            secondary_icd10_codes=["Z99999", "X00000"],
            cpt_hcpcs_codes=[],
            admission_date=None,
            discharge_date=None,
        )
        assert result["comorbidity_risk_score"] == 0

    def test_charlson_zero_for_empty_secondary_codes(self) -> None:
        """Charlson score is 0 when there are no secondary codes."""
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM008",
            principal_icd10=None,
            secondary_icd10_codes=[],
            cpt_hcpcs_codes=[],
            admission_date=None,
            discharge_date=None,
        )
        assert result["comorbidity_risk_score"] == 0

    def test_charlson_zero_for_none_secondary_codes(self) -> None:
        """Charlson score is 0 when secondary codes list is None."""
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM009",
            principal_icd10=None,
            secondary_icd10_codes=None,
            cpt_hcpcs_codes=[],
            admission_date=None,
            discharge_date=None,
        )
        assert result["comorbidity_risk_score"] == 0

    def test_feature_store_upsert_writes_all_columns(self) -> None:
        """compute_and_upsert writes all six clinical columns to the Feature Store."""
        result = self.engineer.compute_and_upsert(
            claim_id="CLAIM010",
            principal_icd10="I200",
            secondary_icd10_codes=["I50"],
            cpt_hcpcs_codes=["93000"],
            admission_date=date(2024, 1, 1),
            discharge_date=date(2024, 1, 5),
        )
        expected_keys = {
            "principal_dx_ccs_category",
            "secondary_dx_count",
            "procedure_category_vector",
            "length_of_stay",
            "comorbidity_risk_score",
            "diagnosis_procedure_mismatch",
        }
        assert expected_keys == set(result.keys())

    def test_feature_store_row_accessible_after_upsert(self) -> None:
        """Feature Store row is populated after compute_and_upsert."""
        self.engineer.compute_and_upsert(
            claim_id="CLAIM011",
            principal_icd10="I200",
            secondary_icd10_codes=["G81"],
            cpt_hcpcs_codes=["93460"],
            admission_date=date(2024, 3, 10),
            discharge_date=date(2024, 3, 15),
        )
        row = self.engineer.get_feature_store_row("CLAIM011")
        assert row is not None
        assert row["principal_dx_ccs_category"] == 100
        assert row["comorbidity_risk_score"] == 2  # G81 weight 2
        assert row["length_of_stay"] == 5
