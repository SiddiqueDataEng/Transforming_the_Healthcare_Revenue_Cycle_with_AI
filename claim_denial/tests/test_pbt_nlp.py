# Feature: claim-denial-prediction, Property 12: NLP Defaults Are Applied Consistently on Absent or Failed Notes
"""
Property-based tests for NLPProcessor default-value behaviour.

**Validates: Requirements 7.4, 7.5**

Property 12: NLP Defaults Are Applied Consistently on Absent or Failed Notes
  For any claim with no clinical notes available OR any claim where NLP
  inference fails, the resulting NLP features SHALL satisfy:
    * documents_medical_necessity = 0.5
    * mentions_lack_of_pre_auth   = 0
    * tx_plan_complexity_embedding is a zero float32 vector of exactly 768 dims
    * nlp_notes_absent_flag ∈ {1, 2}
"""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from claim_denial.nlp.nlp_processor import (
    EMBEDDING_DIM,
    FLAG_INFERENCE_ERROR,
    FLAG_NOTES_ABSENT,
    ClinicalNote,
    NLPFeatures,
    NLPProcessor,
)

# ---------------------------------------------------------------------------
# Helper: a model that always raises RuntimeError to simulate inference failure
# ---------------------------------------------------------------------------


class _AlwaysFailModel:
    """Stub model whose encode() always raises, triggering flag=2 defaults."""

    model_name: str = "always-fail-model"
    model_version: str = "0.0.0-test"

    def encode(self, text: str) -> np.ndarray:  # noqa: ARG002
        raise RuntimeError("Simulated NLP inference failure")


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_claim_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=40,
)


def _assert_defaults(features: NLPFeatures, expected_flag: int) -> None:
    """Shared assertion helper for all default-value checks."""
    assert features.medical_necessity == 0.5, (
        f"Expected medical_necessity=0.5, got {features.medical_necessity}"
    )
    assert features.pre_auth == 0, (
        f"Expected pre_auth=0, got {features.pre_auth}"
    )
    assert features.embedding.dtype == np.float32, (
        f"Expected float32 embedding, got {features.embedding.dtype}"
    )
    assert features.embedding.shape == (EMBEDDING_DIM,), (
        f"Expected embedding shape ({EMBEDDING_DIM},), got {features.embedding.shape}"
    )
    assert np.all(features.embedding == 0.0), (
        "Expected all-zero embedding, got non-zero values"
    )
    assert features.flag == expected_flag, (
        f"Expected flag={expected_flag}, got {features.flag}"
    )
    assert features.flag in {FLAG_NOTES_ABSENT, FLAG_INFERENCE_ERROR}, (
        f"flag {features.flag} not in {{1, 2}}"
    )


# ---------------------------------------------------------------------------
# Property 12a — absent notes (None) → flag = 1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(claim_id=_claim_id_strategy)
def test_none_notes_returns_defaults_flag1(claim_id: str) -> None:
    """**Validates: Requirements 7.4**

    When clinical_notes is None, process_claim SHALL return default NLP
    features with flag=1 (notes absent) for any claim ID.
    """
    processor = NLPProcessor()
    features = processor.process_claim(claim_id, None)
    _assert_defaults(features, FLAG_NOTES_ABSENT)


# ---------------------------------------------------------------------------
# Property 12b — empty list of notes → flag = 1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(claim_id=_claim_id_strategy)
def test_empty_notes_list_returns_defaults_flag1(claim_id: str) -> None:
    """**Validates: Requirements 7.4**

    When clinical_notes is an empty list, process_claim SHALL return default
    NLP features with flag=1 (notes absent) for any claim ID.
    """
    processor = NLPProcessor()
    features = processor.process_claim(claim_id, [])
    _assert_defaults(features, FLAG_NOTES_ABSENT)


# ---------------------------------------------------------------------------
# Property 12c — inference failure → flag = 2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(claim_id=_claim_id_strategy)
def test_inference_failure_returns_defaults_flag2(claim_id: str) -> None:
    """**Validates: Requirements 7.5**

    When the NLP model raises during inference, process_claim SHALL return
    default NLP features with flag=2 (inference error) for any claim ID.
    """
    from datetime import date

    processor = NLPProcessor(model=_AlwaysFailModel())

    # Provide a non-empty note list so the absent-notes path is NOT triggered
    notes = [
        ClinicalNote(
            note_type="Progress Note",
            note_text="Patient presents with chest pain.",
            note_date=date(2024, 1, 15),
        )
    ]
    features = processor.process_claim(claim_id, notes)
    _assert_defaults(features, FLAG_INFERENCE_ERROR)


# ---------------------------------------------------------------------------
# Property 12d — apply_defaults returns correct values for both flag values
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    claim_id=_claim_id_strategy,
    flag=st.sampled_from([FLAG_NOTES_ABSENT, FLAG_INFERENCE_ERROR]),
)
def test_apply_defaults_always_returns_valid_defaults(claim_id: str, flag: int) -> None:
    """**Validates: Requirements 7.4, 7.5**

    apply_defaults(claim_id, flag) SHALL always return NLPFeatures with
    medical_necessity=0.5, pre_auth=0, zero float32 768-dim embedding, and
    the exact flag value requested, for any valid claim_id and flag ∈ {1, 2}.
    """
    processor = NLPProcessor()
    features = processor.apply_defaults(claim_id, flag)
    _assert_defaults(features, flag)
