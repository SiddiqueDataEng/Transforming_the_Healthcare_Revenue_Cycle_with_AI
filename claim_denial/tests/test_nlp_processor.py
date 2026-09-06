"""
Unit tests for NLPProcessor.

Requirements: 7.1, 7.2, 7.3

Tests cover:
  - Note concatenation: chronological ordering by note_date, truncation at
    exactly 10,000 tokens, empty list → apply_defaults called with flag=1
  - Pattern classifier (detect_prior_auth_gap): keyword matching
  - Model name/version logged in Feature Store after process_claim
"""
from __future__ import annotations

import logging
from datetime import date
from typing import List
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from claim_denial.nlp.nlp_processor import (
    EMBEDDING_DIM,
    FLAG_NOTES_ABSENT,
    MAX_TOKENS,
    ClinicalNote,
    NLPProcessor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_note(text: str, note_date: date, note_type: str = "Progress Note") -> ClinicalNote:
    return ClinicalNote(note_type=note_type, note_text=text, note_date=note_date)


def _tokens(n: int, word: str = "word") -> str:
    """Return a string of exactly *n* whitespace-separated tokens."""
    return " ".join([word] * n)


# ---------------------------------------------------------------------------
# Note concatenation — chronological ordering (Requirement 7.1)
# ---------------------------------------------------------------------------


class TestPrepareNotesOrdering:
    """Notes SHALL be concatenated in chronological order by note_date."""

    def test_two_notes_sorted_ascending(self):
        """Older note text must come before newer note text."""
        processor = NLPProcessor()
        notes = [
            _make_note("NEWER", date(2024, 3, 1)),
            _make_note("OLDER", date(2024, 1, 1)),
        ]
        result = processor.prepare_notes(notes)
        older_pos = result.index("OLDER")
        newer_pos = result.index("NEWER")
        assert older_pos < newer_pos, (
            "Expected 'OLDER' text to appear before 'NEWER' text after chronological sort"
        )

    def test_three_notes_sorted_ascending(self):
        """Three notes are sorted oldest-first regardless of input order."""
        processor = NLPProcessor()
        notes = [
            _make_note("THIRD", date(2024, 3, 1)),
            _make_note("FIRST", date(2024, 1, 1)),
            _make_note("SECOND", date(2024, 2, 1)),
        ]
        result = processor.prepare_notes(notes)
        assert result.index("FIRST") < result.index("SECOND") < result.index("THIRD")

    def test_same_date_notes_all_included(self):
        """Notes sharing the same date must all appear in the output."""
        processor = NLPProcessor()
        notes = [
            _make_note("ALPHA", date(2024, 6, 15)),
            _make_note("BETA", date(2024, 6, 15)),
        ]
        result = processor.prepare_notes(notes)
        assert "ALPHA" in result
        assert "BETA" in result

    def test_single_note_returned_as_is(self):
        """A single note should be returned unchanged (no truncation needed)."""
        processor = NLPProcessor()
        notes = [_make_note("Single note text", date(2024, 1, 1))]
        result = processor.prepare_notes(notes)
        assert result == "Single note text"


# ---------------------------------------------------------------------------
# Note truncation — exactly 10,000 tokens (Requirement 7.1)
# ---------------------------------------------------------------------------


class TestPrepareNotesTruncation:
    """Concatenated notes SHALL be truncated to at most 10,000 tokens."""

    def test_truncation_at_max_tokens(self):
        """Text with more than MAX_TOKENS tokens is truncated to MAX_TOKENS."""
        processor = NLPProcessor()
        note_text = _tokens(MAX_TOKENS + 500)  # 10,500 tokens
        notes = [_make_note(note_text, date(2024, 1, 1))]
        result = processor.prepare_notes(notes)
        assert len(result.split()) == MAX_TOKENS

    def test_exactly_max_tokens_not_truncated(self):
        """Text with exactly MAX_TOKENS tokens is returned without truncation."""
        processor = NLPProcessor()
        note_text = _tokens(MAX_TOKENS)
        notes = [_make_note(note_text, date(2024, 1, 1))]
        result = processor.prepare_notes(notes)
        assert len(result.split()) == MAX_TOKENS

    def test_fewer_than_max_tokens_not_truncated(self):
        """Text with fewer than MAX_TOKENS tokens is returned in full."""
        processor = NLPProcessor()
        note_text = _tokens(100)
        notes = [_make_note(note_text, date(2024, 1, 1))]
        result = processor.prepare_notes(notes)
        assert len(result.split()) == 100

    def test_multi_note_truncation_respects_chronological_order(self):
        """When combined notes exceed MAX_TOKENS, truncation preserves the oldest tokens."""
        processor = NLPProcessor()
        # Older note: 9,000 "old" tokens; newer note: 2,000 "new" tokens — combined 11,000
        older = _make_note(_tokens(9_000, "old"), date(2024, 1, 1))
        newer = _make_note(_tokens(2_000, "new"), date(2024, 6, 1))
        result = processor.prepare_notes([newer, older])  # unsorted input
        tokens = result.split()
        assert len(tokens) == MAX_TOKENS
        # First 9,000 tokens should be "old" (from the older note)
        assert all(t == "old" for t in tokens[:9_000])
        # The remaining 1,000 tokens come from the "new" note
        assert all(t == "new" for t in tokens[9_000:])


# ---------------------------------------------------------------------------
# Empty note list → apply_defaults with flag=1 (Requirement 7.4)
# ---------------------------------------------------------------------------


class TestEmptyNotesAppliesDefaults:
    """When notes is empty or None, process_claim must apply defaults with flag=1."""

    def test_empty_list_calls_apply_defaults_flag1(self):
        """process_claim with an empty list delegates to apply_defaults(flag=1)."""
        processor = NLPProcessor()
        with patch.object(processor, "apply_defaults", wraps=processor.apply_defaults) as mock_ad:
            features = processor.process_claim("CLM001", [])
        mock_ad.assert_called_once_with("CLM001", FLAG_NOTES_ABSENT)
        assert features.flag == FLAG_NOTES_ABSENT

    def test_none_notes_calls_apply_defaults_flag1(self):
        """process_claim with None delegates to apply_defaults(flag=1)."""
        processor = NLPProcessor()
        with patch.object(processor, "apply_defaults", wraps=processor.apply_defaults) as mock_ad:
            features = processor.process_claim("CLM002", None)
        mock_ad.assert_called_once_with("CLM002", FLAG_NOTES_ABSENT)
        assert features.flag == FLAG_NOTES_ABSENT

    def test_empty_list_returns_correct_defaults(self):
        """process_claim with empty list returns zero embedding and correct defaults."""
        processor = NLPProcessor()
        features = processor.process_claim("CLM003", [])
        assert features.medical_necessity == 0.5
        assert features.pre_auth == 0
        assert features.embedding.shape == (EMBEDDING_DIM,)
        assert features.embedding.dtype == np.float32
        assert np.all(features.embedding == 0.0)
        assert features.flag == FLAG_NOTES_ABSENT


# ---------------------------------------------------------------------------
# detect_prior_auth_gap — keyword classifier (Requirement 7.2)
# ---------------------------------------------------------------------------


class TestDetectPriorAuthGap:
    """detect_prior_auth_gap must return 1 for known keywords, 0 otherwise."""

    @pytest.mark.parametrize("text", [
        "Patient requires prior authorization for this procedure.",
        "The claim was denied due to prior auth not obtained.",
        "Pre-authorization was not secured before admission.",
        "pre authorization denied by payer",
        "preauth was missing from the record",
        "authorization required before service",
        "Service was not authorized by the payer",
        "There is a noted auth gap in this patient's record",
        # case-insensitivity checks
        "PRIOR AUTHORIZATION required",
        "Prior Auth Gap noted",
    ])
    def test_keyword_present_returns_1(self, text: str):
        """Any recognized prior-auth keyword in the text must return 1."""
        processor = NLPProcessor()
        assert processor.detect_prior_auth_gap(text) == 1, (
            f"Expected 1 for text containing pre-auth keyword: {text!r}"
        )

    @pytest.mark.parametrize("text", [
        "Patient presents with chest pain and shortness of breath.",
        "Discharge summary: patient recovered well.",
        "History and physical examination normal.",
        "Labs within reference ranges.",
        "",
    ])
    def test_no_keyword_returns_0(self, text: str):
        """Text with no prior-auth keywords must return 0."""
        processor = NLPProcessor()
        assert processor.detect_prior_auth_gap(text) == 0, (
            f"Expected 0 for text with no pre-auth keyword: {text!r}"
        )

    def test_prior_authorization_exact(self):
        """'prior authorization' (lowercase, two words) must match."""
        processor = NLPProcessor()
        assert processor.detect_prior_auth_gap("prior authorization required") == 1

    def test_prior_auth_exact(self):
        """'prior auth' must match."""
        processor = NLPProcessor()
        assert processor.detect_prior_auth_gap("prior auth was denied") == 1


# ---------------------------------------------------------------------------
# Model name/version in Feature Store (Requirement 7.3)
# ---------------------------------------------------------------------------


class TestModelNameVersionLogged:
    """After process_claim, model_name and model_version must appear in the
    Feature Store row for that claim."""

    def test_model_name_written_to_feature_store(self):
        """nlp_model_name in Feature Store row must match the model's model_name."""
        feature_store: dict = {}
        processor = NLPProcessor(feature_store=feature_store)
        notes = [_make_note("Normal clinical note text.", date(2024, 1, 1))]
        processor.process_claim("CLM100", notes)

        row = processor.get_feature_store_row("CLM100")
        assert row is not None
        assert "nlp_model_name" in row
        assert row["nlp_model_name"] == processor._model.model_name

    def test_model_version_written_to_feature_store(self):
        """nlp_model_version in Feature Store row must match the model's model_version."""
        feature_store: dict = {}
        processor = NLPProcessor(feature_store=feature_store)
        notes = [_make_note("Normal clinical note text.", date(2024, 1, 1))]
        processor.process_claim("CLM101", notes)

        row = processor.get_feature_store_row("CLM101")
        assert row is not None
        assert "nlp_model_version" in row
        assert row["nlp_model_version"] == processor._model.model_version

    def test_model_name_written_even_on_defaults_path(self):
        """Model name/version are written to Feature Store even when defaults apply (empty notes)."""
        feature_store: dict = {}
        processor = NLPProcessor(feature_store=feature_store)
        processor.process_claim("CLM102", [])

        row = processor.get_feature_store_row("CLM102")
        assert row is not None
        assert row["nlp_model_name"] == processor._model.model_name
        assert row["nlp_model_version"] == processor._model.model_version

    def test_custom_model_name_version_propagated(self):
        """A custom model's name/version must appear in the Feature Store row."""

        class _CustomModel:
            model_name: str = "custom-bert-v2"
            model_version: str = "2.3.1"

            def encode(self, text: str) -> np.ndarray:
                return np.ones(EMBEDDING_DIM, dtype=np.float32)

        feature_store: dict = {}
        processor = NLPProcessor(model=_CustomModel(), feature_store=feature_store)
        notes = [_make_note("Clinical note.", date(2024, 5, 1))]
        processor.process_claim("CLM103", notes)

        row = processor.get_feature_store_row("CLM103")
        assert row is not None
        assert row["nlp_model_name"] == "custom-bert-v2"
        assert row["nlp_model_version"] == "2.3.1"
