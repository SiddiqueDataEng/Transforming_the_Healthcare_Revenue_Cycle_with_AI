"""
NLP feature extraction for the Claim Denial Prediction system.

Implements NLPProcessor, which extracts three NLP feature columns from
free-text clinical notes (Requirements 7.1–7.6):

  * ``documents_medical_necessity`` — clinical BERT embedding → scalar score
  * ``mentions_lack_of_pre_auth``   — keyword/pattern classifier (binary 0/1)
  * ``tx_plan_complexity_embedding`` — 768-dim float32 BERT embedding vector

The live clinical BERT model is **stubbed** via ``MockClinicalBERT``, which
returns deterministic, realistic-looking 768-dim float32 arrays derived from
a hash of the input text.  Swap in a real ``transformers`` pipeline by
replacing the ``_model`` attribute on construction.

Chronological note ordering is derived from ``ClinicalNote.note_date`` (and
``note_datetime`` when available from the shared ``models.ClinicalNote``
dataclass).  A lightweight local ``ClinicalNote`` dataclass is also exported
for callers that do not use the shared models module.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of whitespace-separated tokens to retain after concatenation.
MAX_TOKENS: int = 10_000

#: Dimensionality of the BERT embedding vector.
EMBEDDING_DIM: int = 768

#: Default medical necessity score (absent notes / inference error).
DEFAULT_MEDICAL_NECESSITY: float = 0.5

#: Flag value — no clinical notes available.
FLAG_NOTES_ABSENT: int = 1

#: Flag value — NLP inference error or timeout.
FLAG_INFERENCE_ERROR: int = 2

# ---------------------------------------------------------------------------
# Prior-authorisation keyword / pattern list (Requirement 7.2)
# ---------------------------------------------------------------------------

_PRIOR_AUTH_PATTERNS: List[str] = [
    r"prior\s+authorization",
    r"pre[-\s]authorization",
    r"pre\s+authorization",
    r"auth\s+gap",
    r"prior\s+auth",
    r"preauth",
    r"authorization\s+required",
    r"not\s+authorized",
]

_PRIOR_AUTH_REGEX: re.Pattern = re.compile(
    "|".join(_PRIOR_AUTH_PATTERNS),
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClinicalNote:
    """
    A single free-text clinical note.

    This lightweight dataclass is defined here so that callers that do not
    import ``claim_denial.models`` can still use ``NLPProcessor``.  If you
    already have ``claim_denial.models.ClinicalNote`` instances, they are
    compatible because ``NLPProcessor`` only reads ``note_date``, ``text``,
    and optionally ``note_datetime``.

    Attributes
    ----------
    note_type:
        Document type label (e.g. "H&P", "Discharge Summary", "Progress Note").
    note_text:
        Free-text body of the clinical note.
    note_date:
        ISO-8601 date the note was authored (used for chronological ordering).
    """

    note_type: str
    note_text: str
    note_date: date


@dataclass
class NLPFeatures:
    """
    Container for the three NLP feature columns produced by ``NLPProcessor``.

    Attributes
    ----------
    medical_necessity:
        Decimal confidence score ∈ [0.0, 1.0] rounded to 4 decimal places.
    pre_auth:
        Binary prior-authorisation gap indicator (0 or 1).
    embedding:
        Float32 numpy array of exactly 768 dimensions.
    flag:
        0 — normal output.
        1 — notes absent (defaults applied, Requirement 7.4).
        2 — inference error (defaults applied, Requirement 7.5).
    """

    medical_necessity: float
    pre_auth: int
    embedding: np.ndarray
    flag: int


# ---------------------------------------------------------------------------
# Mock Clinical BERT (stub — no internet download required)
# ---------------------------------------------------------------------------


class MockClinicalBERT:
    """
    Lightweight stub for a clinical BERT model.

    Produces deterministic, realistic-looking 768-dim float32 embeddings by
    seeding a numpy ``RandomState`` with an MD5 hash of the input text.  The
    values fall in approximately (−1, 1), similar to actual BERT hidden states.

    Attributes
    ----------
    model_name:
        Canonical name logged alongside each NLP output.
    model_version:
        Semantic version string logged alongside each NLP output.
    """

    model_name: str = "mock-clinical-bert"
    model_version: str = "0.1.0-stub"

    def encode(self, text: str) -> np.ndarray:
        """
        Return a deterministic 768-dim float32 embedding for *text*.

        The embedding is derived from a 32-bit seed produced by MD5-hashing
        the UTF-8 encoded text, then sampling from a standard normal
        distribution seeded with that value.

        Parameters
        ----------
        text:
            Input string to embed.

        Returns
        -------
        np.ndarray
            Shape ``(768,)``, dtype ``float32``.
        """
        digest = hashlib.md5(text.encode("utf-8")).digest()
        # Fold first 4 bytes into a uint32 seed
        seed = int.from_bytes(digest[:4], byteorder="big")
        rng = np.random.RandomState(seed)
        embedding = rng.randn(EMBEDDING_DIM).astype(np.float32)
        return embedding


# ---------------------------------------------------------------------------
# NLPProcessor
# ---------------------------------------------------------------------------


class NLPProcessor:
    """
    Extract NLP-derived denial-risk features from free-text clinical notes.

    Parameters
    ----------
    model:
        A BERT-compatible model object with an ``encode(text) → np.ndarray``
        method, a ``model_name`` attribute, and a ``model_version`` attribute.
        Defaults to ``MockClinicalBERT()``.
    feature_store:
        Injectable dict-like Feature Store (``Dict[str, Dict[str, Any]]``).
        When ``None``, an internal empty dict is created.  In production this
        would be replaced by a real HBase/Cassandra client.
    """

    def __init__(
        self,
        model: Optional[object] = None,
        feature_store: Optional[dict] = None,
    ) -> None:
        self._model: MockClinicalBERT = model if model is not None else MockClinicalBERT()
        self._feature_store: dict = feature_store if feature_store is not None else {}

    # ------------------------------------------------------------------
    # Public feature-extraction API
    # ------------------------------------------------------------------

    def prepare_notes(self, clinical_notes: List[ClinicalNote]) -> str:
        """
        Concatenate clinical notes in chronological order and truncate.

        Notes are sorted by ``note_date`` (ascending).  When a note object
        also carries a ``note_datetime`` attribute (as in the shared
        ``models.ClinicalNote`` dataclass), that is used as the secondary sort
        key; otherwise ``note_date`` alone governs ordering.

        The concatenated text is split on whitespace and truncated to at most
        ``MAX_TOKENS`` (10,000) tokens before being rejoined with single
        spaces.

        Parameters
        ----------
        clinical_notes:
            List of clinical note objects.  Each must expose ``note_date`` and
            either ``note_text`` (this module's dataclass) or ``text``
            (``models.ClinicalNote``).

        Returns
        -------
        str
            Truncated, concatenated note text.
        """
        if not clinical_notes:
            return ""

        def _sort_key(note: ClinicalNote):
            # Support both the local ClinicalNote (note_text) and the shared
            # models.ClinicalNote (text / note_datetime).
            dt = getattr(note, "note_datetime", None)
            if dt is not None:
                return dt
            # Combine note_date with midnight so we can compare uniformly
            return datetime.combine(note.note_date, datetime.min.time())

        sorted_notes = sorted(clinical_notes, key=_sort_key)

        # Collect text — support both attribute names
        parts: List[str] = []
        for note in sorted_notes:
            text = getattr(note, "note_text", None) or getattr(note, "text", "")
            if text:
                parts.append(text)

        combined = " ".join(parts)

        # Truncate to MAX_TOKENS whitespace-delimited tokens
        tokens = combined.split()
        if len(tokens) > MAX_TOKENS:
            tokens = tokens[:MAX_TOKENS]

        return " ".join(tokens)

    def compute_bert_embedding(self, text: str) -> np.ndarray:
        """
        Encode *text* with the clinical BERT model and return a 768-dim vector.

        The model name and version are logged at DEBUG level alongside each
        invocation for reproducibility (Requirement 7.3).

        Parameters
        ----------
        text:
            Truncated, concatenated note text (output of ``prepare_notes``).

        Returns
        -------
        np.ndarray
            Shape ``(768,)``, dtype ``float32``.

        Raises
        ------
        RuntimeError
            Re-raises any exception thrown by the underlying model after
            logging, so that ``NLPProcessor.process_claim`` can catch it and
            apply defaults (flag=2).
        """
        logger.debug(
            "compute_bert_embedding: model=%s version=%s text_len=%d",
            self._model.model_name,
            self._model.model_version,
            len(text),
        )
        embedding: np.ndarray = self._model.encode(text)

        if embedding.shape != (EMBEDDING_DIM,):
            raise RuntimeError(
                f"Model returned embedding of shape {embedding.shape}; "
                f"expected ({EMBEDDING_DIM},)"
            )

        return embedding.astype(np.float32)

    def compute_medical_necessity_score(self, embedding: np.ndarray) -> float:
        """
        Derive a medical necessity confidence score from *embedding*.

        The score is computed as the mean of the embedding values clamped to
        [0.0, 1.0], then rounded to 4 decimal places.

        Parameters
        ----------
        embedding:
            Float32 numpy array of shape ``(768,)``.

        Returns
        -------
        float
            Decimal in [0.0, 1.0] rounded to 4 decimal places.
        """
        mean_val: float = float(np.mean(embedding))
        # Clamp mean to [0, 1]
        score = max(0.0, min(1.0, mean_val))
        return round(score, 4)

    def detect_prior_auth_gap(self, text: str) -> int:
        """
        Apply a keyword/pattern classifier for prior-authorisation gaps.

        Scans *text* for any of the following phrases (case-insensitive):

        * "prior authorization"
        * "pre-authorization" / "pre authorization"
        * "auth gap"
        * "prior auth"
        * "preauth"
        * "authorization required"
        * "not authorized"

        Parameters
        ----------
        text:
            Concatenated note text to scan.

        Returns
        -------
        int
            1 if a prior-authorisation gap keyword is found, 0 otherwise.
        """
        if _PRIOR_AUTH_REGEX.search(text):
            return 1
        return 0

    def apply_defaults(self, claim_id: str, reason: int) -> NLPFeatures:
        """
        Return default NLP features for a claim that cannot be processed.

        Defaults (Requirement 7.4 / 7.5):
        * ``medical_necessity = 0.5``
        * ``pre_auth = 0``
        * ``embedding = np.zeros(768, dtype=float32)``
        * ``flag = reason``

        When ``reason == FLAG_INFERENCE_ERROR`` (2), the claim ID and failure
        reason are logged at WARNING level.

        Parameters
        ----------
        claim_id:
            Unique claim identifier (for logging).
        reason:
            ``FLAG_NOTES_ABSENT`` (1) — no clinical notes available.
            ``FLAG_INFERENCE_ERROR`` (2) — NLP inference failed.

        Returns
        -------
        NLPFeatures
            Default feature container with ``flag=reason``.
        """
        if reason == FLAG_INFERENCE_ERROR:
            logger.warning(
                "apply_defaults: claim_id=%s flag=%d (inference error); "
                "applying NLP defaults.",
                claim_id,
                reason,
            )

        return NLPFeatures(
            medical_necessity=DEFAULT_MEDICAL_NECESSITY,
            pre_auth=0,
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            flag=reason,
        )

    # ------------------------------------------------------------------
    # End-to-end claim processing
    # ------------------------------------------------------------------

    def process_claim(
        self,
        claim_id: str,
        clinical_notes: Optional[List[ClinicalNote]],
    ) -> NLPFeatures:
        """
        Extract NLP features for a single claim and upsert them to the
        Feature Store.

        Pipeline:

        1. If *clinical_notes* is ``None`` or empty, return defaults with
           ``flag=1`` (notes absent).
        2. Concatenate and truncate notes (``prepare_notes``).
        3. Compute BERT embedding (``compute_bert_embedding``).
        4. Compute medical necessity score (``compute_medical_necessity_score``).
        5. Detect prior-auth gap (``detect_prior_auth_gap``).
        6. Upsert NLP feature columns to the Feature Store.
        7. On any inference exception: log, return defaults with ``flag=2``.

        Parameters
        ----------
        claim_id:
            Unique claim identifier.
        clinical_notes:
            List of clinical note objects, or ``None`` / empty list.

        Returns
        -------
        NLPFeatures
            Computed (or default) NLP feature container.
        """
        if not clinical_notes:
            features = self.apply_defaults(claim_id, FLAG_NOTES_ABSENT)
            self._upsert(claim_id, features)
            return features

        try:
            text = self.prepare_notes(clinical_notes)
            embedding = self.compute_bert_embedding(text)
            necessity = self.compute_medical_necessity_score(embedding)
            pre_auth = self.detect_prior_auth_gap(text)

            features = NLPFeatures(
                medical_necessity=necessity,
                pre_auth=pre_auth,
                embedding=embedding,
                flag=0,
            )

        except Exception as exc:
            logger.error(
                "process_claim: inference error for claim_id=%s: %s",
                claim_id,
                exc,
                exc_info=True,
            )
            features = self.apply_defaults(claim_id, FLAG_INFERENCE_ERROR)

        self._upsert(claim_id, features)
        return features

    def process_batch(
        self,
        claims: List[tuple],
    ) -> dict:
        """
        Process a batch of ``(claim_id, clinical_notes)`` tuples.

        This method satisfies the 3-hour SLA requirement (Requirement 7.6) by
        processing each claim sequentially.  In production, parallelism
        (e.g., PySpark ``flatMap``) should be layered above this method.

        Parameters
        ----------
        claims:
            Iterable of ``(claim_id, notes_or_none)`` pairs.

        Returns
        -------
        dict
            Mapping of ``claim_id → NLPFeatures``.
        """
        results: dict = {}
        for claim_id, notes in claims:
            results[claim_id] = self.process_claim(claim_id, notes)
        return results

    # ------------------------------------------------------------------
    # Feature Store access helpers
    # ------------------------------------------------------------------

    def _upsert(self, claim_id: str, features: NLPFeatures) -> None:
        """Write NLP feature columns to the Feature Store row for *claim_id*."""
        columns = {
            "documents_medical_necessity": features.medical_necessity,
            "mentions_lack_of_pre_auth": features.pre_auth,
            "tx_plan_complexity_embedding": features.embedding.tolist(),
            "nlp_notes_absent_flag": features.flag,
            "nlp_model_name": self._model.model_name,
            "nlp_model_version": self._model.model_version,
        }
        if claim_id not in self._feature_store:
            self._feature_store[claim_id] = {}
        self._feature_store[claim_id].update(columns)

    def get_feature_store_row(self, claim_id: str) -> Optional[dict]:
        """Return the full Feature Store row for *claim_id*, or ``None``."""
        return self._feature_store.get(claim_id)

    @property
    def feature_store(self) -> dict:
        """Direct access to the underlying Feature Store dict (for testing)."""
        return self._feature_store
