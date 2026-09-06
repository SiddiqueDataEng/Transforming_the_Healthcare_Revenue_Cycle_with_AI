"""
FeatureRecordSerializer — Avro serialization / deserialization for
ClaimFeatureRecord using fastavro.

Serialization contract
----------------------
* Every serialized record embeds a ``schema_version`` field so that
  readers can apply the correct schema migration if needed.
* Deserialization applies chained migrations via :class:`SchemaRegistry`
  when the embedded version differs from :data:`CURRENT_SCHEMA_VERSION`.
* If no migration rule exists the record is quarantined by raising
  :class:`QuarantineError`.
* Corrupt / unreadable records are logged and skipped (the caller
  receives ``None`` back from :meth:`FeatureRecordSerializer.deserialize`).

Requirements: 15.1, 15.2, 15.3, 15.5
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict
from typing import Optional

import fastavro

from claim_denial.models import ClaimFeatureRecord
from claim_denial.schemas.registry import NoMigrationRuleError, SchemaRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Current schema version
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Avro schema — derived from ClaimFeatureRecord fields in models.py
# ---------------------------------------------------------------------------

_AVRO_SCHEMA: dict = {
    "type": "record",
    "name": "ClaimFeatureRecord",
    "namespace": "com.rcm.features",
    "fields": [
        # Core identifiers
        {"name": "schema_version",                       "type": "string"},
        {"name": "claim_id",                             "type": "string"},
        {"name": "claim_status",                         "type": "string"},

        # ── Claim-Level Features ──────────────────────────────────────
        {"name": "claim_submission_date_doy",            "type": "int",    "default": 0},
        {"name": "claim_submission_date_month",          "type": "int",    "default": 0},
        {"name": "patient_age",                          "type": "int",    "default": -1},
        {"name": "gender",                               "type": "int",    "default": -1},
        {"name": "ins_type_medicare",                    "type": "int",    "default": 0},
        {"name": "ins_type_medicaid",                    "type": "int",    "default": 0},
        {"name": "ins_type_commercial",                  "type": "int",    "default": 0},
        {"name": "ins_type_tricare",                     "type": "int",    "default": 0},
        {"name": "ins_type_champva",                     "type": "int",    "default": 0},
        {"name": "ins_type_other",                       "type": "int",    "default": 0},
        {"name": "claim_type_professional",              "type": "int",    "default": 0},
        {"name": "claim_type_institutional",             "type": "int",    "default": 0},
        {"name": "adm_type_emergency",                   "type": "int",    "default": 0},
        {"name": "adm_type_elective",                    "type": "int",    "default": 0},
        {"name": "adm_type_urgent",                      "type": "int",    "default": 0},
        {"name": "adm_type_trauma",                      "type": "int",    "default": 0},
        {"name": "adm_src_physician_referral",           "type": "int",    "default": 0},
        {"name": "adm_src_transfer_hospital",            "type": "int",    "default": 0},
        {"name": "adm_src_transfer_snf",                 "type": "int",    "default": 0},
        {"name": "adm_src_er",                           "type": "int",    "default": 0},
        {"name": "adm_src_court_law",                    "type": "int",    "default": 0},
        {"name": "adm_src_not_available",                "type": "int",    "default": 0},
        {"name": "partial_ehr_flag",                     "type": ["null", "string"], "default": None},
        {"name": "ehr_null_match_flag",                  "type": "int",    "default": 0},

        # ── Provider / Facility Features ──────────────────────────────
        {"name": "billing_provider_id_encoded",          "type": "float",  "default": 0.0},
        {"name": "performing_physician_specialty",       "type": "string", "default": "UNKNOWN"},
        {"name": "provider_historical_denial_rate",      "type": "float",  "default": 0.0},
        {"name": "provider_claim_volume",                "type": "int",    "default": 0},
        {"name": "facility_bed_size",                    "type": "int",    "default": 0},

        # ── Clinical Features ─────────────────────────────────────────
        {"name": "principal_dx_ccs_category",            "type": "int",    "default": 0},
        {"name": "secondary_dx_count",                   "type": "int",    "default": 0},
        {
            "name": "procedure_category_vector",
            "type": {"type": "array", "items": "int"},
            "default": [],
        },
        {"name": "length_of_stay",                       "type": "int",    "default": -2},
        {"name": "comorbidity_risk_score",               "type": "int",    "default": 0},
        {"name": "diagnosis_procedure_mismatch",         "type": "int",    "default": 0},

        # ── Payer Behavior Features ───────────────────────────────────
        {"name": "payer_historical_denial_rate",         "type": "float",  "default": 0.0},
        {"name": "payer_historical_denial_rate_by_type", "type": "float",  "default": 0.0},
        {"name": "days_since_last_payment",              "type": "int",    "default": -1},
        {"name": "payer_contract_stop_loss",             "type": "int",    "default": 0},

        # ── NLP Features ─────────────────────────────────────────────
        {"name": "documents_medical_necessity",          "type": "float",  "default": 0.5},
        {"name": "mentions_lack_of_pre_auth",            "type": "int",    "default": 0},
        {
            "name": "tx_plan_complexity_embedding",
            "type": {"type": "array", "items": "float"},
            "default": [],
        },
        {"name": "nlp_notes_absent_flag",                "type": "int",    "default": 0},
        {"name": "nlp_model_name",                       "type": ["null", "string"], "default": None},
        {"name": "nlp_model_version",                    "type": ["null", "string"], "default": None},

        # ── Scoring Output ────────────────────────────────────────────
        {"name": "predicted_denial_score",               "type": ["null", "float"],  "default": None},
        {"name": "score_quality_flag",                   "type": ["null", "string"], "default": None},
        {"name": "model_version_id",                     "type": ["null", "string"], "default": None},
        {"name": "scoring_timestamp_utc",                "type": ["null", "string"], "default": None},

        # ── Ground Truth (post-adjudication) ─────────────────────────
        {"name": "adjudication_outcome",                 "type": ["null", "string"], "default": None},
        {"name": "adjudication_timestamp_utc",           "type": ["null", "string"], "default": None},
    ],
}

# Pre-parse the schema so fastavro can validate it once at import time.
_PARSED_SCHEMA = fastavro.parse_schema(_AVRO_SCHEMA)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class QuarantineError(Exception):
    """Raised when a record must be quarantined (no migration rule)."""

    def __init__(self, claim_id: str, embedded_version: str, current_version: str) -> None:
        self.claim_id = claim_id
        self.embedded_version = embedded_version
        self.current_version = current_version
        super().__init__(
            f"Quarantine: claim_id={claim_id!r}, "
            f"embedded_version={embedded_version!r}, "
            f"current_version={current_version!r} — no migration rule exists"
        )


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

class FeatureRecordSerializer:
    """
    Serialize / deserialize :class:`~claim_denial.models.ClaimFeatureRecord`
    objects to/from Avro bytes using fastavro.

    Parameters
    ----------
    registry:
        Optional :class:`~claim_denial.schemas.registry.SchemaRegistry`.
        If ``None`` a fresh (empty) registry is used.  Inject a populated
        registry to enable schema migration during deserialization.
    current_version:
        The schema version string to embed in every serialized record.
        Defaults to :data:`CURRENT_SCHEMA_VERSION`.
    """

    def __init__(
        self,
        registry: Optional[SchemaRegistry] = None,
        current_version: str = CURRENT_SCHEMA_VERSION,
    ) -> None:
        self._registry: SchemaRegistry = registry if registry is not None else SchemaRegistry()
        self._current_version: str = current_version

    # ------------------------------------------------------------------
    # serialize
    # ------------------------------------------------------------------

    def serialize(self, record: ClaimFeatureRecord) -> bytes:
        """Encode *record* as Avro bytes using the versioned schema.

        The ``schema_version`` field in the serialized record is
        **always overwritten** with :attr:`_current_version` so that
        every byte stream carries the correct version regardless of what
        the caller stored on the dataclass.

        Parameters
        ----------
        record:
            A :class:`~claim_denial.models.ClaimFeatureRecord` instance.

        Returns
        -------
        bytes
            Avro-encoded bytes (single-record container, using
            ``io.BytesIO``).
        """
        record_dict = asdict(record)
        # Always embed the current schema version (Requirement 15.1)
        record_dict["schema_version"] = self._current_version

        buf = io.BytesIO()
        fastavro.schemaless_writer(buf, _PARSED_SCHEMA, record_dict)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # deserialize
    # ------------------------------------------------------------------

    def deserialize(self, data: bytes) -> Optional[ClaimFeatureRecord]:
        """Decode Avro *data* bytes into a :class:`~claim_denial.models.ClaimFeatureRecord`.

        Processing steps
        ----------------
        1. Read the raw dict from the Avro byte stream.
        2. Extract the embedded ``schema_version``.
        3. If it differs from :attr:`_current_version`, apply chained
           migrations via the :class:`~claim_denial.schemas.registry.SchemaRegistry`.
        4. If no migration rule exists, log the mismatch and raise
           :class:`QuarantineError` (Requirement 15.5).
        5. On any other deserialization failure, log the Claim ID and
           reason and return ``None`` so the caller can skip the record
           (Requirement 15.2).

        Parameters
        ----------
        data:
            Avro-encoded bytes as produced by :meth:`serialize`.

        Returns
        -------
        ClaimFeatureRecord | None
            The deserialized record, or ``None`` if the record is corrupt
            and should be skipped.

        Raises
        ------
        QuarantineError
            When the embedded schema version has no registered migration
            path to the current version.
        """
        try:
            buf = io.BytesIO(data)
            record_dict: dict = fastavro.schemaless_reader(buf, _PARSED_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            claim_id = "<unknown>"
            logger.error(
                "Deserialization failure: claim_id=%s reason=%s",
                claim_id,
                str(exc),
            )
            return None

        embedded_version: str = record_dict.get("schema_version", "")
        claim_id: str = record_dict.get("claim_id", "<unknown>")

        # Apply schema migration if needed (Requirement 15.5)
        if embedded_version != self._current_version:
            try:
                record_dict = self._registry.apply_migrations(
                    record_dict,
                    embedded_version,
                    self._current_version,
                )
            except NoMigrationRuleError:
                logger.error(
                    "Schema-version mismatch — no migration rule: "
                    "claim_id=%s embedded_version=%s current_version=%s",
                    claim_id,
                    embedded_version,
                    self._current_version,
                )
                raise QuarantineError(claim_id, embedded_version, self._current_version)

        try:
            return ClaimFeatureRecord(**record_dict)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Deserialization failure (field mapping): claim_id=%s reason=%s",
                claim_id,
                str(exc),
            )
            return None

    # ------------------------------------------------------------------
    # pretty_print
    # ------------------------------------------------------------------

    def pretty_print(self, record: ClaimFeatureRecord) -> str:
        """Return a human-readable JSON string of *record*.

        The output contains field names and typed values, suitable for
        audit logs and debugging (Requirement 15.3).

        Parameters
        ----------
        record:
            A :class:`~claim_denial.models.ClaimFeatureRecord` instance.

        Returns
        -------
        str
            Pretty-printed JSON string (2-space indentation).
        """
        return json.dumps(asdict(record), indent=2, default=str)
