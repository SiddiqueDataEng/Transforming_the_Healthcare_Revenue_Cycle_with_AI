"""
Model Registry for the Claim Denial Prediction system.

Provides a lightweight, MLflow-backed registry that:
  - Registers trained model artifacts with stage "Staging" or "Failed".
  - Enforces stage transitions via human review gates.
  - Retains at least 2 most-recent Production versions in a loadable state.
  - Exposes ``get_production_model()`` for the Batch_Scorer.

Requirements: 9.8, 9.9, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from claim_denial.models import ModelStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NoProductionModelError(RuntimeError):
    """Raised when no model version with stage 'Production' exists."""


class InvalidStageTransitionError(ValueError):
    """Raised when a requested stage transition is not permitted."""


# ---------------------------------------------------------------------------
# ModelArtifact
# ---------------------------------------------------------------------------


@dataclass
class ModelArtifact:
    """
    A versioned model artifact stored in the registry.

    Attributes
    ----------
    version_id:
        Unique string identifier for this model version (e.g. "v1", "v2").
    stage:
        Current ModelStage value (Staging / Production / Archived / Failed).
    training_date:
        ISO-8601 datetime string when training completed.
    algorithm:
        One of "XGBoost", "CatBoost", or "LightGBM".
    algorithm_version:
        Library version string (e.g. "1.7.6").
    hyperparameters:
        Full hyperparameter dict from Bayesian search.
    precision:
        Held-out test set precision.
    recall:
        Held-out test set recall.
    f1:
        Held-out test set F1.
    auc_roc:
        Held-out test set AUC-ROC.
    auc_pr:
        Held-out test set AUC-PR.
    ece:
        Expected Calibration Error (10-bin).
    shap_top20:
        Top-20 features by mean absolute SHAP value.
    train_record_count:
        Records in training split.
    test_record_count:
        Records in held-out test split.
    denied_class_pct:
        Denied class percentage in training.
    cutoff_date:
        Train/test boundary date (ISO-8601).
    reviewer_id:
        Human reviewer identifier (set on gate actions).
    review_timestamp:
        ISO-8601 datetime when reviewed.
    rejection_reason:
        Populated on rejection.
    trigger_reason:
        e.g. "precision_threshold_breach".
    container_image_tag:
        Batch scorer Docker image tag.
    container_image_digest:
        Batch scorer Docker image digest.
    model_object:
        The in-memory model object (e.g. XGBoost Booster).  Not persisted by
        this lightweight registry; set when the artifact is loaded from disk.
    """

    version_id: str
    stage: ModelStage
    training_date: str
    algorithm: str
    algorithm_version: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    ece: float = 0.0
    shap_top20: List[Dict[str, Any]] = field(default_factory=list)
    train_record_count: int = 0
    test_record_count: int = 0
    denied_class_pct: float = 0.0
    cutoff_date: Optional[str] = None
    reviewer_id: Optional[str] = None
    review_timestamp: Optional[str] = None
    rejection_reason: Optional[str] = None
    trigger_reason: Optional[str] = None
    container_image_tag: Optional[str] = None
    container_image_digest: Optional[str] = None
    # In-memory model object; not serialised by this class.
    model_object: Any = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Lightweight, in-process model registry.

    In production this class wraps MLflow's Model Registry.  For tests and
    local development it uses an in-memory list of :class:`ModelArtifact`
    objects.

    Stage machine (Requirement 10)
    --------------------------------
    ::

        [*] → Staging    (Precision ≥ 0.90 AND ECE ≤ 0.05)
        [*] → Failed     (Precision < 0.90 OR  ECE > 0.05)
        Staging   → Production  (human approval)
        Staging   → Archived    (human rejection)
        Production → Archived   (superseded by new Production)
        Failed    → [terminal]
        Archived  → [terminal]
    """

    #: Precision threshold required for Staging promotion.
    PRECISION_THRESHOLD: float = 0.90
    #: ECE threshold required for Staging promotion.
    ECE_THRESHOLD: float = 0.05

    def __init__(self) -> None:
        # Ordered list of all registered artifacts (oldest first).
        self._versions: List[ModelArtifact] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(self, artifact: ModelArtifact) -> ModelArtifact:
        """
        Register a trained model artifact.

        The stage on *artifact* is overridden by this method based on the
        precision / ECE thresholds (Requirements 9.8, 9.9).

        Parameters
        ----------
        artifact:
            Newly trained model artifact.  The ``stage`` field will be set
            by this method; any value provided by the caller is ignored.

        Returns
        -------
        ModelArtifact
            The registered artifact with its assigned stage.
        """
        if (
            artifact.precision >= self.PRECISION_THRESHOLD
            and artifact.ece <= self.ECE_THRESHOLD
        ):
            artifact.stage = ModelStage.STAGING
            logger.info(
                "Model %s registered as Staging (precision=%.4f, ece=%.4f).",
                artifact.version_id,
                artifact.precision,
                artifact.ece,
            )
        else:
            artifact.stage = ModelStage.FAILED
            logger.warning(
                "Model %s registered as Failed (precision=%.4f, ece=%.4f).",
                artifact.version_id,
                artifact.precision,
                artifact.ece,
            )

        self._versions.append(artifact)
        return artifact

    # ------------------------------------------------------------------
    # Human review gates
    # ------------------------------------------------------------------

    def approve_model(self, version_id: str, reviewer_id: str) -> None:
        """
        Transition a Staging model to Production.

        The previously active Production model (if any) is automatically
        transitioned to Archived.

        Requirements 10.2, 10.4

        Parameters
        ----------
        version_id:
            The version identifier of the model to promote.
        reviewer_id:
            Identifier of the human reviewer approving the promotion.

        Raises
        ------
        InvalidStageTransitionError
            If the target model is not currently in stage "Staging".
        KeyError
            If *version_id* is not found in the registry.
        """
        artifact = self._get_version(version_id)

        if artifact.stage != ModelStage.STAGING:
            raise InvalidStageTransitionError(
                f"Model {version_id} is in stage '{artifact.stage.value}'; "
                "only Staging models can be approved."
            )

        # Archive the current Production model (if any).
        current_production = self._current_production()
        if current_production is not None:
            current_production.stage = ModelStage.ARCHIVED
            logger.info(
                "Model %s (previously Production) archived as %s is promoted.",
                current_production.version_id,
                version_id,
            )

        # Promote the target model.
        artifact.stage = ModelStage.PRODUCTION
        artifact.reviewer_id = reviewer_id
        artifact.review_timestamp = datetime.utcnow().isoformat() + "Z"

        logger.info(
            "Model %s promoted to Production by reviewer '%s'.",
            version_id,
            reviewer_id,
        )

    def reject_model(
        self,
        version_id: str,
        reviewer_id: str,
        reason: str,
    ) -> None:
        """
        Transition a Staging model to Archived (rejected).

        Requirement 10.3

        Parameters
        ----------
        version_id:
            The version identifier of the model to reject.
        reviewer_id:
            Identifier of the human reviewer rejecting the model.
        reason:
            Human-readable reason for rejection.

        Raises
        ------
        InvalidStageTransitionError
            If the target model is not currently in stage "Staging".
        KeyError
            If *version_id* is not found in the registry.
        """
        artifact = self._get_version(version_id)

        if artifact.stage != ModelStage.STAGING:
            raise InvalidStageTransitionError(
                f"Model {version_id} is in stage '{artifact.stage.value}'; "
                "only Staging models can be rejected."
            )

        artifact.stage = ModelStage.ARCHIVED
        artifact.reviewer_id = reviewer_id
        artifact.review_timestamp = datetime.utcnow().isoformat() + "Z"
        artifact.rejection_reason = reason

        logger.info(
            "Model %s rejected and archived by reviewer '%s' (reason: %s).",
            version_id,
            reviewer_id,
            reason,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_production_model(self) -> ModelArtifact:
        """
        Return the current Production model artifact.

        Requirement 10.6

        Returns
        -------
        ModelArtifact
            The single model version currently in stage "Production".

        Raises
        ------
        NoProductionModelError
            When no model version with stage "Production" exists.
        """
        artifact = self._current_production()
        if artifact is None:
            raise NoProductionModelError(
                "No model version with stage 'Production' exists in the registry."
            )
        return artifact

    def list_versions(
        self,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelArtifact]:
        """
        Return all registered model versions, optionally filtered by *stage*.

        Parameters
        ----------
        stage:
            When provided, only versions with this stage are returned.

        Returns
        -------
        List[ModelArtifact]
            Artifacts in registration order (oldest first).
        """
        if stage is None:
            return list(self._versions)
        return [v for v in self._versions if v.stage == stage]

    def get_recent_production_versions(self, n: int = 2) -> List[ModelArtifact]:
        """
        Return the *n* most-recently promoted Production (or now Archived) model
        versions in reverse chronological order (newest first).

        Requirement 10.5 — retain at least 2 most-recent Production versions.

        Parameters
        ----------
        n:
            Number of versions to return.

        Returns
        -------
        List[ModelArtifact]
            Up to *n* most-recent versions that were once Production-stage.
        """
        # Collect all versions that are currently Production or were previously
        # Production (now Archived) — identified by having a review_timestamp
        # and no rejection_reason (approved artifacts only).
        promoted = [
            v
            for v in self._versions
            if v.stage in (ModelStage.PRODUCTION, ModelStage.ARCHIVED)
            and v.reviewer_id is not None
            and v.rejection_reason is None
        ]
        # Sort by review_timestamp descending (newest first).
        promoted.sort(key=lambda v: v.review_timestamp or "", reverse=True)
        return promoted[:n]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_production(self) -> Optional[ModelArtifact]:
        """Return the single artifact currently in stage Production, or None."""
        production = [v for v in self._versions if v.stage == ModelStage.PRODUCTION]
        if not production:
            return None
        # There should only ever be one Production version at a time.
        return production[-1]

    def _get_version(self, version_id: str) -> ModelArtifact:
        """Return the artifact with *version_id*, raising KeyError if absent."""
        for artifact in self._versions:
            if artifact.version_id == version_id:
                return artifact
        raise KeyError(f"Model version '{version_id}' not found in registry.")
