"""
RetrainingTrigger — automated retraining orchestration for the Claim Denial
Prediction system.

Responsibilities
----------------
- Check the shared ``retraining_state`` dict and call ``model_trainer.run_training``
  when ``retraining_trigger == "active"`` and no run is already in progress
  (Requirements 13.1, 13.2).
- Track consecutive training failures and emit escalation alerts after
  ``escalation_threshold`` (default 3) consecutive ``Failed`` registrations
  (Requirements 13.3, 13.5).
- Emit a retraining-failure alert when a model registers as ``Failed``
  (Requirement 13.6).
- Provide ``mark_retrain_succeeded(version_id)`` to set the trigger back to
  ``"inactive"`` after a successful Production promotion (Requirement 13.4).
- Do NOT promote models — promotion is a separate human step via
  ``ModelRegistry.approve_model``.

All patient data is synthetic; no real PHI is processed here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from claim_denial.models import ModelStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert type constants
# ---------------------------------------------------------------------------

ALERT_RETRAINING_FAILURE = "retraining_failure"
ALERT_ESCALATION = "escalation"


# ---------------------------------------------------------------------------
# RetrainingTrigger
# ---------------------------------------------------------------------------


class RetrainingTrigger:
    """
    Automated retraining orchestrator for the Claim Denial Prediction system.

    Parameters
    ----------
    model_trainer:
        A ``ModelTrainer`` instance whose ``run_training(training_dataset,
        trigger_reason=...)`` method is called when retraining is initiated.
    training_dataset_fn:
        ``Callable[[], TrainingDataset]`` — called on demand to produce the
        latest training dataset.
    alert_client:
        ``Callable[[str, str], None]`` — receives ``(alert_type, message)``
        calls for retraining-failure and escalation alerts.
    retraining_state:
        Shared mutable dict with a ``"retraining_trigger"`` key.  This is
        the same dict that ``MonitoringService`` writes
        ``retraining_trigger = "active"`` into.
    model_registry:
        Optional ``ModelRegistry`` for querying the Production model (not
        used for promotion — promotion is a separate human step).
    escalation_threshold:
        Number of consecutive ``Failed`` registrations before an escalation
        alert is emitted (default 3, per Requirement 13.6).
    """

    def __init__(
        self,
        model_trainer,
        training_dataset_fn: Callable[[], Any],
        alert_client: Callable[[str, str], None],
        retraining_state: Dict[str, Any],
        model_registry=None,
        escalation_threshold: int = 3,
    ) -> None:
        self._model_trainer = model_trainer
        self._training_dataset_fn = training_dataset_fn
        self._alert_client = alert_client
        self.retraining_state = retraining_state
        self._model_registry = model_registry
        self._escalation_threshold = escalation_threshold

        # Internal state
        self._retraining_in_progress: bool = False
        self._consecutive_failures: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_trigger(self) -> None:
        """
        Check the retraining trigger and initiate a training run if active.

        Behaviour
        ---------
        - If ``retraining_state["retraining_trigger"] != "active"``, do nothing.
        - If a run is already in progress (``_retraining_in_progress == True``),
          log and do nothing.
        - Otherwise: obtain a fresh training dataset via ``training_dataset_fn``,
          call ``model_trainer.run_training(trigger_reason=
          "precision_threshold_breach")``, and update consecutive-failure
          counters and emit alerts based on the resulting ``ModelArtifact.stage``.

        Requirements: 13.1, 13.2, 13.3, 13.5, 13.6
        """
        trigger = self.retraining_state.get("retraining_trigger")

        if trigger != "active":
            logger.info(
                "check_and_trigger: retraining_trigger=%r — skipping (not active).",
                trigger,
            )
            return

        if self._retraining_in_progress:
            logger.info(
                "check_and_trigger: a retraining run is already in progress — "
                "skipping duplicate trigger."
            )
            return

        logger.info(
            "check_and_trigger: retraining_trigger is 'active' — initiating "
            "retraining run."
        )

        self._retraining_in_progress = True
        try:
            training_dataset = self._training_dataset_fn()
            result = self._model_trainer.run_training(
                training_dataset,
                trigger_reason="precision_threshold_breach",
            )
            artifact = result.model_artifact

            logger.info(
                "check_and_trigger: training completed — version_id=%s stage=%s.",
                artifact.version_id,
                artifact.stage.value,
            )

            self._handle_training_result(artifact)

        finally:
            self._retraining_in_progress = False

    def mark_retrain_succeeded(self, version_id: str) -> None:
        """
        Mark retraining as succeeded after a successful Production promotion.

        Sets ``retraining_state["retraining_trigger"] = "inactive"`` and
        logs the state change with a UTC timestamp and the promoted version.

        This method is intended to be called by the workflow orchestrator
        (e.g., Airflow) after ``model_registry.approve_model(version_id, ...)``
        has been executed externally.

        Requirement: 13.4, 13.5

        Parameters
        ----------
        version_id:
            The version identifier of the model that was promoted to Production.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        self.retraining_state["retraining_trigger"] = "inactive"
        logger.info(
            "mark_retrain_succeeded: retraining_trigger set to 'inactive' — "
            "version_id=%s timestamp=%s.",
            version_id,
            timestamp,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_training_result(self, artifact) -> None:
        """
        Inspect the registered artifact's stage and update internal counters /
        emit alerts accordingly.

        - ``ModelStage.FAILED``: increment ``_consecutive_failures``, emit
          retraining-failure alert; keep ``retraining_trigger = "active"``.
          If ``_consecutive_failures >= _escalation_threshold``, also emit
          an escalation alert (Req 13.6).
        - ``ModelStage.STAGING`` or ``ModelStage.PRODUCTION``: reset
          ``_consecutive_failures`` to 0; no retraining-failure alert.
          The ``retraining_trigger`` remains ``"active"`` until
          ``mark_retrain_succeeded`` is called externally after promotion.

        Requirements: 13.3, 13.5, 13.6
        """
        stage = artifact.stage

        if stage == ModelStage.FAILED:
            self._consecutive_failures += 1
            logger.warning(
                "_handle_training_result: model %s registered as Failed "
                "(consecutive_failures=%d).",
                artifact.version_id,
                self._consecutive_failures,
            )

            # Emit retraining-failure alert (Req 13.4 / 13.6)
            failure_message = (
                f"Retraining run produced a Failed model "
                f"(version_id={artifact.version_id}, "
                f"consecutive_failures={self._consecutive_failures}). "
                f"Existing Production model is unchanged. "
                f"retraining_trigger remains 'active'."
            )
            self._alert_client(ALERT_RETRAINING_FAILURE, failure_message)
            logger.info(
                "_handle_training_result: retraining-failure alert emitted "
                "for version_id=%s.",
                artifact.version_id,
            )

            # Escalation check (Req 13.5 / 13.6)
            if self._consecutive_failures >= self._escalation_threshold:
                escalation_message = (
                    f"Sustained model degradation: {self._consecutive_failures} "
                    f"consecutive retraining runs have produced Failed models. "
                    f"Manual investigation required. "
                    f"Latest failed version_id={artifact.version_id}. "
                    f"Existing Production model is unchanged."
                )
                self._alert_client(ALERT_ESCALATION, escalation_message)
                logger.warning(
                    "_handle_training_result: escalation alert emitted — "
                    "%d consecutive failures (threshold=%d).",
                    self._consecutive_failures,
                    self._escalation_threshold,
                )

            # retraining_trigger stays "active" — do NOT change it here.

        elif stage in (ModelStage.STAGING, ModelStage.PRODUCTION):
            self._consecutive_failures = 0
            logger.info(
                "_handle_training_result: model %s registered as %s — "
                "consecutive_failures reset to 0. "
                "retraining_trigger remains 'active' until "
                "mark_retrain_succeeded() is called after Production promotion.",
                artifact.version_id,
                stage.value,
            )

        else:
            # Archived or any other stage — log but take no special action.
            logger.info(
                "_handle_training_result: model %s registered as %s — "
                "no counter updates.",
                artifact.version_id,
                stage.value,
            )
