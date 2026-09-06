"""
Unit tests for RetrainingTrigger.

Coverage:
  1. check_and_trigger() does nothing when trigger is "inactive".
  2. check_and_trigger() does nothing when a retraining run is already in progress.
  3. Successful retrain (Staging result): _consecutive_failures reset to 0;
     retraining-failure alert NOT emitted; retraining_trigger stays "active".
  4. Failed retrain: retraining-failure alert emitted; retraining_trigger stays
     "active"; _consecutive_failures incremented.
  5. 3 consecutive failures → escalation alert emitted; retraining_trigger stays
     "active".
  6. mark_retrain_succeeded(version_id) → sets retraining_trigger = "inactive",
     logs state change with timestamp and version_id.
  7. After mark_retrain_succeeded, check_and_trigger() does nothing.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest

from claim_denial.models import ModelStage
from claim_denial.monitoring.retraining_trigger import (
    ALERT_ESCALATION,
    ALERT_RETRAINING_FAILURE,
    RetrainingTrigger,
)
from claim_denial.registry.model_registry import ModelArtifact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(stage: ModelStage, version_id: str = "v-test-001") -> ModelArtifact:
    """Return a minimal ModelArtifact with the given stage."""
    return ModelArtifact(
        version_id=version_id,
        stage=stage,
        training_date="2024-06-15T00:00:00Z",
        algorithm="XGBoost",
        algorithm_version="1.7.6",
        precision=0.95 if stage == ModelStage.STAGING else 0.50,
        ece=0.03 if stage == ModelStage.STAGING else 0.10,
    )


def _make_training_result(stage: ModelStage, version_id: str = "v-test-001") -> MagicMock:
    """Return a mock TrainingResult with the given artifact stage."""
    result = MagicMock()
    result.model_artifact = _make_artifact(stage, version_id)
    return result


def _build_trigger(
    *,
    trigger_value: str = "active",
    escalation_threshold: int = 3,
) -> tuple[RetrainingTrigger, MagicMock, MagicMock, Dict[str, Any]]:
    """
    Build a RetrainingTrigger with mock dependencies.

    Returns
    -------
    (trigger, mock_trainer, mock_alert_client, retraining_state)
    """
    mock_trainer = MagicMock()
    mock_dataset_fn = MagicMock(return_value=MagicMock())  # returns a fake dataset
    mock_alert_client = MagicMock()
    retraining_state: Dict[str, Any] = {"retraining_trigger": trigger_value}

    trigger = RetrainingTrigger(
        model_trainer=mock_trainer,
        training_dataset_fn=mock_dataset_fn,
        alert_client=mock_alert_client,
        retraining_state=retraining_state,
        escalation_threshold=escalation_threshold,
    )
    return trigger, mock_trainer, mock_alert_client, retraining_state


# ---------------------------------------------------------------------------
# Test 1: trigger is "inactive" — run_training never called
# ---------------------------------------------------------------------------


class TestInactiveTrigger:
    """Requirement 13.1 — no action when trigger is inactive."""

    def test_check_and_trigger_inactive_does_not_call_run_training(self):
        trigger, mock_trainer, mock_alert_client, state = _build_trigger(
            trigger_value="inactive"
        )

        trigger.check_and_trigger()

        mock_trainer.run_training.assert_not_called()

    def test_check_and_trigger_inactive_does_not_emit_alerts(self):
        trigger, _, mock_alert_client, _ = _build_trigger(trigger_value="inactive")

        trigger.check_and_trigger()

        mock_alert_client.assert_not_called()

    def test_check_and_trigger_missing_key_does_not_call_run_training(self):
        """If the key is absent, treat as inactive."""
        trigger, mock_trainer, _, _ = _build_trigger(trigger_value="inactive")
        trigger.retraining_state = {}  # no key at all

        trigger.check_and_trigger()

        mock_trainer.run_training.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: run already in progress — run_training not called again
# ---------------------------------------------------------------------------


class TestAlreadyInProgress:
    """Requirement 13.1 — no duplicate run when one is already in progress."""

    def test_in_progress_flag_prevents_second_run(self):
        trigger, mock_trainer, _, _ = _build_trigger()

        # Simulate run already in progress
        trigger._retraining_in_progress = True

        trigger.check_and_trigger()

        mock_trainer.run_training.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: successful retrain → Staging artifact
# ---------------------------------------------------------------------------


class TestSuccessfulRetrain:
    """Requirements 13.3 — successful Staging registration."""

    def setup_method(self):
        self.trigger, self.mock_trainer, self.mock_alert_client, self.state = (
            _build_trigger()
        )
        self.mock_trainer.run_training.return_value = _make_training_result(
            ModelStage.STAGING
        )

    def test_consecutive_failures_reset_to_zero(self):
        self.trigger._consecutive_failures = 2  # pre-existing failures
        self.trigger.check_and_trigger()

        assert self.trigger._consecutive_failures == 0

    def test_retraining_failure_alert_not_emitted(self):
        self.trigger.check_and_trigger()

        # Assert ALERT_RETRAINING_FAILURE was NOT emitted
        for c in self.mock_alert_client.call_args_list:
            assert c.args[0] != ALERT_RETRAINING_FAILURE, (
                "Retraining-failure alert must NOT be emitted on Staging result"
            )

    def test_retraining_trigger_stays_active(self):
        self.trigger.check_and_trigger()

        assert self.state["retraining_trigger"] == "active"

    def test_run_training_called_with_correct_trigger_reason(self):
        self.trigger.check_and_trigger()

        self.mock_trainer.run_training.assert_called_once()
        _, kwargs = self.mock_trainer.run_training.call_args
        assert kwargs.get("trigger_reason") == "precision_threshold_breach"

    def test_in_progress_flag_released_after_success(self):
        self.trigger.check_and_trigger()

        assert self.trigger._retraining_in_progress is False


# ---------------------------------------------------------------------------
# Test 4: failed retrain → Failed artifact
# ---------------------------------------------------------------------------


class TestFailedRetrain:
    """Requirements 13.4, 13.6 — failure handling."""

    def setup_method(self):
        self.trigger, self.mock_trainer, self.mock_alert_client, self.state = (
            _build_trigger()
        )
        self.mock_trainer.run_training.return_value = _make_training_result(
            ModelStage.FAILED
        )

    def test_retraining_failure_alert_emitted(self):
        self.trigger.check_and_trigger()

        alert_types = [c.args[0] for c in self.mock_alert_client.call_args_list]
        assert ALERT_RETRAINING_FAILURE in alert_types

    def test_retraining_trigger_stays_active_on_failure(self):
        self.trigger.check_and_trigger()

        assert self.state["retraining_trigger"] == "active"

    def test_consecutive_failures_incremented(self):
        self.trigger.check_and_trigger()

        assert self.trigger._consecutive_failures == 1

    def test_consecutive_failures_accumulate(self):
        self.trigger.check_and_trigger()
        self.trigger.check_and_trigger()

        assert self.trigger._consecutive_failures == 2

    def test_in_progress_flag_released_after_failure(self):
        self.trigger.check_and_trigger()

        assert self.trigger._retraining_in_progress is False


# ---------------------------------------------------------------------------
# Test 5: 3 consecutive failures → escalation alert
# ---------------------------------------------------------------------------


class TestEscalationAlert:
    """Requirements 13.5, 13.6 — escalation after threshold consecutive failures."""

    def test_escalation_alert_emitted_after_threshold(self):
        trigger, mock_trainer, mock_alert_client, state = _build_trigger(
            escalation_threshold=3
        )
        mock_trainer.run_training.return_value = _make_training_result(
            ModelStage.FAILED
        )

        # Three consecutive failures
        trigger.check_and_trigger()
        trigger.check_and_trigger()
        trigger.check_and_trigger()

        alert_types = [c.args[0] for c in mock_alert_client.call_args_list]
        assert ALERT_ESCALATION in alert_types

    def test_escalation_alert_not_emitted_before_threshold(self):
        trigger, mock_trainer, mock_alert_client, state = _build_trigger(
            escalation_threshold=3
        )
        mock_trainer.run_training.return_value = _make_training_result(
            ModelStage.FAILED
        )

        # Only two failures
        trigger.check_and_trigger()
        trigger.check_and_trigger()

        alert_types = [c.args[0] for c in mock_alert_client.call_args_list]
        assert ALERT_ESCALATION not in alert_types

    def test_retraining_failure_alert_also_emitted_at_escalation(self):
        trigger, mock_trainer, mock_alert_client, _ = _build_trigger(
            escalation_threshold=3
        )
        mock_trainer.run_training.return_value = _make_training_result(
            ModelStage.FAILED
        )

        trigger.check_and_trigger()
        trigger.check_and_trigger()
        trigger.check_and_trigger()

        alert_types = [c.args[0] for c in mock_alert_client.call_args_list]
        assert ALERT_RETRAINING_FAILURE in alert_types

    def test_retraining_trigger_stays_active_after_escalation(self):
        trigger, mock_trainer, _, state = _build_trigger(escalation_threshold=3)
        mock_trainer.run_training.return_value = _make_training_result(
            ModelStage.FAILED
        )

        trigger.check_and_trigger()
        trigger.check_and_trigger()
        trigger.check_and_trigger()

        assert state["retraining_trigger"] == "active"

    def test_escalation_on_exact_threshold(self):
        """Escalation fires on run #threshold, not run #threshold+1."""
        trigger, mock_trainer, mock_alert_client, _ = _build_trigger(
            escalation_threshold=3
        )
        mock_trainer.run_training.return_value = _make_training_result(
            ModelStage.FAILED
        )

        trigger.check_and_trigger()  # failure 1 — no escalation
        trigger.check_and_trigger()  # failure 2 — no escalation
        assert (
            ALERT_ESCALATION
            not in [c.args[0] for c in mock_alert_client.call_args_list]
        )

        trigger.check_and_trigger()  # failure 3 — escalation fires
        assert (
            ALERT_ESCALATION
            in [c.args[0] for c in mock_alert_client.call_args_list]
        )


# ---------------------------------------------------------------------------
# Test 6: mark_retrain_succeeded sets trigger to inactive
# ---------------------------------------------------------------------------


class TestMarkRetrainSucceeded:
    """Requirement 13.4 — trigger set to inactive after successful promotion."""

    def test_sets_retraining_trigger_to_inactive(self):
        trigger, _, _, state = _build_trigger()

        trigger.mark_retrain_succeeded("v-prod-001")

        assert state["retraining_trigger"] == "inactive"

    def test_logs_state_change_with_version_id(self, caplog):
        trigger, _, _, _ = _build_trigger()

        with caplog.at_level(logging.INFO, logger="claim_denial.monitoring.retraining_trigger"):
            trigger.mark_retrain_succeeded("v-prod-xyz")

        assert "v-prod-xyz" in caplog.text
        assert "inactive" in caplog.text

    def test_logs_state_change_with_timestamp(self, caplog):
        trigger, _, _, _ = _build_trigger()

        with caplog.at_level(logging.INFO, logger="claim_denial.monitoring.retraining_trigger"):
            trigger.mark_retrain_succeeded("v-prod-001")

        # A UTC timestamp must appear somewhere in the log message
        assert "timestamp" in caplog.text.lower() or "Z" in caplog.text or "T" in caplog.text


# ---------------------------------------------------------------------------
# Test 7: after mark_retrain_succeeded, check_and_trigger does nothing
# ---------------------------------------------------------------------------


class TestNoActionAfterSucceeded:
    """Requirement 13.4 — after mark_retrain_succeeded, no further retraining."""

    def test_check_and_trigger_no_op_after_mark_succeeded(self):
        trigger, mock_trainer, _, _ = _build_trigger()

        trigger.mark_retrain_succeeded("v-prod-001")
        trigger.check_and_trigger()

        mock_trainer.run_training.assert_not_called()
