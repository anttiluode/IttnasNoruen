from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Observation:
    command: float
    observed: np.ndarray
    predicted_self: np.ndarray
    residual: np.ndarray
    eligibility: np.ndarray


class IttnasNoruen:
    """Small integration model for expectation, eligibility and slow material.

    This is deliberately not a biophysical neuron. Each branch-like compartment gets:
    a copied outgoing command, its own observed activity, a learned local echo predictor,
    a local eligibility trace, and access to a delayed scalar consequence.

    The important accounting rule is that prediction learning and structural learning are
    separate. Prediction error can update the fast echo model, but slow material changes
    only when ``apply_consequence`` is called.
    """

    def __init__(
        self,
        true_self_signature: Iterable[float],
        *,
        echo_enabled: bool = True,
        echo_lr: float = 0.25,
        eligibility_decay: float = 0.8,
        memory_lr: float = 0.1,
    ) -> None:
        signature = np.asarray(list(true_self_signature), dtype=float)
        if signature.ndim != 1 or signature.size == 0:
            raise ValueError("true_self_signature must be a non-empty 1D vector")
        if not 0.0 <= eligibility_decay <= 1.0:
            raise ValueError("eligibility_decay must be in [0, 1]")

        self.true_self_signature = signature.copy()
        self.echo_enabled = bool(echo_enabled)
        self.echo_lr = float(echo_lr)
        self.eligibility_decay = float(eligibility_decay)
        self.memory_lr = float(memory_lr)

        self.echo_weights = np.zeros_like(signature)
        self.eligibility = np.zeros_like(signature)
        self.material = np.zeros_like(signature)

    @property
    def n_branches(self) -> int:
        return int(self.true_self_signature.size)

    def set_true_self_signature(self, new_signature: Iterable[float]) -> None:
        new = np.asarray(list(new_signature), dtype=float)
        if new.shape != self.true_self_signature.shape:
            raise ValueError("new signature must match the existing branch count")
        self.true_self_signature = new.copy()

    def clear_eligibility(self) -> None:
        self.eligibility.fill(0.0)

    def predict_self(self, command: float) -> np.ndarray:
        if not self.echo_enabled:
            return np.zeros(self.n_branches, dtype=float)
        return self.echo_weights * float(command)

    def observe(
        self,
        command: float,
        external: Iterable[float] | None = None,
        *,
        learn_echo: bool = True,
    ) -> Observation:
        """Observe one local response and update the fast/temporary states.

        ``external`` is used by the experimenter to generate the world response. The
        learner is not told an event label or branch identity; it only receives the summed
        local activity returned in ``observed``.

        Echo learning should be disabled for a deliberately isolated test event when a
        gate wants to measure the already-calibrated predictor. Later gates remove that
        clean phase separation.
        """

        u = float(command)
        if external is None:
            ext = np.zeros(self.n_branches, dtype=float)
        else:
            ext = np.asarray(list(external), dtype=float)
            if ext.shape != self.true_self_signature.shape:
                raise ValueError("external must match the branch count")

        self_generated = self.true_self_signature * u
        observed = self_generated + ext
        predicted = self.predict_self(u)
        residual = observed - predicted

        self.eligibility = self.eligibility_decay * self.eligibility + residual

        if self.echo_enabled and learn_echo:
            # Local one-parameter LMS update at each branch. With self-only calibration,
            # residual is exactly the local self-prediction error. If external activity is
            # correlated with the command this rule can become confounded; that is a
            # deliberate later attacker, not hidden here.
            self.echo_weights = self.echo_weights + self.echo_lr * u * residual

        return Observation(
            command=u,
            observed=observed.copy(),
            predicted_self=predicted.copy(),
            residual=residual.copy(),
            eligibility=self.eligibility.copy(),
        )

    def quiet(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            self.eligibility *= self.eligibility_decay

    def apply_consequence(self, delta: float, *, clear_trace: bool = True) -> np.ndarray:
        """Apply a delayed scalar consequence to the currently surviving eligibility."""

        update = self.memory_lr * float(delta) * self.eligibility
        self.material = self.material + update
        if clear_trace:
            self.clear_eligibility()
        return update.copy()

    def calibrate_self(self, commands: Iterable[float]) -> None:
        """Declared self-only calibration used by the first gates."""

        for command in commands:
            self.observe(float(command), external=None, learn_echo=True)
        self.clear_eligibility()

    def self_prediction_error(self, command: float = 1.0) -> float:
        target = self.true_self_signature * float(command)
        predicted = self.predict_self(float(command))
        return float(np.linalg.norm(target - predicted))

    def target_fraction(self, target_branch: int) -> float:
        total = float(np.sum(np.abs(self.material)))
        if total == 0.0:
            return 0.0
        return float(abs(self.material[target_branch]) / total)

    def off_target_norm(self, target_branch: int) -> float:
        mask = np.ones(self.n_branches, dtype=bool)
        mask[target_branch] = False
        return float(np.linalg.norm(self.material[mask]))

    def soma_readout(self, branch_activity: Iterable[float]) -> float:
        activity = np.asarray(list(branch_activity), dtype=float)
        if activity.shape != self.material.shape:
            raise ValueError("branch_activity must match the branch count")
        return float(np.dot(self.material, activity))
