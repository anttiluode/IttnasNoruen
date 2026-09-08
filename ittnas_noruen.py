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
    branch_gain: np.ndarray


class IttnasNoruen:
    """Small integration model for expectation, eligibility and slow material.

    This is deliberately not a biophysical neuron. Each branch-like compartment gets:
    a copied outgoing command, its own observed activity, a learned local echo predictor,
    local eligibility, and access to a delayed signed teaching signal.

    Two credit mechanisms coexist deliberately:

    * ``eligibility`` is the original aggregate trace and remains as an attacker/reference.
    * receipt traces are counted temporal/action records. A receipt stores the residuals
      accumulated during a declared causal window and may wait for consequence without
      being erased by another receipt's consequence.

    A receipt never contains a causal branch label. It only identifies a declared window
    or action whose delayed teaching signal may arrive later. If the experiment cannot
    delimit such windows, the ambiguity is not solved by this class.
    """

    def __init__(
        self,
        true_self_signature: Iterable[float],
        *,
        echo_enabled: bool = True,
        echo_lr: float = 0.25,
        eligibility_decay: float = 0.8,
        memory_lr: float = 0.1,
        material_feedback: float = 0.5,
        readout_weights: Iterable[float] | None = None,
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
        self.material_feedback = float(material_feedback)

        if readout_weights is None:
            weights = np.ones_like(signature) / signature.size
        else:
            weights = np.asarray(list(readout_weights), dtype=float)
            if weights.shape != signature.shape:
                raise ValueError("readout_weights must match the branch count")
        self.readout_weights = weights.copy()

        self.echo_weights = np.zeros_like(signature)
        self.eligibility = np.zeros_like(signature)
        self.material = np.zeros_like(signature)

        self._open_receipts: dict[str, np.ndarray] = {}
        self._pending_receipts: dict[str, np.ndarray] = {}

    @property
    def n_branches(self) -> int:
        return int(self.true_self_signature.size)

    @property
    def pending_receipt_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending_receipts))

    def set_true_self_signature(self, new_signature: Iterable[float]) -> None:
        new = np.asarray(list(new_signature), dtype=float)
        if new.shape != self.true_self_signature.shape:
            raise ValueError("new signature must match the existing branch count")
        self.true_self_signature = new.copy()

    def branch_gain(self) -> np.ndarray:
        """Return the material-dependent gain used by the toy physical response.

        The relation is intentionally simple, not a biophysical conductance law. Its job
        is to close the loop that the first integration control left open: writing material
        must alter later activity, which can make the self predictor stale.
        """

        return 1.0 + self.material_feedback * self.material

    def physical_response(
        self,
        command: float,
        external: Iterable[float] | None = None,
    ) -> np.ndarray:
        """Pure material-shaped response with no predictor or eligibility update."""

        u = float(command)
        if external is None:
            ext = np.zeros(self.n_branches, dtype=float)
        else:
            ext = np.asarray(list(external), dtype=float)
            if ext.shape != self.true_self_signature.shape:
                raise ValueError("external must match the branch count")
        drive = self.true_self_signature * u + ext
        return self.branch_gain() * drive

    def soma_readout(self, branch_activity: Iterable[float]) -> float:
        """Fixed bounded readout of branch activity.

        Slow material affects this answer indirectly by changing ``physical_response``.
        The readout itself is fixed so inspecting ``w`` is not the behavioral metric.
        """

        activity = np.asarray(list(branch_activity), dtype=float)
        if activity.shape != self.material.shape:
            raise ValueError("branch_activity must match the branch count")
        return float(np.dot(self.readout_weights, activity))

    def soma_response(
        self,
        command: float,
        external: Iterable[float] | None = None,
    ) -> float:
        return self.soma_readout(self.physical_response(command, external))

    def clear_eligibility(self) -> None:
        self.eligibility.fill(0.0)

    def predict_self(self, command: float) -> np.ndarray:
        if not self.echo_enabled:
            return np.zeros(self.n_branches, dtype=float)
        return self.echo_weights * float(command)

    def begin_receipt(self, receipt_id: str) -> None:
        """Open a counted causal window without supplying a branch label."""

        key = str(receipt_id)
        if key in self._open_receipts or key in self._pending_receipts:
            raise ValueError(f"receipt already exists: {key}")
        self._open_receipts[key] = np.zeros(self.n_branches, dtype=float)

    def seal_receipt(self, receipt_id: str) -> np.ndarray:
        """Stop accumulating into a receipt while retaining it for delayed teaching."""

        key = str(receipt_id)
        if key not in self._open_receipts:
            raise KeyError(f"receipt is not open: {key}")
        trace = self._open_receipts.pop(key)
        self._pending_receipts[key] = trace.copy()
        return trace.copy()

    def receipt_trace(self, receipt_id: str) -> np.ndarray:
        key = str(receipt_id)
        if key in self._open_receipts:
            return self._open_receipts[key].copy()
        if key in self._pending_receipts:
            return self._pending_receipts[key].copy()
        raise KeyError(f"unknown receipt: {key}")

    def observe(
        self,
        command: float,
        external: Iterable[float] | None = None,
        *,
        learn_echo: bool = True,
    ) -> Observation:
        """Observe one local response and update predictor/temporary traces.

        ``external`` is used by the experimenter to generate the world response. The
        learner is not told an event label or branch identity; it only receives the summed
        local activity. Receipt membership, when used, is supplied separately as a
        declared temporal/action boundary.
        """

        u = float(command)
        observed = self.physical_response(u, external)
        predicted = self.predict_self(u)
        residual = observed - predicted

        self.eligibility = self.eligibility_decay * self.eligibility + residual

        # A sealed receipt is still a decaying trace while time passes. An open receipt
        # both decays and accumulates this tick's residual.
        for key in tuple(self._pending_receipts):
            self._pending_receipts[key] *= self.eligibility_decay
        for key in tuple(self._open_receipts):
            self._open_receipts[key] = (
                self.eligibility_decay * self._open_receipts[key] + residual
            )

        if self.echo_enabled and learn_echo:
            # One local LMS coefficient per branch. Correlated external activity can be
            # absorbed into this predictor; later gates retain that as an explicit attack.
            self.echo_weights = self.echo_weights + self.echo_lr * u * residual

        return Observation(
            command=u,
            observed=observed.copy(),
            predicted_self=predicted.copy(),
            residual=residual.copy(),
            eligibility=self.eligibility.copy(),
            branch_gain=self.branch_gain().copy(),
        )

    def quiet(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            self.eligibility *= self.eligibility_decay
            for key in tuple(self._open_receipts):
                self._open_receipts[key] *= self.eligibility_decay
            for key in tuple(self._pending_receipts):
                self._pending_receipts[key] *= self.eligibility_decay

    def apply_consequence(self, delta: float, *, clear_trace: bool = True) -> np.ndarray:
        """Legacy aggregate credit: apply a signed teaching signal to all eligibility.

        This intentionally remains available because its failure under overlapping causes
        is part of Gate 3. ``delta`` is a teaching signal here, not an undifferentiated raw
        reward. Calling this method is the write-permission event.
        """

        update = self.memory_lr * float(delta) * self.eligibility
        self.material = self.material + update
        if clear_trace:
            self.clear_eligibility()
        return update.copy()

    def apply_receipt_consequence(self, receipt_id: str, delta: float) -> np.ndarray:
        """Apply delayed teaching to one retained temporal/action receipt.

        The caller identifies which declared action/window received feedback, not which
        branch caused the outcome. Other pending receipts survive for later consequences.
        """

        key = str(receipt_id)
        if key not in self._pending_receipts:
            raise KeyError(f"receipt is not pending: {key}")
        trace = self._pending_receipts.pop(key)
        update = self.memory_lr * float(delta) * trace
        self.material = self.material + update
        return update.copy()

    def calibrate_self(self, commands: Iterable[float]) -> None:
        """Declared self-only calibration used by the historical control gates."""

        for command in commands:
            self.observe(float(command), external=None, learn_echo=True)
        self.clear_eligibility()

    def self_prediction_error(self, command: float = 1.0) -> float:
        u = float(command)
        target = self.branch_gain() * self.true_self_signature * u
        predicted = self.predict_self(u)
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
