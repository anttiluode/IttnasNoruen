from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ittnas_noruen import IttnasNoruen


@dataclass(frozen=True)
class ProtectedReplay:
    name: str
    probe: np.ndarray
    target: float


@dataclass(frozen=True)
class ReplayProjectionStep:
    replay_name: str
    measured_response: float
    predicted_proposal_effect: float
    correction_scalar: float
    proposal_norm: float


class ReplayConstraintBank:
    """Bounded memory of protected questions and their reference answers.

    A dense N-branch probe plus one scalar target costs N+1 stored scalar values in this
    deliberately explicit representation, aside from names/bookkeeping.
    """

    def __init__(self, n_branches: int, max_items: int) -> None:
        if n_branches < 1:
            raise ValueError("n_branches must be positive")
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self.n_branches = int(n_branches)
        self.max_items = int(max_items)
        self._items: list[ProtectedReplay] = []

    def add(self, name: str, probe: Iterable[float], target: float) -> None:
        if len(self._items) >= self.max_items:
            raise OverflowError(
                f"protected replay capacity exhausted: {len(self._items)}/{self.max_items}"
            )
        arr = np.asarray(list(probe), dtype=float)
        if arr.shape != (self.n_branches,):
            raise ValueError("probe must match branch count")
        self._items.append(ProtectedReplay(str(name), arr.copy(), float(target)))

    @property
    def items(self) -> tuple[ProtectedReplay, ...]:
        return tuple(self._items)

    @property
    def stored_scalar_values(self) -> int:
        return len(self._items) * (self.n_branches + 1)


class ReplayPlasticityCoordinator:
    """Shape a temporary gain-change proposal by replaying protected responses.

    This is the concrete mechanism suggested by Astra's review. It never receives a
    protected sensitivity matrix. Instead, one replay at a time supplies local activity.
    In the present diagonal gain model, branch j can infer the local sensitivity

        a_j = c_j * residual_j / gain_j

    during a zero-command external replay. The proposal z is then corrected using only
    local a_j*z_j products, their summed scalar h, their summed squared norm, and one
    broadcast correction scalar.

    The method is deliberately model-specific. Coupled nonlinear dendrites will require
    another sensitivity-estimation mechanism.
    """

    def __init__(
        self,
        learner: IttnasNoruen,
        *,
        max_protected: int,
        gain_floor: float = 1e-9,
    ) -> None:
        self.learner = learner
        self.bank = ReplayConstraintBank(learner.n_branches, max_protected)
        self.gain_floor = float(gain_floor)
        self.replay_count = 0
        self.probe_energy = 0.0

    def add_protected(self, name: str, probe: Iterable[float], target: float) -> None:
        self.bank.add(name, probe, target)

    def measure_local_gain_sensitivity(
        self,
        probe: Iterable[float],
    ) -> tuple[np.ndarray, float]:
        """Replay one external question and recover local gain sensitivities.

        The replay uses command=0 so the measured return is not mixed with self-command
        activity. The calculation uses the learner's local material-dependent gain and
        fixed readout coupling, but not the supplied probe vector after the physical
        response has been generated.
        """

        arr = np.asarray(list(probe), dtype=float)
        if arr.shape != (self.learner.n_branches,):
            raise ValueError("probe must match branch count")

        observed = self.learner.physical_response(0.0, arr)
        gains = self.learner.branch_gain()
        if np.any(np.abs(gains) < self.gain_floor):
            raise FloatingPointError("cannot infer sensitivity through near-zero branch gain")

        # With command=0, predicted self echo is zero and residual == observed.
        residual = observed
        local_sensitivity = self.learner.readout_weights * residual / gains
        measured_response = self.learner.soma_readout(observed)

        self.replay_count += 1
        self.probe_energy += float(np.dot(arr, arr))
        return local_sensitivity, measured_response

    def proposal_from_new_question(self, probe: Iterable[float]) -> tuple[np.ndarray, float]:
        sensitivity, response = self.measure_local_gain_sensitivity(probe)
        return sensitivity.copy(), response

    def project_against_replay(
        self,
        proposal: np.ndarray,
        replay: ProtectedReplay,
    ) -> tuple[np.ndarray, ReplayProjectionStep]:
        z = np.asarray(proposal, dtype=float)
        if z.shape != (self.learner.n_branches,):
            raise ValueError("proposal must match branch count")

        sensitivity, response = self.measure_local_gain_sensitivity(replay.probe)
        denom = float(np.dot(sensitivity, sensitivity))
        if denom <= 1e-18:
            return z.copy(), ReplayProjectionStep(
                replay_name=replay.name,
                measured_response=response,
                predicted_proposal_effect=0.0,
                correction_scalar=0.0,
                proposal_norm=float(np.linalg.norm(z)),
            )

        # Local branches can form sensitivity_j * z_j. Their summed contribution is h.
        h = float(np.dot(sensitivity, z))
        correction_scalar = h / denom
        corrected = z - sensitivity * correction_scalar
        return corrected, ReplayProjectionStep(
            replay_name=replay.name,
            measured_response=response,
            predicted_proposal_effect=h,
            correction_scalar=correction_scalar,
            proposal_norm=float(np.linalg.norm(corrected)),
        )

    def shape_proposal(
        self,
        proposal: Iterable[float],
        *,
        cycles: int,
    ) -> tuple[np.ndarray, list[ReplayProjectionStep]]:
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        z = np.asarray(list(proposal), dtype=float)
        if z.shape != (self.learner.n_branches,):
            raise ValueError("proposal must match branch count")

        history: list[ReplayProjectionStep] = []
        for _ in range(cycles):
            for replay in self.bank.items:
                z, step = self.project_against_replay(z, replay)
                history.append(step)
        return z, history

    def normalize_for_question(
        self,
        proposal: Iterable[float],
        probe: Iterable[float],
    ) -> tuple[np.ndarray, float]:
        z = np.asarray(list(proposal), dtype=float)
        sensitivity, _response = self.measure_local_gain_sensitivity(probe)
        effect = float(np.dot(sensitivity, z))
        if abs(effect) < 1e-12:
            return np.zeros_like(z), effect
        return z / effect, effect

    def commit_gain_proposal(
        self,
        proposal: Iterable[float],
        *,
        task_error: float,
        learning_rate: float,
    ) -> np.ndarray:
        """Commit an error-authorized gain change after replay has shaped the proposal."""

        z = np.asarray(list(proposal), dtype=float)
        if z.shape != (self.learner.n_branches,):
            raise ValueError("proposal must match branch count")
        beta = float(self.learner.material_feedback)
        if abs(beta) < 1e-12:
            raise ValueError("material_feedback must be non-zero to commit a gain proposal")

        gain_delta = float(learning_rate) * float(task_error) * z
        # branch_gain = 1 + beta * material, so this material delta realizes gain_delta.
        self.learner.material = self.learner.material + gain_delta / beta
        return gain_delta.copy()

    def protected_response_bank(self) -> np.ndarray:
        return np.array(
            [
                self.learner.soma_response(0.0, item.probe)
                for item in self.bank.items
            ],
            dtype=float,
        )

    def protected_targets(self) -> np.ndarray:
        return np.array([item.target for item in self.bank.items], dtype=float)
