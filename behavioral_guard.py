"""Bounded, measured behavior preservation for software parameter updates.

The guard receives a scalar response callback, not a Jacobian or a model's internals.
It measures finite differences, corrects a temporary proposal using replay, and checks
the actual nonlinear candidate response before returning a commit. Candidate evaluation
is an explicit software capability; this is NOT a physically local synaptic rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

import numpy as np


class MeasurementBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Reference:
    query: Hashable
    response: float
    tolerance: float
    minimum: float | None = None
    maximum: float | None = None

    @property
    def lower(self):
        return self.response-self.tolerance if self.minimum is None else self.minimum

    @property
    def upper(self):
        return self.response+self.tolerance if self.maximum is None else self.maximum


def project_response_bounds(proposal, rows, lower, upper, radius, *, cycles=400):
    """Dykstra projection onto measured response slabs and a parameter norm ball.

    This is a bounded numerical search, not an infeasibility certificate. The
    caller must validate finite candidate responses. Correction workspace is
    (M+1)*N scalars in addition to the measured M*N rows.
    """
    z = np.asarray(proposal, dtype=float).copy()
    rows = np.asarray(rows, dtype=float).reshape(-1, z.size)
    lower, upper = np.asarray(lower), np.asarray(upper)
    corrections = np.zeros((len(rows)+1, len(z)))
    norm2 = np.einsum('ij,ij->i', rows, rows)
    for _ in range(cycles):
        previous = z.copy()
        for i, row in enumerate(rows):
            y = z + corrections[i]
            if norm2[i] > 1e-24:
                value = float(row@y)
                z = y + row*((np.clip(value,lower[i],upper[i])-value)/norm2[i])
            else:
                z = y
            corrections[i] = y-z
        y = z + corrections[-1]
        z = y*min(1., radius/max(float(np.linalg.norm(y)),1e-30))
        corrections[-1] = y-z
        values = rows@z
        feasible = (np.all(values >= lower-1e-14) and
                    np.all(values <= upper+1e-14))
        if feasible and np.linalg.norm(z-previous) <= 1e-14:
            break
    return z


class ResponseMeter:
    """Every scalar evaluation, including rejected candidates, spends one call."""

    def __init__(self, response: Callable, max_calls: int):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self._response = response
        self.max_calls = int(max_calls)
        self.calls = 0

    def __call__(self, parameters: np.ndarray, query: Hashable) -> float:
        if self.calls >= self.max_calls:
            raise MeasurementBudgetExceeded("scalar response budget exhausted")
        self.calls += 1
        # A callback cannot alter the caller's material through this argument.
        value = float(self._response(np.array(parameters, copy=True), query))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite measured response")
        return value


def replay_projection(proposal: np.ndarray, rows: np.ndarray,
                      corrections: np.ndarray | None = None, cycles: int = 30) -> np.ndarray:
    """Row-wise projection; rows must have been supplied by counted measurements.

    corrections=0 protects current responses to first order. Nonzero corrections
    also repair previously measured deviation from the retained reference.
    """
    z = np.array(proposal, dtype=float, copy=True)
    rows = np.asarray(rows, dtype=float).reshape(-1, z.size)
    rhs = np.zeros(len(rows)) if corrections is None else np.asarray(corrections, dtype=float)
    if rhs.shape != (len(rows),):
        raise ValueError("one correction is required per replay row")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(rows)) or not np.all(np.isfinite(rhs)):
        raise ValueError("proposal and replay measurements must be finite")
    for _ in range(cycles):
        for row, correction in zip(rows, rhs):
            norm2 = float(row @ row)
            if norm2 > 1e-20:
                z -= row * ((float(row @ z) - correction) / norm2)
    return z


def orthogonal_replay_projection(proposal: np.ndarray, rows: np.ndarray,
                                 corrections: np.ndarray) -> np.ndarray:
    """Orthogonalize a bounded measured sketch before correcting the proposal.

    Nearly parallel replay rows make fixed-count cyclic projection arbitrarily slow.
    Modified Gram-Schmidt (with reorthogonalization) removes this computation bottleneck.
    The additional basis costs at most the same M*N scalars as the measured row bank.
    This is central numerical coordination, not an additional biological claim.
    """
    z = np.asarray(proposal, dtype=float).copy()
    basis: list[np.ndarray] = []
    targets: list[float] = []
    for row, correction in zip(rows, corrections):
        v = np.array(row, dtype=float, copy=True)
        rhs = float(correction)
        for _ in range(2):
            for q, target in zip(basis, targets):
                coefficient = float(v@q)
                v -= coefficient*q
                rhs -= coefficient*target
        norm = float(np.linalg.norm(v))
        if norm > 1e-8*max(float(np.linalg.norm(row)),1e-12):
            basis.append(v/norm)
            targets.append(rhs/norm)
    for q, target in zip(basis, targets):
        z -= q*(float(q@z)-target)
    return z


@dataclass(frozen=True)
class UpdateDecision:
    parameters: np.ndarray
    status: str
    calls: int
    target_error_before: float | None
    target_error_after: float | None
    max_reference_drift: float | None
    step_norm: float
    rejected_candidates: int
    max_constraint_violation: float | None = None


class BehavioralUpdateGuard:
    """Retain a bounded response contract, then propose measured parameter changes.

    Capabilities: query IDs + retained responses; own parameter vector; scalar model
    evaluations at temporary parameter values; finite-difference and replay arithmetic.
    No hidden weights, analytic sensitivities, labels or test-set metrics are consulted.
    """

    def __init__(self, max_references: int, *, fd_step: float = 1e-4,
                 trust_radius: float = 0.2, max_backtracks: int = 8,
                 projection_mode: str = "bounds"):
        if max_references < 1 or fd_step <= 0 or trust_radius <= 0 or max_backtracks < 0:
            raise ValueError("invalid guard budget or step size")
        self.max_references = int(max_references)
        self.fd_step = float(fd_step)
        self.trust_radius = float(trust_radius)
        self.max_backtracks = int(max_backtracks)
        if projection_mode not in ("bounds", "equalities"):
            raise ValueError("projection_mode must be bounds or equalities")
        self.projection_mode = projection_mode
        self.references: list[Reference] = []

    def remember(self, query: Hashable, response: float, tolerance: float):
        if len(self.references) >= self.max_references:
            raise OverflowError("protected response memory is full")
        if tolerance < 0 or not np.isfinite(response) or not np.isfinite(tolerance):
            raise ValueError("reference response and tolerance must be finite")
        if any(item.query == query for item in self.references):
            raise ValueError("duplicate protected query")
        self.references.append(Reference(query, float(response), float(tolerance)))

    def remember_range(self, query: Hashable, *, minimum=-np.inf, maximum=np.inf):
        """Protect an acceptable interval, including a one-sided task margin.

        The task supplies these limits. The guard cannot infer their importance.
        """
        if self.projection_mode != "bounds":
            raise ValueError("interval contracts require bounds mode")
        if len(self.references) >= self.max_references:
            raise OverflowError("protected response memory is full")
        if (np.isnan(minimum) or np.isnan(maximum) or minimum > maximum or
                minimum == np.inf or maximum == -np.inf or
                (not np.isfinite(minimum) and not np.isfinite(maximum))):
            raise ValueError("a nonempty interval with at least one finite bound is required")
        if any(r.query == query for r in self.references):
            raise ValueError("duplicate protected query")
        anchor = (minimum+maximum)/2 if np.isfinite(minimum+maximum) else (
            minimum if np.isfinite(minimum) else maximum)
        self.references.append(Reference(query,float(anchor),float('inf'),
                                         float(minimum),float(maximum)))

    def step(self, parameters: np.ndarray, query: Hashable, target: float,
             meter: ResponseMeter, *, protect: bool = True,
             validate: bool = True) -> UpdateDecision:
        """Atomic software commit. Budget exhaustion or failed checks returns original.

        protect=False and validate=False is the equal-acquisition unprotected control.
        protect=True, validate=False isolates the cost of trusting only a tangent.
        All variants acquire the same response/sensitivity panel.
        """
        theta = np.asarray(parameters, dtype=float).copy()
        if theta.ndim != 1 or not np.all(np.isfinite(theta)) or not np.isfinite(target):
            raise ValueError("finite vector parameters and target are required")
        queries = [r.query for r in self.references] + [query]
        start = meter.calls
        before = None
        rejected = 0

        def finish(status, value=None, error_after=None, drift=None, violation=None):
            result = theta if value is None else value
            return UpdateDecision(result.copy(), status, meter.calls-start, before,
                                  error_after, drift, float(np.linalg.norm(result-theta)), rejected,
                                  violation)

        try:
            current = np.array([meter(theta, q) for q in queries])
            before = float(target-current[-1])
            current_safe = all(r.lower <= value <= r.upper
                               for r,value in zip(self.references,current[:-1]))
            if abs(before) < 1e-8 and current_safe:
                return finish("target_met", error_after=before)
            # Column-wise perturbations: no inverse model or supplied Jacobian.
            sensitivities = np.empty((len(queries), theta.size))
            for j in range(theta.size):
                perturbation = np.zeros_like(theta)
                perturbation[j] = self.fd_step
                plus = np.array([meter(theta+perturbation, q) for q in queries])
                minus = np.array([meter(theta-perturbation, q) for q in queries])
                sensitivities[:, j] = (plus-minus)/(2*self.fd_step)
            new_row = sensitivities[-1]
            norm2 = float(new_row @ new_row)
            if norm2 < 1e-18:
                return finish("no_observed_target_sensitivity")
            proposal = before * new_row / norm2
            repair_component = np.zeros_like(theta)
            if protect and self.projection_mode == "equalities":
                rhs = np.array([r.response for r in self.references])-current[:-1]
                repair = orthogonal_replay_projection(np.zeros_like(theta), sensitivities[:-1], rhs)
                free = orthogonal_replay_projection(new_row, sensitivities[:-1], np.zeros_like(rhs))
                effect = float(new_row@free)
                if effect <= 1e-18:
                    if np.linalg.norm(repair) < 1e-10:
                        return finish("no_feasible_step_observed")
                    proposal = repair
                else:
                    # Set the intended NEW response change after projection. Without
                    # this normalization, near-parallel sensitivities can make an
                    # otherwise feasible update arbitrarily small. The trust radius
                    # and actual-response validation constrain this linear prediction.
                    proposal = repair + free*((before-float(new_row@repair))/effect)
                # Keep the restoration part when limiting the new-task step. Scaling
                # both together can leave old drift pinned at the tolerance boundary.
                repair_norm = float(np.linalg.norm(repair))
                repair_component = repair * min(1., self.trust_radius/max(repair_norm,1e-20))
                freedom = proposal-repair
                free_norm = float(np.linalg.norm(freedom))
                allowance = max(0.,self.trust_radius-float(np.linalg.norm(repair_component)))
                proposal = repair_component + freedom*min(1.,allowance/max(free_norm,1e-20))
            size = float(np.linalg.norm(proposal))
            if size < 1e-10:
                return finish("no_feasible_step_observed")
            proposal *= min(1., self.trust_radius/size)
            for backtrack in range(self.max_backtracks+1):
                step = repair_component + (0.5**backtrack)*(proposal-repair_component)
                if protect and self.projection_mode == "bounds":
                    desired_effect = before*(0.5**backtrack)
                    step = project_response_bounds(
                        step, sensitivities,
                        np.r_[np.array([r.lower for r in self.references])-current[:-1],desired_effect],
                        np.r_[np.array([r.upper for r in self.references])-current[:-1],desired_effect],
                        self.trust_radius)
                    if np.linalg.norm(step) < 1e-10:
                        return finish("no_feasible_step_observed")
                candidate = theta + step
                # These are actual callback responses, not predictions from the tangent.
                observed = np.array([meter(candidate, q) for q in queries])
                errors = np.abs(observed[:-1]-np.array([r.response for r in self.references]))
                violations = [max(0.,r.lower-value,value-r.upper)
                              for r,value in zip(self.references,observed[:-1])]
                # Only floating-point comparison slack; it is not a task tolerance.
                safe = all(v <= 1e-12 for v in violations)
                after = float(target-observed[-1])
                improves = abs(after) < abs(before)-1e-12
                if improves and (safe or not validate):
                    return finish("accepted", candidate, after, float(max(errors, default=0.)),
                                  float(max(violations,default=0.)))
                rejected += 1
            return finish("no_acceptable_step_observed")
        except MeasurementBudgetExceeded:
            return finish("measurement_budget_exhausted")
