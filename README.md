# IttnasNoruen — learning under behavioral constraints

**Can a system improve a new response while preserving the useful behavior that shares its adjustable structure?**

This repository now contains a reusable software update guard, a continuous learning
benchmark, a coupled nonlinear propagation test, and a real-data classifier example.
It measures which proposed changes can be justified by a bounded record of prior
behavior, including the cases where that record is insufficient.

Start with [results and limitations](RESULTS.md) and the
[higher-level interpretation](HIGHER_LEVEL.md).

## What runs

| Experiment | What it establishes | What remains limited |
| --- | --- | --- |
| [Gates 0–3](gates.py) | expectation, delayed credit and material affecting future response | separated controls and addressed examples |
| [Gate 4](gate4_compatible_change.py) | compatible-change and delay-stability reference | supplied sensitivity matrix |
| [Gate 5](gate5_replay_coordination.py) | replay coordinates temporary plasticity without supplied P | clean diagonal-gain shortcut |
| [Gate 6](gate6_continuous_learning.py) | commands, external activity, echo learning, bounded delayed receipts, bounded replay, writes and reversal together | digital action windows and local gain model |
| [Gate 7](gate7_measured_transfer.py) | scalar measurements guide updates of a coupled nonlinear tree | external software observer with temporary candidate evaluation |
| [Digits transfer](digits_transfer.py) | the same guard changes a trained classifier on real handwritten data | ten anchors do not characterize the whole recognition skill |

The continuous proposal method reduces interference in the reference stream.
The nonlinear test exposes the limits of trusting a tangent for a finite change.
The digits example **does not establish an advantage over ordinary adaptation**:
it protects its anchors while some unseen responses change substantially.
All outcomes remain in the [numerical receipts](results/).

## A portable update interface

[behavioral_guard.py](behavioral_guard.py) accepts a callback
`response(parameters, query) -> scalar`. It does not inspect the callback's model
or receive its Jacobian. It measures finite parameter perturbations, shapes a
temporary change, and evaluates actual candidate responses before committing.
Every scalar evaluation consumes a declared budget.

```python
from behavioral_guard import BehavioralUpdateGuard, ResponseMeter

# Supply your response callback, parameters and query identifiers.
meter = ResponseMeter(response, max_calls=2000)
guard = BehavioralUpdateGuard(max_references=3, trust_radius=0.2)
for query in old_queries:
    guard.remember(query, meter(parameters, query), tolerance=0.001)

decision = guard.step(parameters, new_query, desired_response, meter)
parameters = decision.parameters
print(decision.status, decision.calls)
```

In deterministic software, accepted candidates have passed the retained response
checks. Rejection or budget exhaustion returns the original parameters.
This does not certify unseen inputs, unmodeled drift or noisy future trials.

Temporary parameter evaluation is an explicit software capability. It must not be
silently attributed to a biological synapse or an irreversible physical material.
The continuous local interface and this software guard have different privileges.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python gate6_continuous_learning.py --output results/gate6_continuous.json
python gate7_measured_transfer.py --output results/gate7_transfer.json
```

The classifier example uses handwritten digits bundled with scikit-learn; it
does not download a pretrained model or an external dataset:

```bash
python -m pip install -r requirements-examples.txt
python digits_transfer.py --output results/digits_transfer.json
python report_results.py
```

CI runs historical gates, new core benchmarks, tests, and a short real-data smoke
check on Python 3.11 and 3.12. Green engineering checks do not imply superiority
over controls. Performance and costs are reported separately.

## Three distinct jobs

| Signal | What it asks |
| --- | --- |
| Self-prediction residual | What part of activity does my copied command explain? |
| Delayed task error | What response should this particular experience improve? |
| Preservation discrepancy | Which retained answers would this change disturb? |

An efference copy is the copied command. A predictor produces an expectation.
Replay supplies a reference interaction. A discrepancy alone does not identify its
cause, and a familiar answer is not automatically permission to reinforce itself.

Fast proposal dynamics coordinate compensation before altering slow material.
The next major limitation is **reference coverage**: a finite bank can miss useful
behavior even when every stored answer is preserved. That is now measured on
unseen inputs.

## Research boundary and lineage

The projection principle is established in Kaczmarz methods, Gradient Episodic
Memory and Orthogonal Gradient Descent. See [references](HIGHER_LEVEL.md#status-and-prior-work).
We do not claim a new general continual-learning solution or a neuron implementing
this observer.

The emphasis is the interaction of changing material, learned self-prediction,
delayed feedback, bounded memory, measured compatibility and the cost of approving
an update. A continuous **coupled** substrate with physically justified prediction
and coordination pathways remains unfinished.

- [SighImageSuper](https://github.com/anttiluode/SighImageSuper): persistence, interrogation and self/world confounds.
- [GeometricNeuronOriginReview](https://github.com/anttiluode/GeometricNeuronOriginReview): spatial measurement inside nonlinear feedback.
- [Active Dendrite](https://github.com/anttiluode/OperaattoriAktiivinenDendriitti): choosing measurements that expose hidden distinctions.
- [Operaattori](https://github.com/anttiluode/Operaattori): morphology-shaped propagation and geometry sensitivities.
- [Historical Gate 5 handoff](HISTORY_THROUGH_GATE5.md), [compatibility reference](COMPATIBLE_CHANGE_REFERENCE.md), [replay mechanism](GATE5_REPLAY_COORDINATION.md), [continuous-task contract](GATE6_CONTINUOUS_TASK.md).
