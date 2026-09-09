# Gate 8 — protect the ability to reach an answer

A full question can keep working while an incomplete cue stops working. A useful
alternative cue can also remain available while the retrieval policy fails to ask
for it. Gate 8 separates these failures, then asks whether a bounded replay bank
or extra adjustable structure helps. [Results](ACCESS_RESULTS.md) retain the
negative outcomes.

The face/name example motivates the distinction. The actual experiment recognizes
handwritten digits from supplied image fragments. It does not reconstruct an
episodic question or model biological structural growth.

| Object | Implemented measurement | Remaining assumption |
| --- | --- | --- |
| Full-cue response | classification of a complete 8×8 digit | the complete cue is supplied |
| Access through a fragment | classification from upper or lower half alone | the cue transformations are supplied |
| Search for another route | request lower half when the upper response has a small probability gap | confidence can be wrong; the policy is fixed |

An access route here is an input/response relation, not an isolated anatomical
path. Both fragments pass through shared weights. Changing the cue can recover
useful behavior with those weights frozen. Growth is a separate proposal for
adding capacity; this experiment does not identify the mechanism of human recall.

## Repair the preservation contract

The earlier guard projected changes into an exact preservation subspace even when
its final check allowed tolerance. That could unnecessarily forbid learning.
Four independent protected coordinates with nonzero tolerances do not imply that
all acceptable motion is impossible.

The regression example starts at four ones, permits each coordinate to move by
0.1, and asks `g[0] + g[2]` to change from 2 to 2.1. Exact projection refuses it.
The bounded guard completes the change with maximum coordinate drift 0.05.

The default guard now searches within measured response intervals:

```math
l_i-f_\theta(q_i)\leq a_i^\top\Delta\leq u_i-f_\theta(q_i),
\qquad \|\Delta\|_2\leq\rho.
```

A bounded Dykstra projection shapes the proposal. Actual nonlinear responses
decide whether it is accepted. Failed finite search is not an infeasibility
certificate. Historical Gate 7 and the first digits experiment explicitly retain
`projection_mode="equalities"` so their published comparisons remain reproducible.

For recognition, a positive correct-class margin can be protected with
`guard.remember_range(query_id, minimum=required_margin)`. The task supplies the
label and acceptable bound. The guard does not discover importance by itself.

## Protocol

Data are the [digits bundled with scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
Seeds 113/127/139 train a 24-unit tanh classifier on full, upper and lower views.
Test examples are separated before training and calibration. Split hashes,
versions and training warnings are recorded. These are three resampled splits of
one dataset, not independent datasets.

All adaptive methods receive the same old calibration stream, labels, potential
cue types, new 64-example task and 45,000 scalar-callback allowance. There are at
most 12 attempts. The new task requests an increase of 0.2 in mean correct-label
log probability under contrast `0.55*x + 0.22`. Test accuracy is reported separately;
it never chooses banks, updates, growth, settings or stopping.

Each learner retains 20 cues, two per class:

- `exact_full`: full cues, original margins within 0.0001, legacy exact projection.
- `margin_full`: the same full cues, positive lower bounds.
- `margin_mixed`: one upper and one lower cue per class, positive lower bounds.
- `margin_boundary`: the two smallest positive baseline margins per class, across
  all three views. This is static selection, not adaptive interference retrieval.

Margin bounds are `min(0.1, baseline_margin / 2)`. Frozen and unprotected controls
receive the mixed bank; unprotected adaptation still pays for the same measurement
panel. Each selector uses the same calibration stream and bounded storage. Labels
are available during selection, not during held-out retrieval decisions.

`grow_mixed` can allocate four tanh features and 40 output weights after its first
blocked proposal, once, if another full attempt fits the remaining budget. Zero
initial output weights make allocation function-preserving. `large_mixed` has
exactly those extra features from the start of adaptation. Both start from the
same pretrained base; pretraining a larger body is not tested.

Growth did not trigger in the three full runs. The larger control did run, but
triggered growth under stronger interference remains untested. The task was not
tuned until growth won.

## Retrieval and the frozen-material control

The policy sees upper probabilities and asks lower only when the top two differ
by less than 0.2. If it asks, it averages the two probability vectors. No correct
test label enters the policy. An always-two control pays for both queries.

The experimenter computes all candidate responses on the same frozen model.
Reported query counts are responses actually used by each policy; the lower
response does not influence the request decision. Lower-only accuracy reports
what a supplied alternative can do on its own.

A control blanks the upper cue while holding all material fixed. This is an input
channel failure, not a synaptic lesion. The policy may stop at a confidently wrong
blank-input response. Availability of another cue and choosing to use it are
separate capabilities.

Paired diagnostics count full-correct/partial-lost cases after adaptation. These
metrics were added after the first seed pilot, without changing learner settings.
A correct alternative identified using test labels is an evaluator-only
opportunity, not an achieved autonomous rescue.

## Costs and capabilities

The base model has 1,810 numerical parameters. Each bank stores 1,280 input values,
20 labels, 40 bounds/reference values and identifiers. Initial adaptation uses 24
bias offsets. Extra capacity adds 260 fixed values and 40 adjustable values,
raising the adjustable vector to 64. Each streaming selector holds at most 21 cues
while considering the next candidate.

Old core encodings cost 480 values. The new calibration task stores 4,096 inputs,
64 labels and 1,536 core-encoding values. Derivative rows cost at most 21×64 and
Dykstra corrections 22×64 values, plus temporary vectors and array arithmetic.
Candidate evaluation, pretrained weights, cue generators and labels remain
external software capabilities.

A new-task scalar callback evaluates 64 examples; an old-cue callback evaluates
one. Both scalar calls and example evaluations are reported. Initial calibration,
shared base training and experimenter-only scoring are separate from online
adaptation. The resource record does not pretend to be a complete Python heap count.

## What follows

The guard can preserve every retained cue while missing an unrepresented route.
The boundary bank loses fewer routes while achieving less new-task improvement;
a stronger claim requires matched achieved learning, not just a common allowance.

A next retrieval policy can learn when to try another supplied cue using
calibration data and query cost, then freeze before testing. Constructing useful
cues from fragments remains a further problem. These are distinct from adding
structure and from predicting self-generated activity.

Relevant prior work: [Gradient Episodic Memory](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html),
[Orthogonal Gradient Descent](https://proceedings.mlr.press/v108/farajtabar20a.html),
[Maximally Interfered Retrieval](https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf),
and, for the broader cue-dependent memory-search question, the
[Context Maintenance and Retrieval model](https://pmc.ncbi.nlm.nih.gov/articles/PMC2630591/).
The static boundary selector here is not an implementation of MIR.
