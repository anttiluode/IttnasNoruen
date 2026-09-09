# Gate 9 — choose replay for the proposed change

This protocol is fixed before running Gate 9 outcomes. It follows Gate 8's missing
comparison: can choosing vulnerable replay questions protect unseen cue routes
at the same achieved new-task progress?

## Shared task and information

Use bundled handwritten digits, the same 24-hidden-unit tanh model and supplied
full/upper/lower cues as Gate 8. Use new seeds 151, 163, 179. Split test images off
before training/calibration. Train the base on all three views. Adapt 24 hidden
bias offsets to the 64-example contrast task `0.55*x + 0.22`.

Every method gets the same labeled old calibration stream and a bounded pool of
60 records: two correctly classified low-margin cues per class **and view**.
This shared storage policy deliberately represents fragile routes. It is not an
autonomously learned importance function. Each record stores its cue, label,
initial margin and required lower bound `min(0.1, initial_margin/2)`.

Most guards use 20 active references, two per class. The full-pool control uses
all 60 and pays the greater derivative-measurement cost. All methods have the
same 60-record allowance, 45,000 online scalar callbacks and 24 update attempts.
Report scalar calls and model-example evaluations separately. Selection scans,
failed proposals and candidate validation all count.

## Comparisons

- Frozen material: no adaptation.
- Unprotected: the ordinary measured update, with the fixed panel measured but
  without enforcing its bounds.
- Fixed mixed: one fragile upper and one fragile lower cue per class.
- Random scanned: random two per class after paying for the same proposal scan
  as the informed selector.
- Boundary scanned: the smallest initial margins, with the same scan expenditure.
- Interference scanned: select the lowest predicted margin relative to its
  remaining allowed margin after a temporary ordinary update.
- Full pool: all 60 constraints, without a selection scan.

For scanned methods, first make a temporary ordinary proposal using only measured
new-task responses. Measure current/proposed margins for all 60 records. Rank the
informed condition by `(bound - proposed_margin) / max(initial_margin-bound, .01)`.
The random and boundary controls ignore those scores. This tests selection beyond
the cost of extra observations. Fixed/full-pool methods may avoid that scan.

After selection, the existing interval guard measures parameter sensitivities
and validates the actual nonlinear candidate. The temporary scanning proposal
never changes material. A blocked deterministic selection stops; random selection
may retry within the common attempt/call caps.

## Matched progress and held-out access

Record checkpoints at new calibration log-probability improvements of 0.05, 0.10,
0.20 and 0.40, capped at mean log probability -0.001. A checkpoint requires actual
measured error within 1e-6; use no interpolation. Report which methods/seeds reach
it. Pairwise preservation comparisons use only common reached checkpoints and
state their coverage. A failure to reach a checkpoint is never treated as perfect
protection at that progress.

At each reached checkpoint, measure unseen-image accuracy and losses of previously
correct full/upper/lower cues. Separately test cue masks never offered to replay:
upper three rows, lower three rows, left four columns, right four columns. Report
their baseline accuracy as well as loss of initially correct responses. These
harder masks are a distribution-shift test, not guaranteed meaningful human cues.
Test labels are evaluator-only and cannot affect selection, steps or stopping.

The next policy chooses **what to replay before a write**. It does not learn how
to reconstruct a missing cue or which cue to request during recall. Those remain
different problems. Growth and continuous self/world credit are not added here.

## Claim boundary

Selection by anticipated interference is established by
[Maximally Interfered Retrieval](https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf).
This experiment combines a related margin-based selector with a measured interval
guard and explicit access/cost evaluation. It is not an exact reproduction of MIR
or a new general continual-learning method.

Kompressori motivates testing unseen responses. Its low-rank update observation
does not establish that overlap predicts harmful interference here. Gate 9 uses
measured behavior and does not claim to have recovered the response Jacobian.
Positive, null and adverse results all remain reportable outcomes.

## Recorded implementation correction after the first run

The original run is retained as `results/gate9_initial_projection.json`. Multiple
guards stopped at identical progress. A targeted seed-151 audit compared the
measured linear subproblem with SciPy SLSQP: both found the same step, with about
0.02745 actual old-margin violation despite negligible linear violation. The
projection had converged; the finite nonlinear candidate was unsafe.

The counterexample `old=b-a*a >= 0`, `new=a` at `(0,0)` isolates the missing
operation. Increasing `a` along the tangent requires a compensating change in `b`.
The guard now permits at most two measured model-error corrections per rejected
candidate, before shortening the task step. Each correction shifts the local
prediction by the observed model error, reprojects, and queries the actual
candidate again. The original acceptance bounds and budgets do not change.

This is a post-outcome numerical correction, not a preregistered positive finding.
The seeds, storage policy, selectors, tasks, held-out masks and checkpoints remain
unchanged. `--model-corrections 0` reproduces the initial algorithm's outcomes;
the final receipt uses two corrections. Gate 8 explicitly disables corrections to
retain its historical comparison.
