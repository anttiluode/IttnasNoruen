"""Render Gate 9 comparisons while retaining failed progress checkpoints."""
import json
from pathlib import Path
from statistics import mean


def render(data, initial):
    seeds=data["protocol"]["seeds"]
    methods=[r["method"] for r in data["specimens"][0]["runs"]]
    rows=["# Gate 9 — selecting replay for the proposed change", "",
          "Generated from [the final receipt](results/gate9_selected_replay.json) "
          "and [the initial receipt](results/gate9_initial_projection.json). "
          "See [protocol, resources and the recorded correction](GATE9_PROTOCOL.md).", "",
          f"Seeds: {seeds}; one dataset, resampled splits. These are descriptive "
          "comparisons, not independent datasets or a superiority significance test.", "",
          "## Did each method actually reach the requested progress?", "",
          "A checkpoint is an actual measured new-task log-probability improvement "
          "within 1e-6 of its target. Missing checkpoints stay missing. No interpolation.", "",
          "| Method | +0.05 | +0.10 | +0.20 | +0.40 |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for method in methods:
        group=[r for s in data["specimens"] for r in s["runs"] if r["method"]==method]
        counts=[sum(any(p["requested_progress"]==g for p in r["checkpoints"]) for r in group)
                for g in data["protocol"]["checkpoints"]]
        rows.append("| `"+method+"` | "+" | ".join(f"{n}/{len(seeds)}" for n in counts)+" |")
    rows += ["", "## Informed selection versus controls at matched progress", "",
             "Each row uses only seeds on which **both** methods reached the checkpoint. "
             "Negative loss differences favor informed selection. Differences are "
             "percentage points in loss of initially correct cue responses. 'Unseen' "
             "means masks never offered to replay, on held-out images. A one-seed "
             "comparison cannot stand in for the full cohort.", "",
             "| Progress | Comparator | Common seeds | Seen-cue loss difference | Unseen-mask loss difference | Extra scalar calls |",
             "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for p in data["paired"]:
        seen="—" if p["seen_loss_difference"] is None else f"{100*p['seen_loss_difference']:+.3f}"
        unseen="—" if p["unseen_loss_difference"] is None else f"{100*p['unseen_loss_difference']:+.3f}"
        calls="—" if p["call_difference"] is None else f"{p['call_difference']:+.0f}"
        rows.append(f"| {p['requested_progress']:.2f} | `{p['comparator']}` | "
                    f"{p['coverage']}/{p['total_seeds']} | {seen} | {unseen} | {calls} |")
    rows += ["", "## Final states have unequal progress", "",
             "This table exposes costs and failure, rather than ranking protection "
             "at equal learning. Pool losses refer to the 60 stored cues; selected "
             "constraints and protection of the entire pool are different.", "",
             "| Method | Mean progress | Scalar calls | Example evaluations | Mean stored-cue losses | Mean held-out seen-cue losses | Mean unseen-mask losses |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in methods:
        group=[r for s in data["specimens"] for r in s["runs"] if r["method"]==method]
        rows.append(f"| `{method}` | {mean(r['actual_progress'] for r in group):.6f} | "
            f"{mean(r['calls'] for r in group):.0f} | {mean(r['model_example_evaluations'] for r in group):.0f} | "
            f"{mean(r['final_pool_recognition_losses'] for r in group):.2f} | "
            f"{mean(r['final']['seen']['losses'] for r in group):.2f} | "
            f"{mean(r['final']['unseen']['losses'] for r in group):.2f} |")
    rows += ["", "## Before and after measured nonlinear correction", "",
             "The initial algorithm had no model-error corrections. The final one "
             "allows two per rejected candidate, charges those queries, and enforces "
             "the same actual response bounds. This change followed an audit of the "
             "first outcomes and is explicitly post hoc.", "",
             "| Seed | Method | Initial progress | Corrected progress | Initial calls | Corrected calls | Final stop |",
             "| ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    old={(s["seed"],r["method"]):r for s in initial["specimens"] for r in s["runs"]}
    for s in data["specimens"]:
        for r in s["runs"]:
            if r["method"] not in ("fixed_mixed","interference_scanned","all_pool"):
                continue
            before=old[s["seed"],r["method"]]
            rows.append(f"| {s['seed']} | `{r['method']}` | {before['actual_progress']:.6f} | "
                        f"{r['actual_progress']:.6f} | {before['calls']} | {r['calls']} | `{r['stop_reason']}` |")
    rows += ["", "Engineering checks: "+", ".join(f"`{k}={v}`" for k,v in data["checks"].items())+".", "",
             "These checks establish bookkeeping and observed constraint compliance. "
             "They do not require an accuracy advantage or certify unmeasured behavior.", ""]
    return "\n".join(rows)


if __name__=="__main__":
    root=Path(__file__).resolve().parent
    data=json.loads((root/"results/gate9_selected_replay.json").read_text())
    initial=json.loads((root/"results/gate9_initial_projection.json").read_text())
    (root/"SELECTED_REPLAY_RESULTS.md").write_text(render(data,initial))
