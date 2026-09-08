# Gate 4 handoff

This branch turns Astra's latest review into an executable reference instead of another prose-only claim.

- `gate4_compatible_change.py` computes an exact preserving direction for overlapping responses, includes a full-rank impossibility control, compares a naive active-only update, and checks delayed-feedback stability boundaries.
- `COMPATIBLE_CHANGE_REFERENCE.md` documents the exact conditions: distinguishability, history, compatibility and stability.
- `IttnasNoruen(max_receipts=K)` now makes outstanding receipt memory explicitly bounded.
- `GATE5_CONTINUOUS_TASK.md` specifies the next non-privileged experiment: same information budget, no supplied protected-sensitivity matrix, one continuous stream with material feedback, echo learning, overlap and delayed feedback.

The central limitation is deliberate: Gate 4 is given the protected sensitivity matrix `P`. The future learner must obtain or embody equivalent constraint information from permitted signals, replay, geometry or active measurements.
