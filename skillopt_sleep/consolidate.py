"""SkillOpt-Sleep — Stage 4: consolidate (one SkillOpt epoch).

This is the core that makes nightly evolution *safe*: it proposes bounded
edits from replayed failures, applies them to a candidate skill/memory, then
**gates** the candidate on a held-out slice of the user's own tasks. Only a
candidate that strictly improves the held-out score is accepted — the SkillOpt
validation gate, vendored self-contained in ``skillopt_sleep.gate``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from skillopt_sleep.backend import Backend
from skillopt_sleep.memory import apply_edits
from skillopt_sleep.replay import aggregate_scores, replay_batch
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord


# Self-contained validation gate (vendored from SkillOpt; zero dependency on the
# research package, so this open-source tool stays decoupled from the paper code).
from skillopt_sleep.gate import evaluate_gate, select_gate_score
_HAVE_REPO_GATE = True


@dataclass
class ConsolidationResult:
    accepted: bool
    gate_action: str
    baseline_score: float
    candidate_score: float
    new_skill: str
    new_memory: str
    applied_edits: List[EditRecord]
    rejected_edits: List[EditRecord]
    holdout_baseline: float
    holdout_candidate: float
    n_train_failures: int = 0
    n_train_successes: int = 0
    no_edits_reason: str = ""


def _split(tasks: List[TaskRecord]) -> Tuple[List[TaskRecord], List[TaskRecord]]:
    """Return (train_tasks, val_tasks).

    train drives reflect; val gates updates. test is held out entirely from
    consolidation and is scored by the caller. Accepts legacy split names
    (replay->train, holdout->val) for robustness.
    """
    def _norm(s: str) -> str:
        return {"replay": "train", "holdout": "val"}.get(s, s)

    train = [t for t in tasks if _norm(t.split) == "train"]
    val = [t for t in tasks if _norm(t.split) == "val"]
    # be robust if a split is empty: fall back so a night still does something,
    # but never silently use test as val.
    test = [t for t in tasks if _norm(t.split) == "test"]
    if not val:
        # prefer train as the gate reference over nothing; last resort all-but-test
        val = train or [t for t in tasks if _norm(t.split) != "test"] or tasks
    if not train:
        train = val
    return train, val


def consolidate(
    backend: Backend,
    tasks: List[TaskRecord],
    skill: str,
    memory: str,
    *,
    edit_budget: int = 4,
    gate_metric: str = "mixed",
    gate_mixed_weight: float = 0.5,
    gate_mode: str = "on",       # "on" (hard/soft per gate_metric) | "off" (greedy)
    rollouts_k: int = 1,         # >1 => multi-rollout contrastive reflection
    evolve_skill: bool = True,
    evolve_memory: bool = True,
    night: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> ConsolidationResult:
    """Run one consolidation epoch: reflect -> bounded edit -> gate.

    train tasks drive reflect; val tasks gate the update (test is held out by the
    caller). With ``gate_mode='off'`` edits are accepted greedily (no val-improve
    requirement) — the user opts out of hard filtering — but val scores are still
    recorded so the report shows whether quality moved.

    Skill and memory are evolved in sequence (skill first if both enabled).
    """
    train_tasks, val_tasks = _split(tasks)
    gate_off = str(gate_mode).strip().lower() in {"off", "none", "false", "greedy"}

    def log(message: str) -> None:
        if progress is not None:
            progress(message)

    # ── baseline on the VAL slice (the gate reference) ────────────────────
    log(f"baseline replay start: val_tasks={len(val_tasks)}")
    base_pairs = replay_batch(backend, val_tasks, skill, memory)
    base_hard, base_soft = aggregate_scores(base_pairs)
    base_score = select_gate_score(base_hard, base_soft, gate_metric, gate_mixed_weight)
    log(f"baseline replay done: hard={base_hard:.3f} soft={base_soft:.3f}")

    # ── reflect over TRAIN-split failures/successes ───────────────────────
    log(f"train replay start: train_tasks={len(train_tasks)}")
    train_pairs = replay_batch(backend, train_tasks, skill, memory)
    failures = [(t, r) for (t, r) in train_pairs if r.hard < 1.0]
    successes = [(t, r) for (t, r) in train_pairs if r.hard >= 1.0]
    log(f"train replay done: failures={len(failures)} successes={len(successes)}")
    no_edits_reason = "no_train_failures" if not failures else ""

    cand_skill, cand_memory = skill, memory
    all_applied: List[EditRecord] = []
    all_rejected: List[EditRecord] = []

    def _gate_apply(doc: str, edits: List[EditRecord], which: str) -> str:
        nonlocal cand_skill, cand_memory, base_score, all_applied, all_rejected
        if not edits:
            log(f"{which} reflect produced no edits")
            return doc
        new_doc, applied = apply_edits(doc, edits)
        if not applied:
            log(f"{which} edits could not be applied")
            return doc
        # score the candidate on the VAL slice
        log(f"{which} gate replay start: edits={len(applied)}")
        trial_skill = new_doc if which == "skill" else cand_skill
        trial_memory = new_doc if which == "memory" else cand_memory
        pairs = replay_batch(backend, val_tasks, trial_skill, trial_memory)
        h, s = aggregate_scores(pairs)
        cand_score = select_gate_score(h, s, gate_metric, gate_mixed_weight)
        # gate OFF: accept greedily (no regression check); gate ON: strict improve
        if gate_off or cand_score > base_score:
            base_score = max(base_score, cand_score)
            all_applied.extend(applied)
            log(f"{which} gate accepted: score={cand_score:.3f}")
            return new_doc
        all_rejected.extend(applied)
        log(f"{which} gate rejected: score={cand_score:.3f} <= base={base_score:.3f}")
        return doc

    if evolve_skill:
        log("skill reflect start")
        if rollouts_k > 1:
            # multi-rollout contrastive reflection: run each train task K times
            # and distill a rule from the good-vs-bad contrast (the imagination signal).
            from skillopt_sleep.rollout import multi_rollout, contrastive_reflect
            sets = [multi_rollout(backend, t, cand_skill, cand_memory, k=rollouts_k)
                    for t in train_tasks]
            edits = contrastive_reflect(
                backend, sets, cand_skill, cand_memory,
                edit_budget=edit_budget, target="skill",
            )
            # fall back to single-shot reflect if contrast yielded nothing
            if not edits:
                edits = backend.reflect(
                    failures, successes, cand_skill, cand_memory,
                    edit_budget=edit_budget, evolve_skill=True, evolve_memory=False,
                )
        else:
            edits = backend.reflect(
                failures, successes, cand_skill, cand_memory,
                edit_budget=edit_budget, evolve_skill=True, evolve_memory=False,
            )
        log(f"skill reflect done: edits={len(edits)}")
        if failures and not edits and not no_edits_reason:
            no_edits_reason = "skill_reflect_returned_no_edits"
        cand_skill = _gate_apply(cand_skill, edits, "skill")

    if evolve_memory:
        # re-evaluate failures under the (possibly improved) skill
        log("memory replay start")
        train_pairs2 = replay_batch(backend, train_tasks, cand_skill, cand_memory)
        failures2 = [(t, r) for (t, r) in train_pairs2 if r.hard < 1.0]
        successes2 = [(t, r) for (t, r) in train_pairs2 if r.hard >= 1.0]
        log(f"memory replay done: failures={len(failures2)} successes={len(successes2)}")
        log("memory reflect start")
        edits_m = backend.reflect(
            failures2, successes2, cand_skill, cand_memory,
            edit_budget=edit_budget, evolve_skill=False, evolve_memory=True,
        )
        log(f"memory reflect done: edits={len(edits_m)}")
        if failures2 and not edits_m and not no_edits_reason:
            no_edits_reason = "memory_reflect_returned_no_edits"
        cand_memory = _gate_apply(cand_memory, edits_m, "memory")

    # ── final decision, scored on the VAL slice ───────────────────────────
    log("final replay start")
    final_pairs = replay_batch(backend, val_tasks, cand_skill, cand_memory)
    final_hard, final_soft = aggregate_scores(final_pairs)
    final_score = select_gate_score(final_hard, final_soft, gate_metric, gate_mixed_weight)
    base_gate_score = select_gate_score(base_hard, base_soft, gate_metric, gate_mixed_weight)
    log(f"final replay done: score={final_score:.3f} base={base_gate_score:.3f}")

    if gate_off:
        # greedy mode: keep whatever edits we applied; report quality movement
        accepted = bool(all_applied)
        if final_score > base_gate_score:
            action = "greedy_improved"
        elif final_score < base_gate_score:
            action = "greedy_regressed"
        else:
            action = "greedy_flat" if all_applied else "greedy_noop"
    elif _HAVE_REPO_GATE:
        gate = evaluate_gate(
            candidate_skill=cand_skill,
            cand_hard=final_hard,
            current_skill=skill,
            current_score=base_gate_score,
            best_skill=skill,
            best_score=base_gate_score,
            best_step=night - 1,
            global_step=night,
            cand_soft=final_soft,
            metric=gate_metric,
            mixed_weight=gate_mixed_weight,
        )
        action = gate.action
        accepted = bool(all_applied) and final_score > base_gate_score
    else:
        action = "accept" if final_score > base_gate_score else "reject"
        accepted = bool(all_applied) and final_score > base_gate_score

    return ConsolidationResult(
        accepted=accepted,
        gate_action=action,
        baseline_score=base_gate_score,
        candidate_score=final_score,
        new_skill=cand_skill if accepted else skill,
        new_memory=cand_memory if accepted else memory,
        applied_edits=all_applied,
        rejected_edits=all_rejected,
        holdout_baseline=base_hard,
        holdout_candidate=final_hard,
        n_train_failures=len(failures),
        n_train_successes=len(successes),
        no_edits_reason="" if (all_applied or all_rejected) else no_edits_reason,
    )
