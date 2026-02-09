"""Phase 3 Experiment 1: emergent latent-cause recruitment in structured world."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agent.pymdp_agent_phase3 import ActiveInferenceEmergentAffordanceAgent
from env_phase3.io import load_world
from experiments.metrics import (
    plot_phase3_dominance_rate,
    plot_phase3_prediction_error,
    plot_phase3_slot_specialization,
    summarize_phase3_runs,
)


def run_phase3_emergence(
    episodes: int = 50,
    max_steps: int = 8,
    num_slots: int = 4,
    world_path: str = "env_phase3/worlds/phase3_train_world.json",
):
    world = load_world(world_path, seed=41)
    agent = ActiveInferenceEmergentAffordanceAgent(
        num_objects=world.num_objects,
        num_slots=num_slots,
        policy_precision=5.0,
        recruit_threshold=0.7,
        recruit_scale=4.0,
        reserve_prior_mass=0.05,
        enable_slot_reduction=True,
        dirichlet_decay=0.02,
        seed=41,
    )

    runs = []
    for ep in range(episodes):
        out = agent.rollout_episode(world=world, max_steps=max_steps, enable_trace=True)
        runs.append(out)
        dominant = bool(out["reserve_slot_became_dominant"])
        max_kl = float(np.max(out["final_slot_kl_per_slot"]))
        print(
            f"[phase3 emergence ep={ep + 1:02d}] "
            f"success={out['success']} steps={out['steps_to_success']} "
            f"avg_PE={out['avg_prediction_error']:.3f} "
            f"reserve_dominant={dominant} "
            f"max_slot_KL={max_kl:.3f}"
        )

    summary = summarize_phase3_runs(runs)
    print("\nPhase 3 emergence summary")
    for key, value in summary.items():
        print(f"- {key}: {value:.4f}")

    theta = agent._slot_means()  # diagnostics for final slot specialization
    print("- final slot means p((reach,climb) in [11,01,10,00])")
    for k in range(theta.shape[0]):
        probs = ", ".join(f"{v:.3f}" for v in theta[k])
        print(f"  slot_{k}: [{probs}] usage={agent.slot_usage[k]:.2f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    pe_saved = plot_phase3_prediction_error(
        runs,
        out_path="experiments/results/phase3_emergence_prediction_error.png",
        title="Phase 3 Emergence: Prediction Error",
    )
    kl_saved = plot_phase3_slot_specialization(
        runs,
        out_path="experiments/results/phase3_emergence_slot_specialization.png",
        title="Phase 3 Emergence: Slot Specialization",
    )
    dom_saved = plot_phase3_dominance_rate(
        runs,
        out_path="experiments/results/phase3_emergence_reserve_dominance.png",
        title="Phase 3 Emergence: Reserve Slot Dominance",
    )
    if pe_saved:
        print("- saved plot: experiments/results/phase3_emergence_prediction_error.png")
    if kl_saved:
        print("- saved plot: experiments/results/phase3_emergence_slot_specialization.png")
    if dom_saved:
        print("- saved plot: experiments/results/phase3_emergence_reserve_dominance.png")

    last_trace = runs[-1].get("trace", []) if runs else []
    if last_trace:
        trace_path = Path("experiments/results/phase3_emergence_last_trace.json")
        trace_path.write_text(json.dumps(last_trace, indent=2, default=lambda x: np.asarray(x).tolist()))
        print("- saved trace: experiments/results/phase3_emergence_last_trace.json")

    return runs, summary, agent


if __name__ == "__main__":
    run_phase3_emergence()
