"""Phase 3 control: no-structure world should not yield stable specialization."""

from __future__ import annotations

from pathlib import Path

from agent.pymdp_agent_phase3 import ActiveInferenceEmergentAffordanceAgent
from env_phase3.io import load_world
from experiments.metrics import (
    plot_phase3_dominance_rate,
    plot_phase3_prediction_error,
    plot_phase3_slot_specialization,
    summarize_phase3_runs,
)


def run_phase3_no_structure_control(
    episodes: int = 50,
    max_steps: int = 8,
    num_slots: int = 4,
    world_path: str = "env_phase3/worlds/phase3_no_structure_world.json",
):
    world = load_world(world_path, seed=61)
    agent = ActiveInferenceEmergentAffordanceAgent(
        num_objects=world.num_objects,
        num_slots=num_slots,
        policy_precision=5.0,
        recruit_threshold=0.7,
        recruit_scale=4.0,
        reserve_prior_mass=0.05,
        enable_slot_reduction=True,
        dirichlet_decay=0.02,
        seed=61,
    )

    runs = []
    for ep in range(episodes):
        out = agent.rollout_episode(world=world, max_steps=max_steps)
        runs.append(out)
        print(
            f"[phase3 control ep={ep + 1:02d}] "
            f"avg_PE={out['avg_prediction_error']:.3f} "
            f"reserve_dominant={out['reserve_slot_became_dominant']}"
        )

    summary = summarize_phase3_runs(runs)
    print("\nPhase 3 no-structure control summary")
    for key, value in summary.items():
        print(f"- {key}: {value:.4f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    pe_saved = plot_phase3_prediction_error(
        runs,
        out_path="experiments/results/phase3_control_prediction_error.png",
        title="Phase 3 Control: Prediction Error",
    )
    kl_saved = plot_phase3_slot_specialization(
        runs,
        out_path="experiments/results/phase3_control_slot_specialization.png",
        title="Phase 3 Control: Slot Specialization",
    )
    dom_saved = plot_phase3_dominance_rate(
        runs,
        out_path="experiments/results/phase3_control_reserve_dominance.png",
        title="Phase 3 Control: Reserve Slot Dominance",
    )
    if pe_saved:
        print("- saved plot: experiments/results/phase3_control_prediction_error.png")
    if kl_saved:
        print("- saved plot: experiments/results/phase3_control_slot_specialization.png")
    if dom_saved:
        print("- saved plot: experiments/results/phase3_control_reserve_dominance.png")

    return runs, summary, agent


if __name__ == "__main__":
    run_phase3_no_structure_control()
