"""Phase 3 Experiment 2: transfer of emergent latent causes to novel objects."""

from __future__ import annotations

from pathlib import Path

from agent.pymdp_agent_phase3 import ActiveInferenceEmergentAffordanceAgent
from env_phase3.io import load_world
from experiments.metrics import (
    plot_phase2_transfer_steps,
    plot_phase3_transfer_comparison,
    summarize_phase3_runs,
)


def run_phase3_transfer(
    train_episodes: int = 120,
    transfer_episodes: int = 25,
    max_steps: int = 8,
    num_slots: int = 4,
    train_world_path: str = "env_phase3/worlds/phase3_train_world.json",
    transfer_world_path: str = "env_phase3/worlds/phase3_transfer_world.json",
):
    train_world = load_world(train_world_path, seed=51)
    transfer_world = load_world(transfer_world_path, seed=52)

    pretrain_agent = ActiveInferenceEmergentAffordanceAgent(
        num_objects=train_world.num_objects,
        num_slots=num_slots,
        policy_precision=5.0,
        recruit_threshold=0.7,
        recruit_scale=4.0,
        reserve_prior_mass=0.05,
        enable_slot_reduction=True,
        dirichlet_decay=0.02,
        seed=51,
    )
    for _ in range(train_episodes):
        pretrain_agent.rollout_episode(world=train_world, max_steps=max_steps)

    transferred_agent = ActiveInferenceEmergentAffordanceAgent(
        num_objects=transfer_world.num_objects,
        num_slots=num_slots,
        policy_precision=5.0,
        recruit_threshold=0.7,
        recruit_scale=4.0,
        reserve_prior_mass=0.05,
        enable_slot_reduction=True,
        dirichlet_decay=0.02,
        seed=52,
    )
    transferred_agent.import_transfer_state(pretrain_agent.export_transfer_state())

    fresh_agent = ActiveInferenceEmergentAffordanceAgent(
        num_objects=transfer_world.num_objects,
        num_slots=num_slots,
        policy_precision=5.0,
        recruit_threshold=0.7,
        recruit_scale=4.0,
        reserve_prior_mass=0.05,
        enable_slot_reduction=True,
        dirichlet_decay=0.02,
        seed=53,
    )

    transferred_runs = []
    fresh_runs = []

    for ep in range(transfer_episodes):
        out_t = transferred_agent.rollout_episode(
            world=transfer_world,
            max_steps=max_steps,
            reset_object_beliefs=True,
        )
        out_f = fresh_agent.rollout_episode(
            world=transfer_world,
            max_steps=max_steps,
            reset_object_beliefs=True,
        )
        transferred_runs.append(out_t)
        fresh_runs.append(out_f)

        print(
            f"[phase3 transfer ep={ep + 1:02d}] "
            f"steps(transferred/fresh)={out_t['steps_to_success']}/{out_f['steps_to_success']} "
            f"explore(transferred/fresh)={out_t['exploratory_actions']}/{out_f['exploratory_actions']} "
            f"PE(transferred/fresh)={out_t['avg_prediction_error']:.3f}/{out_f['avg_prediction_error']:.3f}"
        )

    summary_t = summarize_phase3_runs(transferred_runs)
    summary_f = summarize_phase3_runs(fresh_runs)

    print("\nPhase 3 transfer summary (transferred)")
    for key, value in summary_t.items():
        print(f"- {key}: {value:.4f}")

    print("\nPhase 3 transfer summary (fresh baseline)")
    for key, value in summary_f.items():
        print(f"- {key}: {value:.4f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    steps_saved = plot_phase2_transfer_steps(
        transferred_runs,
        fresh_runs,
        out_path="experiments/results/phase3_transfer_steps.png",
        title="Phase 3 Transfer: Steps to Success",
    )
    exp_saved = plot_phase3_transfer_comparison(
        transferred_runs,
        fresh_runs,
        out_path="experiments/results/phase3_transfer_exploration.png",
        title="Phase 3 Transfer: Exploratory Actions",
    )
    if steps_saved:
        print("- saved plot: experiments/results/phase3_transfer_steps.png")
    if exp_saved:
        print("- saved plot: experiments/results/phase3_transfer_exploration.png")

    return {
        "transferred_runs": transferred_runs,
        "fresh_runs": fresh_runs,
        "transferred_summary": summary_t,
        "fresh_summary": summary_f,
    }


if __name__ == "__main__":
    run_phase3_transfer()
