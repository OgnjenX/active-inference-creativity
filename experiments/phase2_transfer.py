"""Stage B transfer: evaluate selected-model transfer in Phase 2."""

from __future__ import annotations

from pathlib import Path

from agent.pymdp_agent_phase2_level2 import ActiveInferenceStructureLearningAgent
from env_phase2.io import load_world
from experiments.metrics import (
    plot_phase2_model_posterior,
    plot_phase2_transfer_steps,
    summarize_phase2_runs,
)


def run_phase2_transfer(
    train_episodes: int = 30,
    transfer_episodes: int = 20,
    max_steps: int = 6,
    train_world_path: str = "env_phase2/worlds/phase2_train_world.json",
    transfer_world_path: str = "env_phase2/worlds/phase2_transfer_world.json",
    occam_scale: float = 0.1,
):
    train_world = load_world(train_world_path, seed=21)
    transfer_world = load_world(transfer_world_path, seed=22)

    pretrain_agent = ActiveInferenceStructureLearningAgent(
        policy_precision=5.0,
        model_precision=1.0,
        occam_scale=occam_scale,
    )
    for _ in range(train_episodes):
        pretrain_agent.rollout_episode(world=train_world, max_steps=max_steps)

    transferred_agent = ActiveInferenceStructureLearningAgent(
        policy_precision=5.0,
        model_precision=1.0,
        occam_scale=occam_scale,
    )
    transferred_agent.import_transfer_state(pretrain_agent.export_transfer_state(), reset_object_beliefs=True)

    fresh_agent = ActiveInferenceStructureLearningAgent(
        policy_precision=5.0,
        model_precision=1.0,
        occam_scale=occam_scale,
    )

    transferred_runs = []
    fresh_runs = []

    for ep in range(transfer_episodes):
        out_t = transferred_agent.rollout_episode(world=transfer_world, max_steps=max_steps)
        out_f = fresh_agent.rollout_episode(world=transfer_world, max_steps=max_steps)
        transferred_runs.append(out_t)
        fresh_runs.append(out_f)

        post_t = out_t["model_posterior_final"]
        post_f = out_f["model_posterior_final"]
        print(
            f"[phase2 transfer ep={ep + 1:02d}] "
            f"steps(transferred/fresh)={out_t['steps_to_success']}/{out_f['steps_to_success']} "
            f"selected(transferred/fresh)={out_t['selected_family_final']}/{out_f['selected_family_final']} "
            f"P_factorized(transferred/fresh)={post_t[1]:.3f}/{post_f[1]:.3f}"
        )

    summary_t = summarize_phase2_runs(transferred_runs)
    summary_f = summarize_phase2_runs(fresh_runs)

    print("\nPhase 2 transfer summary (transferred structure state)")
    for key, value in summary_t.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")

    print("\nPhase 2 transfer summary (fresh baseline)")
    for key, value in summary_f.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    steps_saved = plot_phase2_transfer_steps(
        transferred_runs,
        fresh_runs,
        out_path="experiments/results/phase2_transfer_steps.png",
        title="Phase 2 Transfer: Steps to Success",
    )
    post_t_saved = plot_phase2_model_posterior(
        transferred_runs,
        out_path="experiments/results/phase2_transfer_transferred_posterior.png",
        title="Phase 2 Transfer: Model Posterior (Transferred)",
    )
    post_f_saved = plot_phase2_model_posterior(
        fresh_runs,
        out_path="experiments/results/phase2_transfer_fresh_posterior.png",
        title="Phase 2 Transfer: Model Posterior (Fresh)",
    )
    if steps_saved:
        print("- saved plot: experiments/results/phase2_transfer_steps.png")
    if post_t_saved:
        print("- saved plot: experiments/results/phase2_transfer_transferred_posterior.png")
    if post_f_saved:
        print("- saved plot: experiments/results/phase2_transfer_fresh_posterior.png")

    return {
        "transferred_runs": transferred_runs,
        "fresh_runs": fresh_runs,
        "transferred_summary": summary_t,
        "fresh_summary": summary_f,
    }


if __name__ == "__main__":
    run_phase2_transfer()
