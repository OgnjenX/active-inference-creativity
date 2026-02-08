"""Stage B training: Phase 2 + Level 2 model comparison diagnostics."""

from __future__ import annotations

from pathlib import Path

from agent.pymdp_agent_phase2_level2 import ActiveInferenceStructureLearningAgent
from env_phase2.io import load_world
from experiments.metrics import (
    plot_phase2_model_evidence,
    plot_phase2_model_evidence_per_step,
    plot_phase2_model_posterior,
    summarize_phase2_runs,
)


def run_phase2_training(
    episodes: int = 40,
    max_steps: int = 6,
    world_path: str = "env_phase2/worlds/phase2_train_world.json",
    occam_scale: float = 0.1,
):
    world = load_world(world_path, seed=11)
    agent = ActiveInferenceStructureLearningAgent(
        policy_precision=5.0,
        model_precision=1.0,
        occam_scale=occam_scale,
    )

    runs = []
    for ep in range(episodes):
        out = agent.rollout_episode(world=world, max_steps=max_steps)
        runs.append(out)
        post = out["model_posterior_final"]
        print(
            f"[phase2 train ep={ep + 1:02d}] "
            f"success={out['success']} steps={out['steps_to_success']} "
            f"P(flat/factorized)={post[0]:.3f}/{post[1]:.3f} "
            f"selected={out['selected_family_final']} "
            f"F(flat/factorized)={out['cumulative_fe_flat']:.3f}/{out['cumulative_fe_factorized']:.3f} "
            f"C(flat/factorized)={out['complexity_flat']:.3f}/{out['complexity_factorized']:.3f}"
        )

    summary = summarize_phase2_runs(runs)
    print("\nPhase 2 + Level 2 training summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    evidence_saved = plot_phase2_model_evidence(
        runs,
        out_path="experiments/results/phase2_train_model_evidence.png",
        title="Phase 2 Training: Cumulative Model Free Energy",
    )
    evidence_rate_saved = plot_phase2_model_evidence_per_step(
        runs,
        out_path="experiments/results/phase2_train_model_evidence_per_step.png",
        title="Phase 2 Training: Mean Model Free Energy per Step",
    )
    posterior_saved = plot_phase2_model_posterior(
        runs,
        out_path="experiments/results/phase2_train_model_posterior.png",
        title="Phase 2 Training: Model Posterior over Episodes",
    )
    if evidence_saved:
        print("- saved plot: experiments/results/phase2_train_model_evidence.png")
    if evidence_rate_saved:
        print("- saved plot: experiments/results/phase2_train_model_evidence_per_step.png")
    if posterior_saved:
        print("- saved plot: experiments/results/phase2_train_model_posterior.png")

    return runs, summary, agent


if __name__ == "__main__":
    run_phase2_training()
