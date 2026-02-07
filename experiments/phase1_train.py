"""Experiment 1: Affordance learning by active inference."""

from __future__ import annotations

from pathlib import Path

from agent.generative_model import build_phase1_model
from agent.pymdp_agent import ActiveInferenceAffordanceAgent
from env.objects import WorldObject
from env.world import AffordanceWorld
from experiments.metrics import plot_affordance_belief_over_episodes, summarize_runs


def run_training(episodes: int = 25, max_steps: int = 8):
    """Run training episodes and summarize affordance learning."""
    model = build_phase1_model()
    agent = ActiveInferenceAffordanceAgent(model=model)

    climbable_object = WorldObject(name="object_A", height=1)
    world = AffordanceWorld(obj=climbable_object, target_height=1)

    all_runs = []
    for episode in range(episodes):
        out = agent.rollout_episode(
            world=world,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )
        all_runs.append(out)

        last = out["logs"][-1]
        print(
            f"[train ep={episode + 1:02d}] "
            f"success={out['success']} steps={out['steps_to_success']} "
            f"q_aff(climbable)={last.q_aff_climbable:.3f} "
            f"q_pi(climb)={last.q_pi_climb:.3f} "
            f"G(climb)={last.g_climb:.3f} "
            f"prag={last.pragmatic_climb:.3f} epist={last.epistemic_climb:.3f}"
        )

    summary = summarize_runs(all_runs)
    print("\nTraining summary")
    for k, v in summary.items():
        print(f"- {k}: {v:.4f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    plot_saved = plot_affordance_belief_over_episodes(
        all_runs,
        out_path="experiments/results/phase1_train_beliefs.png",
        title="Phase 1 Training: Final Affordance Beliefs per Episode",
    )
    if plot_saved:
        print("- saved plot: experiments/results/phase1_train_beliefs.png")

    return all_runs, summary


if __name__ == "__main__":
    run_training()
