"""Experiment 1: Affordance learning by active inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agent.generative_model import AFF_CLIMBABLE, build_phase1_model
from agent.pymdp_agent import ActiveInferenceAffordanceAgent
from env.actions import CLIMB
from env.objects import WorldObject
from env.world import AffordanceWorld
from experiments.metrics import (
    plot_affordance_belief_over_episodes,
    plot_learning_kl_over_episodes,
    summarize_learning_diagnostics,
    summarize_runs,
)


def run_training(
    episodes: int = 25,
    max_steps: int = 8,
    enable_parameter_learning: bool = False,
    dirichlet_prior_strength: float = 8.0,
):
    """Run training episodes and summarize affordance learning."""
    model = build_phase1_model()
    agent = ActiveInferenceAffordanceAgent(
        model=model,
        enable_parameter_learning=enable_parameter_learning,
        dirichlet_prior_strength=dirichlet_prior_strength,
    )

    climbable_object = WorldObject(name="object_A", height=1)
    world = AffordanceWorld(obj=climbable_object, target_height=1)

    all_runs = []
    target_climbable = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    target_not_climbable = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    for episode in range(episodes):
        out = agent.rollout_episode(
            world=world,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )
        if enable_parameter_learning:
            learned_climbable = agent.model["B_height"][CLIMB, 1]
            learned_not = agent.model["B_height"][CLIMB, 2]
            out["b_target_distance"] = float(
                0.5
                * (
                    np.mean(np.abs(learned_climbable - target_climbable))
                    + np.mean(np.abs(learned_not - target_not_climbable))
                )
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

    if enable_parameter_learning:
        learning_summary = summarize_learning_diagnostics(all_runs, metric_key="b_target_distance")
        print("\nLevel 1 parameter-learning diagnostics")
        print(f"- initial_target_distance: {learning_summary['initial_metric']:.6f}")
        print(f"- final_target_distance: {learning_summary['final_metric']:.6f}")
        print(f"- delta_target_distance: {learning_summary['delta_metric']:.6f}")
        learned_climb = agent.model["B_height"][CLIMB, AFF_CLIMBABLE]
        print(
            "- learned B[climb, climbable] column(low->next/high->next): "
            f"{learned_climb[:, 0].round(4).tolist()} / {learned_climb[:, 1].round(4).tolist()}"
        )

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    plot_saved = plot_affordance_belief_over_episodes(
        all_runs,
        out_path="experiments/results/phase1_train_beliefs.png",
        title="Phase 1 Training: Final Affordance Beliefs per Episode",
    )
    if plot_saved:
        print("- saved plot: experiments/results/phase1_train_beliefs.png")
    if enable_parameter_learning:
        kl_plot_saved = plot_learning_kl_over_episodes(
            all_runs,
            out_path="experiments/results/phase1_train_level1_b_kl.png",
            title="Phase 1 Level 1: Distance to Target Transition",
            metric_key="b_target_distance",
        )
        if kl_plot_saved:
            print("- saved plot: experiments/results/phase1_train_level1_b_kl.png")

    return all_runs, summary


if __name__ == "__main__":
    run_training()
