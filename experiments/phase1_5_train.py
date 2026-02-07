"""Experiment: Phase 1.5 two-object affordance disambiguation by active inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agent.generative_model import build_phase1_5_model
from agent.pymdp_agent_phase1_5 import ActiveInferenceDisambiguationAgent
from env.actions import CLIMB_OBJECT_1, CLIMB_OBJECT_2
from env.objects import WorldObject
from env.world import TwoObjectAffordanceWorld
from experiments.metrics import (
    exploratory_ratio_window,
    plot_phase1_5_beliefs,
    plot_phase1_5_entropy,
    plot_learning_kl_over_episodes,
    summarize_learning_diagnostics,
    summarize_phase1_5_runs,
)


def run_phase1_5_training(
    episodes: int = 30,
    max_steps: int = 10,
    enable_parameter_learning: bool = False,
    dirichlet_prior_strength: float = 8.0,
):
    """Run disambiguation training where exactly one of two objects is climbable."""

    model = build_phase1_5_model()
    agent = ActiveInferenceDisambiguationAgent(
        model=model,
        policy_precision=4.0,
        seed=1,
        enable_parameter_learning=enable_parameter_learning,
        dirichlet_prior_strength=dirichlet_prior_strength,
    )

    world = TwoObjectAffordanceWorld(
        object_1=WorldObject(name="object_1", height=1),
        object_2=WorldObject(name="object_2", height=0),
        target_height=1,
    )

    run_outputs = []
    target_climbable = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    target_not_climbable = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    for episode in range(episodes):
        out = agent.rollout_episode(
            world=world,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )
        if enable_parameter_learning:
            b = agent.model["B_height"]
            # For action climb_object_1, aff1 controls transition regardless of aff2.
            obj1_climbable = np.mean(np.abs(b[CLIMB_OBJECT_1, 1] - target_climbable))
            obj1_not = np.mean(np.abs(b[CLIMB_OBJECT_1, 2] - target_not_climbable))
            # For action climb_object_2, aff2 controls transition regardless of aff1.
            obj2_climbable = np.mean(np.abs(b[CLIMB_OBJECT_2, :, 1] - target_climbable))
            obj2_not = np.mean(np.abs(b[CLIMB_OBJECT_2, :, 2] - target_not_climbable))
            out["b_target_distance"] = float(0.25 * (obj1_climbable + obj1_not + obj2_climbable + obj2_not))
        run_outputs.append(out)
        last = out["logs"][-1]
        print(
            f"[phase1_5 train ep={episode + 1:02d}] "
            f"success={out['success']} steps={out['steps_to_success']} "
            f"q1(climb)={last.q_aff_1_climbable:.3f} "
            f"q2(climb)={last.q_aff_2_climbable:.3f} "
            f"H={last.affordance_entropy_total:.3f} mode={last.mode} "
            f"epi(obj1/obj2)={last.epistemic_climb_object_1:.3f}/"
            f"{last.epistemic_climb_object_2:.3f} "
            f"prag(obj1/obj2)={last.pragmatic_climb_object_1:.3f}/"
            f"{last.pragmatic_climb_object_2:.3f}"
        )

    summary = summarize_phase1_5_runs(run_outputs)
    print("\nPhase 1.5 training summary")
    for key, value in summary.items():
        print(f"- {key}: {value:.4f}")

    if enable_parameter_learning:
        learning_summary = summarize_learning_diagnostics(
            run_outputs, metric_key="b_target_distance"
        )
        exploration_summary = exploratory_ratio_window(run_outputs, window_size=5)
        print("\nLevel 1 parameter-learning diagnostics")
        print(f"- initial_target_distance: {learning_summary['initial_metric']:.6f}")
        print(f"- final_target_distance: {learning_summary['final_metric']:.6f}")
        print(f"- delta_target_distance: {learning_summary['delta_metric']:.6f}")
        print(
            "- exploratory_ratio_early_vs_late: "
            f"{exploration_summary['early_ratio']:.4f} -> {exploration_summary['late_ratio']:.4f}"
        )

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    beliefs_plot_saved = plot_phase1_5_beliefs(
        run_outputs,
        out_path="experiments/results/phase1_5_train_beliefs.png",
        title="Phase 1.5 Training: Climbable Beliefs by Object",
    )
    entropy_plot_saved = plot_phase1_5_entropy(
        run_outputs,
        out_path="experiments/results/phase1_5_train_entropy.png",
        title="Phase 1.5 Training: Final Affordance Entropy",
    )
    if beliefs_plot_saved:
        print("- saved plot: experiments/results/phase1_5_train_beliefs.png")
    if entropy_plot_saved:
        print("- saved plot: experiments/results/phase1_5_train_entropy.png")
    if enable_parameter_learning:
        kl_plot_saved = plot_learning_kl_over_episodes(
            run_outputs,
            out_path="experiments/results/phase1_5_train_level1_b_kl.png",
            title="Phase 1.5 Level 1: Distance to Target Transition",
            metric_key="b_target_distance",
        )
        if kl_plot_saved:
            print("- saved plot: experiments/results/phase1_5_train_level1_b_kl.png")

    return run_outputs, summary


if __name__ == "__main__":
    run_phase1_5_training()
