"""Experiment: Phase 1.5 two-object affordance disambiguation by active inference."""

from __future__ import annotations

from pathlib import Path

from agent.generative_model import build_phase1_5_model
from agent.pymdp_agent_phase1_5 import ActiveInferenceDisambiguationAgent
from env.objects import WorldObject
from env.world import TwoObjectAffordanceWorld
from experiments.metrics import (
    plot_phase1_5_beliefs,
    plot_phase1_5_entropy,
    summarize_phase1_5_runs,
)


def run_phase1_5_training(episodes: int = 30, max_steps: int = 10):
    """Run disambiguation training where exactly one of two objects is climbable."""

    model = build_phase1_5_model()
    agent = ActiveInferenceDisambiguationAgent(model=model, policy_precision=4.0, seed=1)

    world = TwoObjectAffordanceWorld(
        object_1=WorldObject(name="object_1", height=1),
        object_2=WorldObject(name="object_2", height=0),
        target_height=1,
    )

    run_outputs = []
    for episode in range(episodes):
        out = agent.rollout_episode(
            world=world,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )
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

    return run_outputs, summary


if __name__ == "__main__":
    run_phase1_5_training()
