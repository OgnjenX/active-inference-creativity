"""Experiment 2: Transfer to a new object instance with fixed model structure."""

from __future__ import annotations

from pathlib import Path

from agent.generative_model import build_phase1_model
from agent.pymdp_agent import ActiveInferenceAffordanceAgent
from env.objects import WorldObject
from env.world import AffordanceWorld
from experiments.metrics import (
    episodes_to_confidence,
    plot_affordance_belief_over_episodes,
    summarize_runs,
)


def run_transfer(train_episodes: int = 20, transfer_episodes: int = 10, max_steps: int = 8):
    model = build_phase1_model()

    # Pre-train on object_A (climbable)
    pretrain_agent = ActiveInferenceAffordanceAgent(model=model)
    object_a = WorldObject(name="object_A", height=1)
    world_train = AffordanceWorld(obj=object_a, target_height=1)
    for _ in range(train_episodes):
        pretrain_agent.rollout_episode(world=world_train, max_steps=max_steps, carry_affordance_belief=True)

    transferred_aff_belief = pretrain_agent.q_aff.copy()

    # Transfer to object_B with same hidden dynamics.
    object_b = WorldObject(name="object_B", height=1)
    world_transfer = AffordanceWorld(obj=object_b, target_height=1)

    transferred_agent = ActiveInferenceAffordanceAgent(model=model)
    transferred_agent.q_aff = transferred_aff_belief.copy()

    fresh_agent = ActiveInferenceAffordanceAgent(model=model)

    transfer_runs = []
    fresh_runs = []

    for ep in range(transfer_episodes):
        out_transfer = transferred_agent.rollout_episode(
            world=world_transfer,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )
        out_fresh = fresh_agent.rollout_episode(
            world=world_transfer,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )

        transfer_runs.append(out_transfer)
        fresh_runs.append(out_fresh)

        last_t = out_transfer["logs"][-1]
        last_f = out_fresh["logs"][-1]
        print(
            f"[transfer ep={ep + 1:02d}] "
            f"steps(transferred/fresh)={out_transfer['steps_to_success']}/{out_fresh['steps_to_success']} "
            f"q_aff_climbable(transferred/fresh)={last_t.q_aff_climbable:.3f}/{last_f.q_aff_climbable:.3f}"
        )

    transfer_summary = summarize_runs(transfer_runs)
    fresh_summary = summarize_runs(fresh_runs)

    print("\nTransfer summary (transferred beliefs)")
    for k, v in transfer_summary.items():
        print(f"- {k}: {v:.4f}")

    print("\nBaseline summary (fresh agent)")
    for k, v in fresh_summary.items():
        print(f"- {k}: {v:.4f}")

    threshold = 0.7
    transfer_explore = episodes_to_confidence(transfer_runs, threshold=threshold)
    fresh_explore = episodes_to_confidence(fresh_runs, threshold=threshold)
    print(
        f"\nExploratory episodes to reach q_aff(climbable)>={threshold:.1f} "
        f"(transferred/fresh): {transfer_explore}/{fresh_explore}"
    )

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    transferred_plot_saved = plot_affordance_belief_over_episodes(
        transfer_runs,
        out_path="experiments/results/phase1_transfer_transferred_beliefs.png",
        title="Phase 1 Transfer: Final Beliefs (Transferred Agent)",
    )
    fresh_plot_saved = plot_affordance_belief_over_episodes(
        fresh_runs,
        out_path="experiments/results/phase1_transfer_fresh_beliefs.png",
        title="Phase 1 Transfer: Final Beliefs (Fresh Agent)",
    )
    if transferred_plot_saved:
        print("- saved plots: experiments/results/phase1_transfer_transferred_beliefs.png")
    if fresh_plot_saved:
        print("- saved plots: experiments/results/phase1_transfer_fresh_beliefs.png")

    return {
        "transferred": transfer_summary,
        "fresh": fresh_summary,
        "transferred_runs": transfer_runs,
        "fresh_runs": fresh_runs,
    }


if __name__ == "__main__":
    run_transfer()
