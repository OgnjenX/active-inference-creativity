"""Experiment: Phase 1.5 transfer with optional object-identity swap."""

from __future__ import annotations

from pathlib import Path

from agent.generative_model import build_phase1_5_model
from agent.pymdp_agent_phase1_5 import ActiveInferenceDisambiguationAgent
from env.objects import WorldObject
from env.world import TwoObjectAffordanceWorld
from experiments.metrics import (
    episodes_to_disambiguation,
    plot_phase1_5_beliefs,
    plot_phase1_5_entropy,
    summarize_phase1_5_runs,
)


def run_phase1_5_transfer(
    train_episodes: int = 20,
    transfer_episodes: int = 15,
    max_steps: int = 10,
    swap_identities: bool = False,
):
    """Pretrain then transfer to a new two-object world instance."""

    model = build_phase1_5_model()

    pretrain_agent = ActiveInferenceDisambiguationAgent(model=model, policy_precision=4.0, seed=21)
    pretrain_world = TwoObjectAffordanceWorld(
        object_1=WorldObject(name="object_1", height=1),
        object_2=WorldObject(name="object_2", height=0),
        target_height=1,
    )
    for _ in range(train_episodes):
        pretrain_agent.rollout_episode(
            world=pretrain_world,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )

    transfer_agent = ActiveInferenceDisambiguationAgent(model=model, policy_precision=4.0, seed=22)
    transfer_agent.q_aff_1 = pretrain_agent.q_aff_1.copy()
    transfer_agent.q_aff_2 = pretrain_agent.q_aff_2.copy()

    fresh_agent = ActiveInferenceDisambiguationAgent(model=model, policy_precision=4.0, seed=23)

    if swap_identities:
        transfer_world = TwoObjectAffordanceWorld(
            object_1=WorldObject(name="object_1", height=0),
            object_2=WorldObject(name="object_2", height=1),
            target_height=1,
        )
    else:
        transfer_world = TwoObjectAffordanceWorld(
            object_1=WorldObject(name="object_1", height=1),
            object_2=WorldObject(name="object_2", height=0),
            target_height=1,
        )

    transferred_runs = []
    fresh_runs = []

    for ep in range(transfer_episodes):
        out_transferred = transfer_agent.rollout_episode(
            world=transfer_world,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )
        out_fresh = fresh_agent.rollout_episode(
            world=transfer_world,
            max_steps=max_steps,
            carry_affordance_belief=True,
        )

        transferred_runs.append(out_transferred)
        fresh_runs.append(out_fresh)

        last_t = out_transferred["logs"][-1]
        last_f = out_fresh["logs"][-1]
        print(
            f"[phase1_5 transfer ep={ep + 1:02d}] "
            f"steps(transferred/fresh)={out_transferred['steps_to_success']}/{out_fresh['steps_to_success']} "
            f"q1(climb)={last_t.q_aff_1_climbable:.3f}/{last_f.q_aff_1_climbable:.3f} "
            f"q2(climb)={last_t.q_aff_2_climbable:.3f}/{last_f.q_aff_2_climbable:.3f}"
        )

    summary_transferred = summarize_phase1_5_runs(transferred_runs)
    summary_fresh = summarize_phase1_5_runs(fresh_runs)

    print("\nPhase 1.5 transfer summary (transferred beliefs)")
    for key, value in summary_transferred.items():
        print(f"- {key}: {value:.4f}")

    print("\nPhase 1.5 baseline summary (fresh beliefs)")
    for key, value in summary_fresh.items():
        print(f"- {key}: {value:.4f}")

    disambiguation_threshold = 0.45
    transferred_disambiguation = episodes_to_disambiguation(
        transferred_runs,
        threshold=disambiguation_threshold,
    )
    fresh_disambiguation = episodes_to_disambiguation(
        fresh_runs,
        threshold=disambiguation_threshold,
    )
    print(
        f"\nEpisodes to disambiguation |q1(climb)-q2(climb)|>={disambiguation_threshold:.2f} "
        f"(transferred/fresh): {transferred_disambiguation}/{fresh_disambiguation}"
    )

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    transferred_beliefs_saved = plot_phase1_5_beliefs(
        transferred_runs,
        out_path="experiments/results/phase1_5_transfer_transferred_beliefs.png",
        title="Phase 1.5 Transfer: Climbable Beliefs (Transferred Agent)",
    )
    fresh_beliefs_saved = plot_phase1_5_beliefs(
        fresh_runs,
        out_path="experiments/results/phase1_5_transfer_fresh_beliefs.png",
        title="Phase 1.5 Transfer: Climbable Beliefs (Fresh Agent)",
    )
    transferred_entropy_saved = plot_phase1_5_entropy(
        transferred_runs,
        out_path="experiments/results/phase1_5_transfer_transferred_entropy.png",
        title="Phase 1.5 Transfer: Entropy (Transferred Agent)",
    )
    fresh_entropy_saved = plot_phase1_5_entropy(
        fresh_runs,
        out_path="experiments/results/phase1_5_transfer_fresh_entropy.png",
        title="Phase 1.5 Transfer: Entropy (Fresh Agent)",
    )

    if transferred_beliefs_saved:
        print("- saved plot: experiments/results/phase1_5_transfer_transferred_beliefs.png")
    if fresh_beliefs_saved:
        print("- saved plot: experiments/results/phase1_5_transfer_fresh_beliefs.png")
    if transferred_entropy_saved:
        print("- saved plot: experiments/results/phase1_5_transfer_transferred_entropy.png")
    if fresh_entropy_saved:
        print("- saved plot: experiments/results/phase1_5_transfer_fresh_entropy.png")

    return {
        "transferred": summary_transferred,
        "fresh": summary_fresh,
        "transferred_runs": transferred_runs,
        "fresh_runs": fresh_runs,
    }


if __name__ == "__main__":
    run_phase1_5_transfer()
