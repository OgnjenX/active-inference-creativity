"""Diagnostic script: model preference should depend on world structure."""

from __future__ import annotations

from pathlib import Path

from agent.pymdp_agent_phase2_level2 import ActiveInferenceStructureLearningAgent
from env_phase2.io import load_world
from experiments.metrics import plot_phase2_model_posterior, summarize_phase2_runs


def run_model_selection_sanity(
    episodes: int = 35,
    max_steps: int = 6,
    structured_world: str = "env_phase2/worlds/phase2_train_world.json",
    unstructured_world: str = "env_phase2/worlds/phase2_symmetric_world.json",
    occam_scale: float = 0.1,
):
    """Run model-selection sanity checks.

    Runs multiple episodes in a structured and an unstructured world using
    an ActiveInferenceStructureLearningAgent, summarizes results, and
    saves model-posterior plots to `experiments/results`.

    Args:
        episodes: Number of episodes to run per condition.
        max_steps: Maximum steps per episode.
        structured_world: Path to the structured-world JSON file.
        unstructured_world: Path to the unstructured-world JSON file.
        occam_scale: Scale for Occam's penalty in model comparison.

    Returns:
        A dict mapping condition labels ("structured", "unstructured") to
        lists of run results produced by `agent.rollout_episode`.
    """

    results = {}
    for label, path, seed in (
        ("structured", structured_world, 31),
        ("unstructured", unstructured_world, 32),
    ):
        world = load_world(path, seed=seed)
        agent = ActiveInferenceStructureLearningAgent(
            policy_precision=5.0,
            model_precision=1.0,
            occam_scale=occam_scale,
        )
        runs = []
        for _ in range(episodes):
            runs.append(agent.rollout_episode(world=world, max_steps=max_steps))
        results[label] = runs

    print("Model-selection sanity summary")
    for label, runs in results.items():
        summary = summarize_phase2_runs(runs)
        print(
            f"- {label}: avg_final_p_factorized={summary['avg_final_p_factorized']:.4f} "
            f"avg_final_score(flat/factorized)="
            f"{summary['avg_final_score_flat']:.4f}/{summary['avg_final_score_factorized']:.4f}"
        )

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    for label, runs in results.items():
        saved = plot_phase2_model_posterior(
            runs,
            out_path=f"experiments/results/phase2_{label}_model_posterior.png",
            title=f"Phase 2 {label.title()}: Model Posterior",
        )
        if saved:
            print(f"- saved plot: experiments/results/phase2_{label}_model_posterior.png")

    return results


if __name__ == "__main__":
    run_model_selection_sanity()
