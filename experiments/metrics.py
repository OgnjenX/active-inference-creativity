"""Metrics and lightweight plotting for Phase 1 experiments."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List


def summarize_runs(run_outputs: List[Dict]) -> Dict[str, float]:
    success_rate = sum(1 for r in run_outputs if r["success"]) / max(len(run_outputs), 1)
    avg_steps = mean(r["steps_to_success"] for r in run_outputs)
    avg_climbable_belief = mean(float(r["final_aff_belief"][1]) for r in run_outputs)
    return {
        "success_rate": success_rate,
        "avg_steps_to_success": avg_steps,
        "avg_final_climbable_belief": avg_climbable_belief,
    }


def episodes_to_confidence(run_outputs: List[Dict], threshold: float = 0.7) -> int:
    for i, run in enumerate(run_outputs, start=1):
        if float(run["final_aff_belief"][1]) >= threshold:
            return i
    return len(run_outputs)


def plot_affordance_belief_over_episodes(run_outputs: List[Dict], out_path: str, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping plot generation.")
        return False

    y_unknown = [float(r["final_aff_belief"][0]) for r in run_outputs]
    y_climbable = [float(r["final_aff_belief"][1]) for r in run_outputs]
    y_not = [float(r["final_aff_belief"][2]) for r in run_outputs]
    x = list(range(1, len(run_outputs) + 1))

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y_unknown, label="unknown")
    plt.plot(x, y_climbable, label="climbable")
    plt.plot(x, y_not, label="not_climbable")
    plt.xlabel("Episode")
    plt.ylabel("Belief")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True
