"""Metrics and lightweight plotting for Phase 1 experiments."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List

# Constant for missing matplotlib message
_MATPLOTLIB_MISSING_MSG = "matplotlib not installed; skipping plot generation."


def summarize_runs(run_outputs: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics over a collection of episode results."""
    success_rate = sum(1 for r in run_outputs if r["success"]) / max(len(run_outputs), 1)
    avg_steps = mean(r["steps_to_success"] for r in run_outputs)
    avg_climbable_belief = mean(float(r["final_aff_belief"][1]) for r in run_outputs)
    return {
        "success_rate": success_rate,
        "avg_steps_to_success": avg_steps,
        "avg_final_climbable_belief": avg_climbable_belief,
    }


def episodes_to_confidence(
    run_outputs: List[Dict], threshold: float = 0.7
) -> int:
    """Return the first episode where affordance belief exceeds threshold."""
    for i, run in enumerate(run_outputs, start=1):
        if float(run["final_aff_belief"][1]) >= threshold:
            return i
    return len(run_outputs)


def plot_affordance_belief_over_episodes(
    run_outputs: List[Dict], out_path: str, title: str
) -> bool:
    """Plot affordance beliefs across episodes and save to file."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
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


def summarize_phase1_5_runs(run_outputs: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics for Phase 1.5 episodes."""
    if not run_outputs:
        return {
            "success_rate": 0.0,
            "avg_steps_to_success": 0.0,
            "avg_exploratory_actions": 0.0,
            "avg_exploitative_actions": 0.0,
            "avg_final_affordance_entropy": 0.0,
            "avg_final_aff1_climbable_belief": 0.0,
            "avg_final_aff2_climbable_belief": 0.0,
        }

    success_rate = sum(1 for r in run_outputs if r["success"]) / max(len(run_outputs), 1)
    avg_steps = mean(r["steps_to_success"] for r in run_outputs)
    avg_exploratory = mean(r["exploratory_actions"] for r in run_outputs)
    avg_exploitative = mean(r["exploitative_actions"] for r in run_outputs)
    valid_logs = [r["logs"][-1].affordance_entropy_total for r in run_outputs if r["logs"]]
    avg_aff_entropy = mean(valid_logs) if valid_logs else 0.0
    avg_aff1_climb = mean(float(r["final_aff_1_belief"][1]) for r in run_outputs)
    avg_aff2_climb = mean(float(r["final_aff_2_belief"][1]) for r in run_outputs)
    return {
        "success_rate": success_rate,
        "avg_steps_to_success": avg_steps,
        "avg_exploratory_actions": avg_exploratory,
        "avg_exploitative_actions": avg_exploitative,
        "avg_final_affordance_entropy": avg_aff_entropy,
        "avg_final_aff1_climbable_belief": avg_aff1_climb,
        "avg_final_aff2_climbable_belief": avg_aff2_climb,
    }


def summarize_learning_diagnostics(
    run_outputs: List[Dict], metric_key: str = "b_kl_divergence"
) -> Dict[str, float]:
    """Summarize convergence diagnostics for a scalar per-episode metric."""
    if not run_outputs:
        return {"initial_metric": 0.0, "final_metric": 0.0, "delta_metric": 0.0}

    values = [float(r.get(metric_key, 0.0)) for r in run_outputs]
    return {
        "initial_metric": values[0],
        "final_metric": values[-1],
        "delta_metric": values[-1] - values[0],
    }


def episodes_to_disambiguation(
    run_outputs: List[Dict], threshold: float = 0.5
) -> int:
    """Return first episode where belief difference exceeds threshold."""
    for i, run in enumerate(run_outputs, start=1):
        diff = abs(float(run["final_aff_1_belief"][1]) - float(run["final_aff_2_belief"][1]))
        if diff >= threshold:
            return i
    return len(run_outputs)


def plot_phase1_5_beliefs(
    run_outputs: List[Dict], out_path: str, title: str
) -> bool:
    """Plot two-object belief evolution and save to file."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    aff1 = [float(r["final_aff_1_belief"][1]) for r in run_outputs]
    aff2 = [float(r["final_aff_2_belief"][1]) for r in run_outputs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, aff1, label="object_1 climbable belief")
    plt.plot(x, aff2, label="object_2 climbable belief")
    plt.xlabel("Episode")
    plt.ylabel("Belief")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_phase1_5_entropy(
    run_outputs: List[Dict], out_path: str, title: str
) -> bool:
    """Plot affordance entropy evolution and save to file."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    entropies = [float(r["logs"][-1].affordance_entropy_total) for r in run_outputs if r["logs"]]
    x = x[: len(entropies)]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, entropies, label="total affordance entropy")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_learning_kl_over_episodes(
    run_outputs: List[Dict], out_path: str, title: str, metric_key: str = "b_kl_divergence"
) -> bool:
    """Plot a scalar learning metric over episodes."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    y = [float(r.get(metric_key, 0.0)) for r in run_outputs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, label=metric_key)
    plt.xlabel("Episode")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def exploratory_ratio_window(run_outputs: List[Dict], window_size: int = 5) -> Dict[str, float]:
    """Return exploratory action ratio in early vs late windows."""
    if not run_outputs:
        return {"early_ratio": 0.0, "late_ratio": 0.0}

    head = run_outputs[:window_size]
    tail = run_outputs[-window_size:]

    def _ratio(slice_runs: List[Dict]) -> float:
        exploratory = sum(int(r.get("exploratory_actions", 0)) for r in slice_runs)
        exploitative = sum(int(r.get("exploitative_actions", 0)) for r in slice_runs)
        denom = exploratory + exploitative
        return float(exploratory / denom) if denom > 0 else 0.0

    return {"early_ratio": _ratio(head), "late_ratio": _ratio(tail)}
