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


def summarize_phase2_runs(run_outputs: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics for Phase 2 + Level 2 experiments."""
    if not run_outputs:
        return {
            "success_rate": 0.0,
            "avg_steps_to_success": 0.0,
            "avg_final_p_flat": 0.0,
            "avg_final_p_factorized": 0.0,
            "avg_final_cum_fe_flat": 0.0,
            "avg_final_cum_fe_factorized": 0.0,
            "avg_episode_fe_flat": 0.0,
            "avg_episode_fe_factorized": 0.0,
            "avg_episode_fe_per_step_flat": 0.0,
            "avg_episode_fe_per_step_factorized": 0.0,
            "avg_final_complexity_flat": 0.0,
            "avg_final_complexity_factorized": 0.0,
            "avg_final_score_flat": 0.0,
            "avg_final_score_factorized": 0.0,
            "factorized_selected_rate": 0.0,
        }

    success_rate = sum(1 for r in run_outputs if r["success"]) / max(len(run_outputs), 1)
    avg_steps = mean(r["steps_to_success"] for r in run_outputs)
    avg_p_flat = mean(float(r["model_posterior_final"][0]) for r in run_outputs)
    avg_p_factor = mean(float(r["model_posterior_final"][1]) for r in run_outputs)
    avg_fe_flat = mean(float(r["cumulative_fe_flat"]) for r in run_outputs)
    avg_fe_factor = mean(float(r["cumulative_fe_factorized"]) for r in run_outputs)
    episode_fe_flat = [
        float(sum(getattr(log, "fe_inc_flat", 0.0) for log in r.get("logs", []))) for r in run_outputs
    ]
    episode_fe_factor = [
        float(sum(getattr(log, "fe_inc_factorized", 0.0) for log in r.get("logs", []))) for r in run_outputs
    ]
    episode_steps = [max(len(r.get("logs", [])), 1) for r in run_outputs]
    episode_fe_rate_flat = [episode_fe_flat[i] / episode_steps[i] for i in range(len(run_outputs))]
    episode_fe_rate_factor = [episode_fe_factor[i] / episode_steps[i] for i in range(len(run_outputs))]
    avg_complexity_flat = mean(float(r.get("complexity_flat", 0.0)) for r in run_outputs)
    avg_complexity_factor = mean(float(r.get("complexity_factorized", 0.0)) for r in run_outputs)
    factorized_selected = sum(1 for r in run_outputs if r["selected_family_final"] == "factorized")
    factorized_selected_rate = factorized_selected / max(len(run_outputs), 1)

    return {
        "success_rate": success_rate,
        "avg_steps_to_success": avg_steps,
        "avg_final_p_flat": avg_p_flat,
        "avg_final_p_factorized": avg_p_factor,
        "avg_final_cum_fe_flat": avg_fe_flat,
        "avg_final_cum_fe_factorized": avg_fe_factor,
        "avg_episode_fe_flat": mean(episode_fe_flat),
        "avg_episode_fe_factorized": mean(episode_fe_factor),
        "avg_episode_fe_per_step_flat": mean(episode_fe_rate_flat),
        "avg_episode_fe_per_step_factorized": mean(episode_fe_rate_factor),
        "avg_final_complexity_flat": avg_complexity_flat,
        "avg_final_complexity_factorized": avg_complexity_factor,
        "avg_final_score_flat": avg_fe_flat + avg_complexity_flat,
        "avg_final_score_factorized": avg_fe_factor + avg_complexity_factor,
        "factorized_selected_rate": factorized_selected_rate,
    }


def plot_phase2_model_evidence(run_outputs: List[Dict], out_path: str, title: str) -> bool:
    """Plot cumulative free energy trajectories by model family."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    fe_flat = [float(r["cumulative_fe_flat"]) for r in run_outputs]
    fe_factor = [float(r["cumulative_fe_factorized"]) for r in run_outputs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, fe_flat, label="flat cumulative free energy")
    plt.plot(x, fe_factor, label="factorized cumulative free energy")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative free energy (lower is better)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_phase2_model_evidence_per_step(run_outputs: List[Dict], out_path: str, title: str) -> bool:
    """Plot per-episode free-energy rate (mean FE per step) by model family."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    y_flat = []
    y_factor = []
    for run in run_outputs:
        logs = run.get("logs", [])
        n_steps = max(len(logs), 1)
        fe_flat = float(sum(getattr(log, "fe_inc_flat", 0.0) for log in logs))
        fe_factor = float(sum(getattr(log, "fe_inc_factorized", 0.0) for log in logs))
        y_flat.append(fe_flat / n_steps)
        y_factor.append(fe_factor / n_steps)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y_flat, label="flat mean FE per step")
    plt.plot(x, y_factor, label="factorized mean FE per step")
    plt.xlabel("Episode")
    plt.ylabel("Mean free energy per step (lower is better)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_phase2_model_posterior(run_outputs: List[Dict], out_path: str, title: str) -> bool:
    """Plot posterior preference over model families by episode."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    p_flat = [float(r["model_posterior_final"][0]) for r in run_outputs]
    p_factor = [float(r["model_posterior_final"][1]) for r in run_outputs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, p_flat, label="P(flat)")
    plt.plot(x, p_factor, label="P(factorized)")
    plt.xlabel("Episode")
    plt.ylabel("Posterior probability")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_phase2_transfer_steps(
    transferred_runs: List[Dict], fresh_runs: List[Dict], out_path: str, title: str
) -> bool:
    """Plot steps-to-success for transferred vs fresh agents."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, max(len(transferred_runs), len(fresh_runs)) + 1))
    y_transferred = [float(r["steps_to_success"]) for r in transferred_runs]
    y_fresh = [float(r["steps_to_success"]) for r in fresh_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x[: len(y_transferred)], y_transferred, label="transferred")
    plt.plot(x[: len(y_fresh)], y_fresh, label="fresh")
    plt.xlabel("Episode")
    plt.ylabel("Steps to success")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def summarize_phase3_runs(run_outputs: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics for Phase 3 latent-slot experiments."""
    if not run_outputs:
        return {
            "success_rate": 0.0,
            "avg_steps_to_success": 0.0,
            "avg_prediction_error": 0.0,
            "avg_exploratory_actions": 0.0,
            "avg_exploitative_actions": 0.0,
            "reserve_slot_dominance_rate": 0.0,
            "avg_slot_kl_max": 0.0,
            "avg_object_slot_entropy": 0.0,
        }

    success_rate = sum(1 for r in run_outputs if r["success"]) / max(len(run_outputs), 1)
    avg_steps = mean(float(r["steps_to_success"]) for r in run_outputs)
    avg_pe = mean(float(r.get("avg_prediction_error", 0.0)) for r in run_outputs)
    avg_exploratory = mean(float(r.get("exploratory_actions", 0.0)) for r in run_outputs)
    avg_exploitative = mean(float(r.get("exploitative_actions", 0.0)) for r in run_outputs)
    dominance_rate = (
        sum(1 for r in run_outputs if bool(r.get("reserve_slot_became_dominant", False)))
        / max(len(run_outputs), 1)
    )

    slot_kl_max = []
    object_entropies = []
    for r in run_outputs:
        vals = r.get("final_slot_kl_per_slot")
        if vals is None:
            continue
        arr = [float(v) for v in vals]
        if arr:
            slot_kl_max.append(max(arr))
        ent = r.get("final_slot_entropy_per_object")
        if ent is not None:
            ent_arr = [float(v) for v in ent]
            if ent_arr:
                object_entropies.append(sum(ent_arr) / len(ent_arr))
    avg_slot_kl_max = mean(slot_kl_max) if slot_kl_max else 0.0
    avg_obj_entropy = mean(object_entropies) if object_entropies else 0.0

    return {
        "success_rate": float(success_rate),
        "avg_steps_to_success": float(avg_steps),
        "avg_prediction_error": float(avg_pe),
        "avg_exploratory_actions": float(avg_exploratory),
        "avg_exploitative_actions": float(avg_exploitative),
        "reserve_slot_dominance_rate": float(dominance_rate),
        "avg_slot_kl_max": float(avg_slot_kl_max),
        "avg_object_slot_entropy": float(avg_obj_entropy),
    }


def plot_phase3_prediction_error(run_outputs: List[Dict], out_path: str, title: str) -> bool:
    """Plot average stepwise prediction error per episode."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    y = [float(r.get("avg_prediction_error", 0.0)) for r in run_outputs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, label="avg prediction error")
    plt.xlabel("Episode")
    plt.ylabel("Negative log likelihood")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_phase3_slot_specialization(run_outputs: List[Dict], out_path: str, title: str) -> bool:
    """Plot max slot KL-to-uniform per episode (higher means sharper slot)."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    y = []
    for run in run_outputs:
        vals = run.get("final_slot_kl_per_slot")
        if vals is None:
            y.append(0.0)
            continue
        arr = [float(v) for v in vals]
        y.append(max(arr) if arr else 0.0)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, label="max slot KL to uniform")
    plt.xlabel("Episode")
    plt.ylabel("KL")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_phase3_dominance_rate(run_outputs: List[Dict], out_path: str, title: str) -> bool:
    """Plot cumulative rate of reserve-slot dominance over episodes."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, len(run_outputs) + 1))
    flags = [1.0 if bool(r.get("reserve_slot_became_dominant", False)) else 0.0 for r in run_outputs]
    cum = []
    running = 0.0
    for i, f in enumerate(flags, start=1):
        running += f
        cum.append(running / float(i))

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, cum, label="cumulative reserve dominance rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_phase3_transfer_comparison(
    transferred_runs: List[Dict], fresh_runs: List[Dict], out_path: str, title: str
) -> bool:
    """Plot exploratory actions per episode for transferred vs fresh agents."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        print(_MATPLOTLIB_MISSING_MSG)
        return False

    x = list(range(1, max(len(transferred_runs), len(fresh_runs)) + 1))
    y_t = [float(r.get("exploratory_actions", 0.0)) for r in transferred_runs]
    y_f = [float(r.get("exploratory_actions", 0.0)) for r in fresh_runs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x[: len(y_t)], y_t, label="transferred exploratory actions")
    plt.plot(x[: len(y_f)], y_f, label="fresh exploratory actions")
    plt.xlabel("Episode")
    plt.ylabel("Exploratory actions")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True
