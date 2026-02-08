"""Stage A demo: validate Phase 2 environment outcome generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from env_phase2.actions import CLIMB_OBJECT_1, CLIMB_OBJECT_2
from env_phase2.io import load_world


def run_demo(episodes: int = 500):
    world_path = Path("env_phase2/worlds/phase2_train_world.json")
    world = load_world(world_path, seed=13)

    stats = {
        CLIMB_OBJECT_1: defaultdict(int),
        CLIMB_OBJECT_2: defaultdict(int),
    }

    for action in (CLIMB_OBJECT_1, CLIMB_OBJECT_2):
        for _ in range(episodes):
            world.reset()
            obs, done, _ = world.step(action)
            stats[action]["reach_yes"] += int(obs["can_reach"] == 1)
            stats[action]["reach_no"] += int(obs["can_reach"] == 0)
            stats[action]["climb_success"] += int(obs["climb_result"] == 1)
            stats[action]["climb_fail"] += int(obs["climb_result"] == 0)
            stats[action]["done"] += int(done)

    print("Stage A demo: empirical one-step outcomes")
    for action in (CLIMB_OBJECT_1, CLIMB_OBJECT_2):
        total = max(stats[action]["climb_success"] + stats[action]["climb_fail"], 1)
        success_rate = stats[action]["climb_success"] / total
        reach_rate = stats[action]["reach_yes"] / total
        action_name = "climb_object_1" if action == CLIMB_OBJECT_1 else "climb_object_2"
        print(
            f"- {action_name}: success_rate={success_rate:.3f} reach_yes_rate={reach_rate:.3f} "
            f"counts={dict(stats[action])}"
        )

    return stats


if __name__ == "__main__":
    run_demo()
