"""Phase 3 active-inference agent with emergent latent-cause recruitment.

Key idea: the agent has reserve latent slots with no predefined semantics.
Prediction error can recruit a reserve slot, and Dirichlet learning specializes
that slot into a reusable latent cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

import numpy as np

from agent.inference_utils import EPS, normalize, softmax
from env_phase3.actions import DO_NOTHING, action_name


def _entropy(prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=float), EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def _kl_to_uniform(prob: np.ndarray) -> float:
    p = normalize(np.asarray(prob, dtype=float))
    u = np.ones_like(p, dtype=float) / float(len(p))
    return float(np.sum(p * (np.log(np.clip(p, EPS, 1.0)) - np.log(u))))


def _symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    p_n = normalize(np.asarray(p, dtype=float))
    q_n = normalize(np.asarray(q, dtype=float))
    kl_pq = float(np.sum(p_n * (np.log(np.clip(p_n, EPS, 1.0)) - np.log(np.clip(q_n, EPS, 1.0)))))
    kl_qp = float(np.sum(q_n * (np.log(np.clip(q_n, EPS, 1.0)) - np.log(np.clip(p_n, EPS, 1.0)))))
    return 0.5 * (kl_pq + kl_qp)


def _obs_index(can_reach: int, climb_result: int) -> int:
    # Ordering: (reach=1,climb=1),(0,1),(1,0),(0,0)
    if can_reach == 1 and climb_result == 1:
        return 0
    if can_reach == 0 and climb_result == 1:
        return 1
    if can_reach == 1 and climb_result == 0:
        return 2
    return 3


class TransferStatePhase3(TypedDict):
    """Portable state payload for transfer tests."""

    a_outcome_dirichlet: np.ndarray
    slot_usage: np.ndarray
    reward_estimate: np.ndarray


@dataclass
class StepLogPhase3:
    """Per-step diagnostics for emergence analysis."""

    step: int
    action: str
    can_reach_obs: str
    climb_result_obs: str
    acted_object_slot: int
    prediction_error: float
    recruited_slot: int
    recruited_from_reserve: bool
    dominant_slot_after_update: int
    dominant_slot_prob: float
    exploratory_action: bool


class EpisodeResultPhase3(TypedDict):
    """Complete episode output."""

    logs: List[StepLogPhase3]
    success: bool
    steps_to_success: int
    exploratory_actions: int
    exploitative_actions: int
    avg_prediction_error: float
    final_slot_entropy_per_object: np.ndarray
    final_slot_kl_per_slot: np.ndarray
    reserve_slot_became_dominant: bool
    trace: List[Dict[str, Any]]


class ActiveInferenceEmergentAffordanceAgent:
    """Agent with open latent object slots and recruitment-by-surprisal."""

    def __init__(
        self,
        num_objects: int,
        num_slots: int = 4,
        policy_precision: float = 5.0,
        c_pref: np.ndarray | None = None,
        prior_strength: float = 1.0,
        recruit_threshold: float = 1.2,
        recruit_scale: float = 2.0,
        reserve_prior_mass: float = 0.2,
        usage_temperature: float = 30.0,
        enable_slot_reduction: bool = True,
        reduction_kl_threshold: float = 0.03,
        dirichlet_decay: float = 0.02,
        seed: int = 0,
    ):
        if num_objects < 2:
            raise ValueError("num_objects must be at least 2")
        if num_slots < 2:
            raise ValueError("num_slots must be at least 2")

        self.num_objects = int(num_objects)
        self.num_slots = int(num_slots)
        self.policy_precision = float(policy_precision)
        self.prior_strength = float(prior_strength)
        self.recruit_threshold = float(recruit_threshold)
        self.recruit_scale = float(recruit_scale)
        self.reserve_prior_mass = float(np.clip(reserve_prior_mass, 0.01, 0.9))
        self.usage_temperature = float(max(usage_temperature, 1.0))
        self.enable_slot_reduction = bool(enable_slot_reduction)
        self.reduction_kl_threshold = float(reduction_kl_threshold)
        self.dirichlet_decay = float(np.clip(dirichlet_decay, 0.0, 0.2))
        self.rng = np.random.default_rng(seed)

        self.c_pref = normalize(np.array([0.01, 0.99], dtype=float) if c_pref is None else c_pref)

        # Dirichlet parameters for p(outcome_joint | latent_slot).
        self.a_outcome_dirichlet = np.ones((self.num_slots, 4), dtype=float) * self.prior_strength
        self.slot_usage = np.zeros(self.num_slots, dtype=float)
        self.reward_estimate = np.zeros(self.num_slots, dtype=float)

        # Slot 0 acts as initially active default; others are reserve capacity.
        self.reserve_slots = np.arange(1, self.num_slots, dtype=int)
        self.q_object_slot = np.zeros((self.num_objects, self.num_slots), dtype=float)
        self._reset_object_beliefs()

    def _reset_object_beliefs(self) -> None:
        """Initialize or reset per-object slot beliefs."""
        active_mass = 1.0 - self.reserve_prior_mass
        reserve_mass = self.reserve_prior_mass / float(max(self.num_slots - 1, 1))
        self.q_object_slot[:] = reserve_mass
        self.q_object_slot[:, 0] = active_mass

    def reset_episode(self, reset_object_beliefs: bool = False) -> None:
        """Reset episode state; keep learned object-slot posteriors by default."""
        if reset_object_beliefs:
            self._reset_object_beliefs()

    def _slot_means(self) -> np.ndarray:
        return self.a_outcome_dirichlet / np.clip(
            self.a_outcome_dirichlet.sum(axis=1, keepdims=True), EPS, None
        )

    def _reserve_preference(self) -> np.ndarray:
        # Less-used slots are preferred during recruitment.
        scaled_usage = self.slot_usage / self.usage_temperature
        pref = np.exp(-scaled_usage)
        return normalize(pref)

    def _predict_joint_obs(self, object_slot_idx: int) -> np.ndarray:
        theta = self._slot_means()
        return normalize(self.q_object_slot[object_slot_idx] @ theta)

    def _pred_can_reach_yes(self, object_slot_idx: int) -> float:
        p_joint = self._predict_joint_obs(object_slot_idx)
        return float(p_joint[0] + p_joint[2])

    def _expected_info_gain_slot(self, object_slot_idx: int) -> float:
        q_prior = self.q_object_slot[object_slot_idx]
        theta = self._slot_means()
        p_obs = normalize(q_prior @ theta)
        h_prior = _entropy(q_prior)
        h_post = 0.0
        for obs_idx in range(4):
            like = theta[:, obs_idx]
            q_post = normalize(q_prior * like)
            h_post += float(p_obs[obs_idx]) * _entropy(q_post)
        return float(h_prior - h_post)

    def _action_terms(self) -> List[Dict[str, float]]:
        terms: List[Dict[str, float]] = []
        terms.append(
            {
                "G": float(-np.log(np.clip(self.c_pref[0], EPS, 1.0))),
                "pragmatic": float(-np.log(np.clip(self.c_pref[0], EPS, 1.0))),
                "epistemic": 0.0,
                "pred_obs_yes": 0.0,
            }
        )
        for obj_idx in range(self.num_objects):
            p_yes = self._pred_can_reach_yes(obj_idx)
            q_o = np.array([1.0 - p_yes, p_yes], dtype=float)
            pragmatic = float(-np.sum(q_o * np.log(np.clip(self.c_pref, EPS, 1.0))))
            epistemic = self._expected_info_gain_slot(obj_idx)
            terms.append(
                {
                    "G": float(pragmatic - epistemic),
                    "pragmatic": pragmatic,
                    "epistemic": float(epistemic),
                    "pred_obs_yes": float(p_yes),
                }
            )
        return terms

    def act(self) -> Dict[str, Any]:
        """Select action by minimizing expected free energy."""
        terms = self._action_terms()
        g = np.array([float(t["G"]) for t in terms], dtype=float)
        q_pi = softmax(-g, precision=self.policy_precision)
        action = int(self.rng.choice(len(q_pi), p=q_pi))
        return {"action": action, "q_pi": q_pi, "terms": terms}

    def _update_object_slot_posterior(
        self,
        object_slot_idx: int,
        obs_idx: int,
    ) -> tuple[np.ndarray, float, int, bool]:
        q_prior = self.q_object_slot[object_slot_idx].copy()
        theta = self._slot_means()
        like = np.clip(theta[:, obs_idx], EPS, 1.0)

        pred = float(np.clip(np.sum(q_prior * like), EPS, 1.0))
        prediction_error = float(-np.log(pred))

        reserve_pref = self._reserve_preference()
        pe_excess = max(prediction_error - self.recruit_threshold, 0.0)
        recruit_bias = self.recruit_scale * pe_excess * reserve_pref
        recruit_bias[0] = 0.0

        log_post = np.log(np.clip(q_prior, EPS, 1.0)) + np.log(like) + recruit_bias
        log_post = log_post - np.max(log_post)
        q_post = normalize(np.exp(log_post))

        reserve_slot = int(np.argmax(q_post[1:]) + 1)
        recruited_from_reserve = bool(
            prediction_error > self.recruit_threshold
            and float(q_post[reserve_slot]) > float(q_prior[reserve_slot]) + 0.05
        )
        return q_post, prediction_error, reserve_slot, recruited_from_reserve

    def _reduce_redundant_slots(self) -> int:
        if not self.enable_slot_reduction:
            return 0
        theta = self._slot_means()
        merges = 0
        for i in range(self.num_slots):
            for j in range(i + 1, self.num_slots):
                if self.slot_usage[j] < 1.0:
                    continue
                sym = _symmetric_kl(theta[i], theta[j])
                if sym < self.reduction_kl_threshold:
                    # Merge j into i and reset j as reserve.
                    self.a_outcome_dirichlet[i] += self.a_outcome_dirichlet[j] - self.prior_strength
                    self.a_outcome_dirichlet[j] = np.ones(4, dtype=float) * self.prior_strength
                    self.slot_usage[i] += self.slot_usage[j]
                    self.slot_usage[j] = 0.0
                    self.reward_estimate[i] = 0.5 * (self.reward_estimate[i] + self.reward_estimate[j])
                    self.reward_estimate[j] = 0.0
                    self.q_object_slot[:, i] += self.q_object_slot[:, j]
                    self.q_object_slot[:, j] = 1e-6
                    self.q_object_slot = self.q_object_slot / np.clip(
                        self.q_object_slot.sum(axis=1, keepdims=True), EPS, None
                    )
                    merges += 1
        return merges

    def update(
        self,
        action: int,
        can_reach_obs: int,
        climb_result_obs: int,
        info: Dict[str, float],
    ) -> Dict[str, Any]:
        """Update slot beliefs and parameters from an observation."""
        if action == DO_NOTHING:
            return {
                "prediction_error": 0.0,
                "recruited_slot": -1,
                "recruited_from_reserve": False,
                "dominant_slot_after_update": int(np.argmax(self.q_object_slot[0])),
                "dominant_slot_prob": float(np.max(self.q_object_slot[0])),
                "slot_merges": 0,
            }

        object_slot_idx = int(action - 1)
        obs_idx = _obs_index(can_reach_obs, climb_result_obs)
        q_post, pe, recruited_slot, recruited_from_reserve = self._update_object_slot_posterior(
            object_slot_idx=object_slot_idx,
            obs_idx=obs_idx,
        )
        self.q_object_slot[object_slot_idx] = q_post

        # Dirichlet learning for outcome model of each slot.
        if self.dirichlet_decay > 0.0:
            self.a_outcome_dirichlet = (
                (1.0 - self.dirichlet_decay) * self.a_outcome_dirichlet
                + self.dirichlet_decay * self.prior_strength
            )
        self.a_outcome_dirichlet[:, obs_idx] += q_post
        self.slot_usage += q_post
        self.reward_estimate = 0.99 * self.reward_estimate + 0.01 * q_post * float(can_reach_obs)

        dominant_slot = int(np.argmax(q_post))
        dominant_prob = float(np.max(q_post))

        merges = self._reduce_redundant_slots()

        return {
            "prediction_error": float(pe),
            "recruited_slot": int(recruited_slot),
            "recruited_from_reserve": bool(recruited_from_reserve),
            "dominant_slot_after_update": dominant_slot,
            "dominant_slot_prob": dominant_prob,
            "slot_merges": int(merges),
        }

    def _slot_specialization_metrics(self) -> Dict[str, np.ndarray]:
        theta = self._slot_means()
        slot_entropy = np.array([_entropy(theta[k]) for k in range(self.num_slots)], dtype=float)
        slot_kl = np.array([_kl_to_uniform(theta[k]) for k in range(self.num_slots)], dtype=float)
        object_slot_entropy = np.array([_entropy(self.q_object_slot[i]) for i in range(self.num_objects)], dtype=float)
        return {
            "slot_entropy": slot_entropy,
            "slot_kl": slot_kl,
            "object_slot_entropy": object_slot_entropy,
        }

    def _make_trace_entry(
        self,
        step_idx: int,
        action: int,
        can_reach_obs: int,
        climb_result_obs: int,
        decision: Dict[str, Any],
        update_info: Dict[str, Any],
        world_info: Dict[str, float],
    ) -> Dict[str, Any]:
        metrics = self._slot_specialization_metrics()
        theta = self._slot_means()
        return {
            "t": int(step_idx),
            "action": action_name(action, self.num_objects),
            "chosen_action": int(action),
            "observation": {
                "can_reach": "yes" if can_reach_obs == 1 else "no",
                "climb_result": "success" if climb_result_obs == 1 else "fail",
            },
            "q_pi": np.asarray(decision["q_pi"], dtype=float).copy(),
            "efe_per_action": np.array([float(t["G"]) for t in decision["terms"]], dtype=float),
            "pragmatic_per_action": np.array([float(t["pragmatic"]) for t in decision["terms"]], dtype=float),
            "epistemic_per_action": np.array([float(t["epistemic"]) for t in decision["terms"]], dtype=float),
            "prediction_error": float(update_info["prediction_error"]),
            "recruited_slot": int(update_info["recruited_slot"]),
            "recruited_from_reserve": bool(update_info["recruited_from_reserve"]),
            "dominant_slot_after_update": int(update_info["dominant_slot_after_update"]),
            "dominant_slot_prob": float(update_info["dominant_slot_prob"]),
            "slot_means": theta.copy(),
            "slot_usage": self.slot_usage.copy(),
            "object_slot_beliefs": self.q_object_slot.copy(),
            "slot_entropy": metrics["slot_entropy"].copy(),
            "slot_kl_to_uniform": metrics["slot_kl"].copy(),
            "object_slot_entropy": metrics["object_slot_entropy"].copy(),
            "slot_merges": int(update_info["slot_merges"]),
            "world_info": dict(world_info),
        }

    def export_transfer_state(self) -> TransferStatePhase3:
        """Export learned latent-cause parameters for transfer."""
        return {
            "a_outcome_dirichlet": self.a_outcome_dirichlet.copy(),
            "slot_usage": self.slot_usage.copy(),
            "reward_estimate": self.reward_estimate.copy(),
        }

    def import_transfer_state(self, payload: TransferStatePhase3) -> None:
        """Import learned latent-cause parameters for transfer."""
        self.a_outcome_dirichlet = np.array(payload["a_outcome_dirichlet"], dtype=float).copy()
        self.slot_usage = np.array(payload["slot_usage"], dtype=float).copy()
        self.reward_estimate = np.array(payload["reward_estimate"], dtype=float).copy()

    def rollout_episode(
        self,
        world,
        max_steps: int = 8,
        enable_trace: bool = False,
        reset_object_beliefs: bool = False,
    ) -> EpisodeResultPhase3:
        """Run one episode."""
        if int(world.num_objects) != self.num_objects:
            raise ValueError(f"World/object mismatch: world={world.num_objects} agent={self.num_objects}")

        self.reset_episode(reset_object_beliefs=reset_object_beliefs)
        world.reset()

        logs: List[StepLogPhase3] = []
        trace: List[Dict[str, Any]] = []
        prediction_errors: List[float] = []
        success_step = None
        reserve_slot_became_dominant = False
        exploratory_actions = 0
        exploitative_actions = 0

        for step_idx in range(max_steps):
            decision = self.act()
            action = int(decision["action"])

            if action == DO_NOTHING:
                exploitative = False
            else:
                obj_idx = int(action - 1)
                expected = np.array([self._pred_can_reach_yes(i) for i in range(self.num_objects)], dtype=float)
                best_obj = int(np.argmax(expected))
                exploitative = obj_idx == best_obj

            obs, done, info = world.step(action)
            can_reach_obs = int(obs["can_reach"])
            climb_result_obs = int(obs["climb_result"])
            update_info = self.update(
                action=action,
                can_reach_obs=can_reach_obs,
                climb_result_obs=climb_result_obs,
                info=info,
            )

            if action != DO_NOTHING:
                prediction_errors.append(float(update_info["prediction_error"]))
                dominant = int(update_info["dominant_slot_after_update"])
                if (
                    dominant > 0
                    and float(update_info["dominant_slot_prob"]) > 0.55
                    and float(self.slot_usage[dominant]) > 10.0
                ):
                    reserve_slot_became_dominant = True
                if exploitative:
                    exploitative_actions += 1
                else:
                    exploratory_actions += 1

            logs.append(
                StepLogPhase3(
                    step=int(step_idx),
                    action=action_name(action, self.num_objects),
                    can_reach_obs="yes" if can_reach_obs == 1 else "no",
                    climb_result_obs="success" if climb_result_obs == 1 else "fail",
                    acted_object_slot=int(info.get("acted_object_slot", -1.0)),
                    prediction_error=float(update_info["prediction_error"]),
                    recruited_slot=int(update_info["recruited_slot"]),
                    recruited_from_reserve=bool(update_info["recruited_from_reserve"]),
                    dominant_slot_after_update=int(update_info["dominant_slot_after_update"]),
                    dominant_slot_prob=float(update_info["dominant_slot_prob"]),
                    exploratory_action=bool(action != DO_NOTHING and not exploitative),
                )
            )

            if enable_trace:
                trace.append(
                    self._make_trace_entry(
                        step_idx=step_idx,
                        action=action,
                        can_reach_obs=can_reach_obs,
                        climb_result_obs=climb_result_obs,
                        decision=decision,
                        update_info=update_info,
                        world_info=info,
                    )
                )

            if done:
                success_step = step_idx + 1
                break

        metrics = self._slot_specialization_metrics()
        return {
            "logs": logs,
            "success": success_step is not None,
            "steps_to_success": success_step if success_step is not None else max_steps,
            "exploratory_actions": int(exploratory_actions),
            "exploitative_actions": int(exploitative_actions),
            "avg_prediction_error": float(np.mean(prediction_errors)) if prediction_errors else 0.0,
            "final_slot_entropy_per_object": metrics["object_slot_entropy"].copy(),
            "final_slot_kl_per_slot": metrics["slot_kl"].copy(),
            "reserve_slot_became_dominant": bool(reserve_slot_became_dominant),
            "trace": trace,
        }
