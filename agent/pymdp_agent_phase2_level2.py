"""Phase 2 + Level 2 active-inference agent with explicit structure learning.

Model family A (flat):
- Per-object local latent parameters only.
- No shared explanatory latent variable.

Model family B (factorized):
- Per-object latent type variable with shared global type parameters.
- Outcomes are conditionally independent of object identity given latent type.

Model comparison uses cumulative variational free energy plus an explicit
complexity (Occam) penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict

import numpy as np

from agent.inference_utils import EPS, normalize, softmax
from env_phase2.actions import CLIMB_OBJECT_1, CLIMB_OBJECT_2, DO_NOTHING, PHASE2_ACTION_NAMES


def _binary_entropy(prob_one: float) -> float:
    p = float(np.clip(prob_one, EPS, 1.0 - EPS))
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def _joint_outcome_log_likelihood(
    obs_reach_yes: int,
    obs_climb_success: int,
    p_reach_yes_and_climb_success: float,
    p_reach_no_and_climb_success: float,
    p_reach_yes_and_climb_fail: float,
    p_reach_no_and_climb_fail: float,
) -> float:
    table = {
        (1, 1): p_reach_yes_and_climb_success,
        (0, 1): p_reach_no_and_climb_success,
        (1, 0): p_reach_yes_and_climb_fail,
        (0, 0): p_reach_no_and_climb_fail,
    }
    p = float(np.clip(table[(obs_reach_yes, obs_climb_success)], EPS, 1.0))
    return float(np.log(p))


class FamilyActionEfe(TypedDict):
    """Expected free energy terms for a single action."""

    G: float
    pragmatic: float
    epistemic: float
    pred_obs_yes: float


class ActDecision(TypedDict):
    """Decision output from the act() method."""

    action: int
    selected_family: str
    q_pi: np.ndarray
    terms: Dict[str, List[FamilyActionEfe]]


class TransferState(TypedDict):
    """State for transfer learning between episodes."""

    model_cumulative_fe: np.ndarray
    model_complexity: np.ndarray
    observation_count: np.ndarray
    model_posterior: np.ndarray
    flat: Dict[str, np.ndarray]
    factorized: Dict[str, np.ndarray]


class BaseFamilyModel:
    """Common interface for Level 2 competing model families."""

    complexity_params: int

    def reset_episode(self) -> None:
        """Reset episode-specific beliefs."""
        raise NotImplementedError

    def expected_free_energy_terms(self, action: int, c_pref: np.ndarray) -> FamilyActionEfe:
        """Compute expected free energy for an action."""
        raise NotImplementedError

    def update(self, action: int, obs_yes: int, obs_climb_success: int) -> float:
        """Update beliefs and return free energy increment."""
        raise NotImplementedError

    def get_transfer_state(self) -> Dict[str, np.ndarray]:
        """Export state for transfer learning."""
        raise NotImplementedError

    def set_transfer_state(self, state: Dict[str, np.ndarray]) -> None:
        """Import state from transfer learning."""
        raise NotImplementedError


class FlatModelFamily(BaseFamilyModel):
    """Model A: object-identity model with independent observation channels."""

    complexity_params = 4

    def __init__(self, prior_strength: float = 2.0):
        self.prior_strength = prior_strength
        # Per object: p(can_reach=yes | object), p(climb_success=1 | object).
        # Flat family assumes conditional independence between observation channels.
        self.alpha_reach = np.ones(2, dtype=float) * prior_strength
        self.beta_reach = np.ones(2, dtype=float) * prior_strength
        self.alpha_climb = np.ones(2, dtype=float) * prior_strength
        self.beta_climb = np.ones(2, dtype=float) * prior_strength
        self.q_height = np.array([1.0, 0.0], dtype=float)
        self.q_object_type = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)

    def reset_episode(self) -> None:
        """Reset episode-specific beliefs."""
        self.q_height = np.array([1.0, 0.0], dtype=float)
        self.q_object_type = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)

    def _obj_idx(self, action: int) -> int:
        if action == CLIMB_OBJECT_1:
            return 0
        if action == CLIMB_OBJECT_2:
            return 1
        raise ValueError("Action does not target an object.")

    def _means(self, idx: int) -> tuple[float, float]:
        p_reach = float(self.alpha_reach[idx] / (self.alpha_reach[idx] + self.beta_reach[idx]))
        p_climb = float(self.alpha_climb[idx] / (self.alpha_climb[idx] + self.beta_climb[idx]))
        return p_reach, p_climb

    def _joint_outcome_probs(self, action: int) -> tuple[float, float, float, float]:
        """Return p(r=1,c=1), p(r=0,c=1), p(r=1,c=0), p(r=0,c=0)."""

        if action == DO_NOTHING:
            p_high = float(self.q_height[1])
            return 0.0, 0.0, p_high, 1.0 - p_high

        idx = self._obj_idx(action)
        p_reach, p_climb = self._means(idx)
        p_r1_c1 = p_reach * p_climb
        p_r0_c1 = (1.0 - p_reach) * p_climb
        p_r1_c0 = p_reach * (1.0 - p_climb)
        p_r0_c0 = (1.0 - p_reach) * (1.0 - p_climb)
        return p_r1_c1, p_r0_c1, p_r1_c0, p_r0_c0

    def _predict_reach_yes(self, action: int) -> float:
        p_r1_c1, _, p_r1_c0, _ = self._joint_outcome_probs(action)
        return float(p_r1_c1 + p_r1_c0)

    def expected_free_energy_terms(self, action: int, c_pref: np.ndarray) -> FamilyActionEfe:
        p_yes = self._predict_reach_yes(action)
        q_o = np.array([1.0 - p_yes, p_yes], dtype=float)
        pragmatic = float(-np.sum(q_o * np.log(np.clip(c_pref, EPS, 1.0))))

        epistemic = 0.0
        if action in (CLIMB_OBJECT_1, CLIMB_OBJECT_2):
            idx = self._obj_idx(action)
            p_reach, p_climb = self._means(idx)
            prior_uncertainty = _binary_entropy(p_reach) + _binary_entropy(p_climb)

            p_outcomes = self._joint_outcome_probs(action)
            expected_post = 0.0
            for (obs_reach, obs_climb), p_obs in [
                ((1, 1), p_outcomes[0]),
                ((0, 1), p_outcomes[1]),
                ((1, 0), p_outcomes[2]),
                ((0, 0), p_outcomes[3]),
            ]:
                if p_obs <= EPS:
                    continue
                a_r = self.alpha_reach[idx] + float(obs_reach)
                b_r = self.beta_reach[idx] + float(1 - obs_reach)
                a_c = self.alpha_climb[idx] + float(obs_climb)
                b_c = self.beta_climb[idx] + float(1 - obs_climb)
                p_r_post = float(a_r / (a_r + b_r))
                p_c_post = float(a_c / (a_c + b_c))
                expected_post += p_obs * (_binary_entropy(p_r_post) + _binary_entropy(p_c_post))

            epistemic = float(prior_uncertainty - expected_post)

        return {
            "G": float(pragmatic - epistemic),
            "pragmatic": pragmatic,
            "epistemic": float(epistemic),
            "pred_obs_yes": float(p_yes),
        }

    def update(self, action: int, obs_yes: int, obs_climb_success: int) -> float:
        p = self._joint_outcome_probs(action)
        fe_increment = float(
            -_joint_outcome_log_likelihood(
                obs_reach_yes=obs_yes,
                obs_climb_success=obs_climb_success,
                p_reach_yes_and_climb_success=p[0],
                p_reach_no_and_climb_success=p[1],
                p_reach_yes_and_climb_fail=p[2],
                p_reach_no_and_climb_fail=p[3],
            )
        )

        self.q_height = np.array([1.0 - obs_yes, obs_yes], dtype=float)

        if action in (CLIMB_OBJECT_1, CLIMB_OBJECT_2):
            idx = self._obj_idx(action)
            self.alpha_reach[idx] += float(obs_yes)
            self.beta_reach[idx] += float(1 - obs_yes)
            self.alpha_climb[idx] += float(obs_climb_success)
            self.beta_climb[idx] += float(1 - obs_climb_success)

        return fe_increment

    def get_transfer_state(self) -> Dict[str, np.ndarray]:
        return {
            "alpha_reach": self.alpha_reach.copy(),
            "beta_reach": self.beta_reach.copy(),
            "alpha_climb": self.alpha_climb.copy(),
            "beta_climb": self.beta_climb.copy(),
        }

    def set_transfer_state(self, state: Dict[str, np.ndarray]) -> None:
        self.alpha_reach = state["alpha_reach"].copy()
        self.beta_reach = state["beta_reach"].copy()
        self.alpha_climb = state["alpha_climb"].copy()
        self.beta_climb = state["beta_climb"].copy()


class FactorizedModelFamily(BaseFamilyModel):
    """Model B: shared latent type variable with global type-conditioned stats."""

    complexity_params = 6

    def __init__(self, prior_strength: float = 2.0):
        self.prior_strength = prior_strength
        # Type-conditioned shared parameters for success and gain.
        self.alpha_success_type = np.array(
            [3.0 * prior_strength, 1.0 * prior_strength], dtype=float
        )
        self.beta_success_type = np.array(
            [1.0 * prior_strength, 3.0 * prior_strength], dtype=float
        )
        self.alpha_gain_type = np.array(
            [3.0 * prior_strength, 1.0 * prior_strength], dtype=float
        )
        self.beta_gain_type = np.array(
            [1.0 * prior_strength, 3.0 * prior_strength], dtype=float
        )
        # Object-specific posterior over hidden type z_j in {0,1}.
        self.q_object_type = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
        self.q_height = np.array([1.0, 0.0], dtype=float)

    def reset_episode(self) -> None:
        """Reset episode-specific beliefs."""
        self.q_height = np.array([1.0, 0.0], dtype=float)

    def _obj_idx(self, action: int) -> int:
        if action == CLIMB_OBJECT_1:
            return 0
        if action == CLIMB_OBJECT_2:
            return 1
        raise ValueError("Action does not target an object.")

    def _type_means(self) -> tuple[np.ndarray, np.ndarray]:
        p_success = self.alpha_success_type / (self.alpha_success_type + self.beta_success_type)
        p_gain = self.alpha_gain_type / (self.alpha_gain_type + self.beta_gain_type)
        return p_success, p_gain

    def _joint_outcome_probs_given_type(self, type_idx: int) -> tuple[float, float, float, float]:
        p_success, p_gain = self._type_means()
        p_s = float(p_success[type_idx])
        p_g = float(p_gain[type_idx])
        p_low = float(self.q_height[0])
        p_high = float(self.q_height[1])

        p_r1_c1 = p_low * p_s * p_g + p_high * p_s
        p_r0_c1 = p_low * p_s * (1.0 - p_g)
        p_r1_c0 = p_high * (1.0 - p_s)
        p_r0_c0 = p_low * (1.0 - p_s)
        return p_r1_c1, p_r0_c1, p_r1_c0, p_r0_c0

    def _joint_outcome_probs(self, action: int) -> tuple[float, float, float, float]:
        if action == DO_NOTHING:
            p_high = float(self.q_height[1])
            return 0.0, 0.0, p_high, 1.0 - p_high

        idx = self._obj_idx(action)
        probs = np.zeros(4, dtype=float)
        for type_idx, p_type in enumerate(self.q_object_type[idx]):
            probs += p_type * np.array(self._joint_outcome_probs_given_type(type_idx), dtype=float)
        return float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3])

    def _predict_reach_yes(self, action: int) -> float:
        p_r1_c1, _, p_r1_c0, _ = self._joint_outcome_probs(action)
        return float(p_r1_c1 + p_r1_c0)

    def _posterior_type_given_obs(
        self, idx: int, obs_yes: int, obs_climb_success: int
    ) -> np.ndarray:
        q_prior = self.q_object_type[idx]
        like = np.zeros(2, dtype=float)
        for type_idx in (0, 1):
            p = self._joint_outcome_probs_given_type(type_idx)
            like[type_idx] = np.exp(
                _joint_outcome_log_likelihood(
                    obs_reach_yes=obs_yes,
                    obs_climb_success=obs_climb_success,
                    p_reach_yes_and_climb_success=p[0],
                    p_reach_no_and_climb_success=p[1],
                    p_reach_yes_and_climb_fail=p[2],
                    p_reach_no_and_climb_fail=p[3],
                )
            )
        return normalize(q_prior * like)

    def expected_free_energy_terms(self, action: int, c_pref: np.ndarray) -> FamilyActionEfe:
        p_yes = self._predict_reach_yes(action)
        q_o = np.array([1.0 - p_yes, p_yes], dtype=float)
        pragmatic = float(-np.sum(q_o * np.log(np.clip(c_pref, EPS, 1.0))))

        epistemic = 0.0
        if action in (CLIMB_OBJECT_1, CLIMB_OBJECT_2):
            idx = self._obj_idx(action)
            prior_h = _binary_entropy(float(self.q_object_type[idx, 1]))

            expected_post_h = 0.0
            p_outcomes = self._joint_outcome_probs(action)
            for (obs_reach, obs_climb), p_obs in [
                ((1, 1), p_outcomes[0]),
                ((0, 1), p_outcomes[1]),
                ((1, 0), p_outcomes[2]),
                ((0, 0), p_outcomes[3]),
            ]:
                if p_obs <= EPS:
                    continue
                q_post = self._posterior_type_given_obs(idx, obs_reach, obs_climb)
                expected_post_h += p_obs * _binary_entropy(float(q_post[1]))

            epistemic = float(prior_h - expected_post_h)

        return {
            "G": float(pragmatic - epistemic),
            "pragmatic": pragmatic,
            "epistemic": float(epistemic),
            "pred_obs_yes": float(p_yes),
        }

    def update(self, action: int, obs_yes: int, obs_climb_success: int) -> float:
        p = self._joint_outcome_probs(action)
        fe_increment = float(
            -_joint_outcome_log_likelihood(
                obs_reach_yes=obs_yes,
                obs_climb_success=obs_climb_success,
                p_reach_yes_and_climb_success=p[0],
                p_reach_no_and_climb_success=p[1],
                p_reach_yes_and_climb_fail=p[2],
                p_reach_no_and_climb_fail=p[3],
            )
        )

        prior_low = float(self.q_height[0])
        self.q_height = np.array([1.0 - obs_yes, obs_yes], dtype=float)

        if action in (CLIMB_OBJECT_1, CLIMB_OBJECT_2):
            idx = self._obj_idx(action)
            q_post = self._posterior_type_given_obs(idx, obs_yes, obs_climb_success)
            self.q_object_type[idx] = q_post

            self.alpha_success_type += q_post * float(obs_climb_success)
            self.beta_success_type += q_post * float(1 - obs_climb_success)
            if obs_climb_success == 1:
                self.alpha_gain_type += q_post * float(prior_low * obs_yes)
                self.beta_gain_type += q_post * float(prior_low * (1 - obs_yes))

        return fe_increment

    def get_transfer_state(self) -> Dict[str, np.ndarray]:
        return {
            "alpha_success_type": self.alpha_success_type.copy(),
            "beta_success_type": self.beta_success_type.copy(),
            "alpha_gain_type": self.alpha_gain_type.copy(),
            "beta_gain_type": self.beta_gain_type.copy(),
            "q_object_type": self.q_object_type.copy(),
        }

    def set_transfer_state(self, state: Dict[str, np.ndarray]) -> None:
        self.alpha_success_type = state["alpha_success_type"].copy()
        self.beta_success_type = state["beta_success_type"].copy()
        self.alpha_gain_type = state["alpha_gain_type"].copy()
        self.beta_gain_type = state["beta_gain_type"].copy()
        self.q_object_type = state["q_object_type"].copy()


@dataclass
class StepLogPhase2:
    """Log entry for a single step in Phase 2."""
    step: int
    action: str
    can_reach_obs: str
    climb_result_obs: str
    selected_family: str
    model_post_flat: float
    model_post_factorized: float
    g_flat_do_nothing: float
    g_flat_obj1: float
    g_flat_obj2: float
    g_factor_do_nothing: float
    g_factor_obj1: float
    g_factor_obj2: float
    fe_inc_flat: float
    fe_inc_factorized: float
    complexity_flat: float
    complexity_factorized: float


class EpisodeResultPhase2(TypedDict):
    """Results from a complete Phase 2 episode."""

    logs: List[StepLogPhase2]
    success: bool
    steps_to_success: int
    cumulative_fe_flat: float
    cumulative_fe_factorized: float
    complexity_flat: float
    complexity_factorized: float
    model_posterior_final: np.ndarray
    selected_family_final: str


class ActiveInferenceStructureLearningAgent:
    """Active-inference controller with online Level 2 model selection."""

    def __init__(
        self,
        policy_precision: float = 5.0,
        model_precision: float = 1.0,
        occam_scale: float = 1.0,
        model_prior: np.ndarray | None = None,
        c_pref: np.ndarray | None = None,
    ):
        self.policy_precision = policy_precision
        self.model_precision = model_precision
        self.occam_scale = occam_scale
        self.c_pref = normalize(np.array([0.01, 0.99], dtype=float) if c_pref is None else c_pref)

        self.model_names: List[Literal["flat", "factorized"]] = ["flat", "factorized"]
        self.families: Dict[str, BaseFamilyModel] = {
            "flat": FlatModelFamily(prior_strength=2.0),
            "factorized": FactorizedModelFamily(prior_strength=2.0),
        }

        if model_prior is None:
            model_prior = np.array([0.6, 0.4], dtype=float)
        self.model_log_prior = np.log(normalize(model_prior))

        self.model_cumulative_fe = np.zeros(2, dtype=float)
        self.model_complexity = np.zeros(2, dtype=float)
        self.observation_count = 0
        self.model_posterior = normalize(np.exp(self.model_log_prior))

    def reset_episode(self) -> None:
        """Reset episode-specific beliefs for all model families."""
        for fam in self.families.values():
            fam.reset_episode()

    def selected_family(self) -> str:
        """Return name of currently selected model family."""
        return self.model_names[int(np.argmax(self.model_posterior))]

    def _update_model_posterior(self) -> None:
        n = max(self.observation_count, 1)
        k_flat = float(self.families["flat"].complexity_params)
        k_factorized = float(self.families["factorized"].complexity_params)
        self.model_complexity[0] = self.occam_scale * 0.5 * k_flat * np.log(n + 1.0)
        self.model_complexity[1] = self.occam_scale * 0.5 * k_factorized * np.log(n + 1.0)

        score = self.model_cumulative_fe + self.model_complexity
        log_post = self.model_log_prior - self.model_precision * score
        centered = log_post - np.max(log_post)
        self.model_posterior = normalize(np.exp(centered))

    def _family_terms(self) -> Dict[str, List[FamilyActionEfe]]:
        terms: Dict[str, List[FamilyActionEfe]] = {}
        for name in self.model_names:
            fam = self.families[name]
            terms[name] = [
                fam.expected_free_energy_terms(DO_NOTHING, self.c_pref),
                fam.expected_free_energy_terms(CLIMB_OBJECT_1, self.c_pref),
                fam.expected_free_energy_terms(CLIMB_OBJECT_2, self.c_pref),
            ]
        return terms

    def act(self) -> ActDecision:
        """Select action using active inference."""
        terms = self._family_terms()
        family_name = self.selected_family()

        g = np.array(
            [
                terms[family_name][0]["G"],
                terms[family_name][1]["G"],
                terms[family_name][2]["G"],
            ],
            dtype=float,
        )
        q_pi = softmax(-g, precision=self.policy_precision)
        action_idx = int(np.argmax(q_pi))
        action = [DO_NOTHING, CLIMB_OBJECT_1, CLIMB_OBJECT_2][action_idx]

        return {
            "action": action,
            "selected_family": family_name,
            "q_pi": q_pi,
            "terms": terms,
        }

    def update(self, action: int, obs_yes: int, obs_climb_success: int) -> Dict[str, float]:
        """Update all model families and return metrics."""
        fe_flat = self.families["flat"].update(
            action=action, obs_yes=obs_yes, obs_climb_success=obs_climb_success
        )
        fe_factorized = self.families["factorized"].update(
            action=action,
            obs_yes=obs_yes,
            obs_climb_success=obs_climb_success,
        )

        self.model_cumulative_fe[0] += fe_flat
        self.model_cumulative_fe[1] += fe_factorized
        self.observation_count += 1
        self._update_model_posterior()

        return {
            "fe_flat": fe_flat,
            "fe_factorized": fe_factorized,
            "post_flat": float(self.model_posterior[0]),
            "post_factorized": float(self.model_posterior[1]),
            "complexity_flat": float(self.model_complexity[0]),
            "complexity_factorized": float(self.model_complexity[1]),
        }

    def rollout_episode(self, world, max_steps: int = 8) -> EpisodeResultPhase2:
        """Run complete episode and return results."""
        self.reset_episode()
        world.reset()

        logs: List[StepLogPhase2] = []
        success_step = None

        for step_idx in range(max_steps):
            decision = self.act()
            action = int(decision["action"])
            obs, done, _ = world.step(action)
            obs_yes = int(obs["can_reach"])
            obs_climb_success = int(obs.get("climb_result", 0))
            post = self.update(action=action, obs_yes=obs_yes, obs_climb_success=obs_climb_success)

            terms = decision["terms"]
            logs.append(
                StepLogPhase2(
                    step=step_idx,
                    action=PHASE2_ACTION_NAMES[action],
                    can_reach_obs="yes" if obs_yes == 1 else "no",
                    climb_result_obs="success" if obs_climb_success == 1 else "fail",
                    selected_family=str(decision["selected_family"]),
                    model_post_flat=float(post["post_flat"]),
                    model_post_factorized=float(post["post_factorized"]),
                    g_flat_do_nothing=float(terms["flat"][0]["G"]),
                    g_flat_obj1=float(terms["flat"][1]["G"]),
                    g_flat_obj2=float(terms["flat"][2]["G"]),
                    g_factor_do_nothing=float(terms["factorized"][0]["G"]),
                    g_factor_obj1=float(terms["factorized"][1]["G"]),
                    g_factor_obj2=float(terms["factorized"][2]["G"]),
                    fe_inc_flat=float(post["fe_flat"]),
                    fe_inc_factorized=float(post["fe_factorized"]),
                    complexity_flat=float(post["complexity_flat"]),
                    complexity_factorized=float(post["complexity_factorized"]),
                )
            )

            if done:
                success_step = step_idx + 1
                break

        final_name = self.selected_family()
        return {
            "logs": logs,
            "success": success_step is not None,
            "steps_to_success": success_step if success_step is not None else max_steps,
            "cumulative_fe_flat": float(self.model_cumulative_fe[0]),
            "cumulative_fe_factorized": float(self.model_cumulative_fe[1]),
            "complexity_flat": float(self.model_complexity[0]),
            "complexity_factorized": float(self.model_complexity[1]),
            "model_posterior_final": self.model_posterior.copy(),
            "selected_family_final": final_name,
        }

    def export_transfer_state(self) -> TransferState:
        """Export state for transfer to new environment."""
        return {
            "model_cumulative_fe": self.model_cumulative_fe.copy(),
            "model_complexity": self.model_complexity.copy(),
            "observation_count": np.array([self.observation_count], dtype=float),
            "model_posterior": self.model_posterior.copy(),
            "flat": self.families["flat"].get_transfer_state(),
            "factorized": self.families["factorized"].get_transfer_state(),
        }

    def import_transfer_state(
        self, payload: TransferState, reset_object_beliefs: bool = True
    ) -> None:
        """Import state from previous environment."""
        self.model_cumulative_fe = np.array(
            payload["model_cumulative_fe"], dtype=float
        ).copy()
        self.model_complexity = np.array(
            payload.get("model_complexity", [0.0, 0.0]), dtype=float
        ).copy()
        self.observation_count = int(
            np.array(payload.get("observation_count", [0.0]), dtype=float)[0]
        )
        self.model_posterior = np.array(payload["model_posterior"], dtype=float).copy()
        self.families["flat"].set_transfer_state(payload["flat"])
        self.families["factorized"].set_transfer_state(payload["factorized"])

        if reset_object_beliefs:
            factorized = self.families["factorized"]
            if isinstance(factorized, FactorizedModelFamily):
                factorized.q_object_type = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
