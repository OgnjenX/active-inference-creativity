"""Active-inference agent for Phase 1.5 two-object affordance disambiguation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, TypedDict

import numpy as np

from agent.generative_model import expected_b_from_alpha, init_dirichlet_from_b, kl_b_tensors
from agent.inference_utils import entropy
from agent.inference_utils_phase1_5 import (
    EfeTermsTwoObject,
    expected_free_energy_terms_two_objects,
    infer_posterior_two_objects,
    normalize,
    predict_height_two_objects,
    softmax,
)
from env.actions import CLIMB_OBJECT_1, CLIMB_OBJECT_2, DO_NOTHING, PHASE1_5_ACTION_NAMES


@dataclass
class StepLogTwoObject:
    """Per-step log for two-object inference dynamics."""

    step: int
    action: str
    can_reach_obs: str
    q_height_low: float
    q_height_high: float
    q_aff_1_unknown: float
    q_aff_1_climbable: float
    q_aff_1_not_climbable: float
    q_aff_2_unknown: float
    q_aff_2_climbable: float
    q_aff_2_not_climbable: float
    affordance_entropy_total: float
    q_pi_do_nothing: float
    q_pi_climb_object_1: float
    q_pi_climb_object_2: float
    g_do_nothing: float
    g_climb_object_1: float
    g_climb_object_2: float
    pragmatic_do_nothing: float
    pragmatic_climb_object_1: float
    pragmatic_climb_object_2: float
    epistemic_do_nothing: float
    epistemic_climb_object_1: float
    epistemic_climb_object_2: float
    mode: str


class PolicyPosteriorTwoObject(TypedDict):
    """Posterior over policies and EFE terms for three actions."""

    q_pi: np.ndarray
    terms: List[EfeTermsTwoObject]


class ActDecisionTwoObject(TypedDict):
    """Action selection decision for Phase 1.5."""

    action: int
    q_pi: np.ndarray
    efe_terms: List[EfeTermsTwoObject]
    prior_height: np.ndarray
    prior_aff_1: np.ndarray
    prior_aff_2: np.ndarray


class EpisodeResultTwoObject(TypedDict):
    """Per-episode outputs for two-object disambiguation."""

    logs: List[StepLogTwoObject]
    success: bool
    steps_to_success: int
    exploratory_actions: int
    exploitative_actions: int
    final_aff_1_belief: np.ndarray
    final_aff_2_belief: np.ndarray
    final_height_belief: np.ndarray
    b_kl_divergence: float


class ActiveInferenceDisambiguationAgent:
    """Agent that disambiguates which object affords climbing via EFE minimization.

    Hidden-state factors:
    - `q_height`: controllable latent state that determines `can_reach`
    - `q_aff_1`, `q_aff_2`: static latent affordance hypotheses per object

    This supports epistemic probing early in episodes and pragmatic exploitation
    after affordance uncertainty collapses.
    """

    def __init__(
        self,
        model: Dict[str, np.ndarray],
        policy_precision: float = 4.0,
        seed: int = 7,
        stochastic_action: bool = True,
        enable_parameter_learning: bool = False,
        dirichlet_prior_strength: float = 8.0,
        learn_action_ids: Sequence[int] = (CLIMB_OBJECT_1, CLIMB_OBJECT_2),
    ):
        self.model = model
        self.policy_precision = policy_precision
        self.stochastic_action = stochastic_action
        self.enable_parameter_learning = enable_parameter_learning
        self.learn_action_ids = tuple(learn_action_ids)
        self.rng = np.random.default_rng(seed)
        self._b_height_true = self.model["B_height"].copy()
        self._alpha_b_height: np.ndarray | None = None
        if self.enable_parameter_learning:
            self._alpha_b_height = init_dirichlet_from_b(
                self._b_height_true,
                prior_strength=dirichlet_prior_strength,
            )
            self.model["B_height"] = expected_b_from_alpha(self._alpha_b_height)

        self.q_height = self.model["D_height"].copy()
        self.q_aff_1 = self.model["D_aff_1"].copy()
        self.q_aff_2 = self.model["D_aff_2"].copy()

    def reset_beliefs(self, carry_affordance_belief: bool = True) -> None:
        """Reset beliefs at the beginning of an episode."""

        self.q_height = self.model["D_height"].copy()
        if not carry_affordance_belief:
            self.q_aff_1 = self.model["D_aff_1"].copy()
            self.q_aff_2 = self.model["D_aff_2"].copy()

    def _policy_posterior(self) -> PolicyPosteriorTwoObject:
        """Compute policy posterior from one-step expected free energy."""
        terms: List[EfeTermsTwoObject] = []
        for action in (DO_NOTHING, CLIMB_OBJECT_1, CLIMB_OBJECT_2):
            terms.append(
                expected_free_energy_terms_two_objects(
                    q_height=self.q_height,
                    q_aff_1=self.q_aff_1,
                    q_aff_2=self.q_aff_2,
                    action=action,
                    a=self.model["A"],
                    b_height=self.model["B_height"],
                    c=self.model["C"],
                )
            )

        g_values = np.array([terms[0]["G"], terms[1]["G"], terms[2]["G"]], dtype=float)
        q_pi = softmax(-g_values, precision=self.policy_precision)
        return {"q_pi": q_pi, "terms": terms}

    def act(self) -> ActDecisionTwoObject:
        """Select action from policy posterior."""

        out = self._policy_posterior()
        if self.stochastic_action:
            action = int(
                self.rng.choice([DO_NOTHING, CLIMB_OBJECT_1, CLIMB_OBJECT_2], p=out["q_pi"])
            )
        else:
            action = int(np.argmax(out["q_pi"]))

        prior_height = predict_height_two_objects(
            q_height=self.q_height,
            q_aff_1=self.q_aff_1,
            q_aff_2=self.q_aff_2,
            b_height=self.model["B_height"],
            action=action,
        )

        return {
            "action": action,
            "q_pi": out["q_pi"],
            "efe_terms": out["terms"],
            "prior_height": prior_height,
            "prior_aff_1": self.q_aff_1.copy(),
            "prior_aff_2": self.q_aff_2.copy(),
        }

    def update(
        self,
        obs_idx: int,
        action: int,
        prior_height: np.ndarray,
        prior_aff_1: np.ndarray,
        prior_aff_2: np.ndarray,
    ) -> None:
        """Update posterior beliefs based on selected action and observation."""

        post = infer_posterior_two_objects(
            obs_idx=obs_idx,
            action=action,
            prior_height=prior_height,
            prior_aff_1=prior_aff_1,
            prior_aff_2=prior_aff_2,
            a=self.model["A"],
            b_height=self.model["B_height"],
        )
        self.q_height = normalize(post["q_height"])
        self.q_aff_1 = normalize(post["q_aff_1"])
        self.q_aff_2 = normalize(post["q_aff_2"])

    def _mode_for_action(self, action: int) -> str:
        """Classify action mode for logging only (not used in control)."""

        if action == DO_NOTHING:
            return "passive"

        climb_diff = float(abs(self.q_aff_1[1] - self.q_aff_2[1]))
        best_action = CLIMB_OBJECT_1 if self.q_aff_1[1] >= self.q_aff_2[1] else CLIMB_OBJECT_2
        if climb_diff < 0.2:
            return "exploratory"
        if action == best_action:
            return "exploitative"
        return "exploratory"

    def _update_b_dirichlet(
        self,
        action: int,
        prior_height: np.ndarray,
        posterior_height: np.ndarray,
        prior_aff_1: np.ndarray,
        prior_aff_2: np.ndarray,
    ) -> None:
        """Update Dirichlet concentration over transition reliability."""

        if not self.enable_parameter_learning or self._alpha_b_height is None:
            return
        if action not in self.learn_action_ids:
            return

        transition_counts = np.outer(posterior_height, prior_height)
        for aff_1_idx, aff_1_weight in enumerate(prior_aff_1):
            for aff_2_idx, aff_2_weight in enumerate(prior_aff_2):
                self._alpha_b_height[action, aff_1_idx, aff_2_idx] += (
                    aff_1_weight * aff_2_weight * transition_counts
                )
        self.model["B_height"] = expected_b_from_alpha(self._alpha_b_height)

    def b_kl_divergence(self) -> float:
        """KL divergence between current and ground-truth transition tensors."""

        return kl_b_tensors(self._b_height_true, self.model["B_height"])

    def rollout_episode(
        self,
        world,
        max_steps: int = 12,
        carry_affordance_belief: bool = True,
    ) -> EpisodeResultTwoObject:
        """Run one full episode and return logs and summary statistics."""

        self.reset_beliefs(carry_affordance_belief=carry_affordance_belief)
        world.reset()

        logs: List[StepLogTwoObject] = []
        success_step = None
        exploratory_actions = 0
        exploitative_actions = 0

        for step_idx in range(max_steps):
            decision = self.act()
            action = int(decision["action"])

            next_obs, done, _ = world.step(action)
            obs_idx = int(next_obs["can_reach"])
            self.update(
                obs_idx=obs_idx,
                action=action,
                prior_height=decision["prior_height"],
                prior_aff_1=decision["prior_aff_1"],
                prior_aff_2=decision["prior_aff_2"],
            )
            self._update_b_dirichlet(
                action=action,
                prior_height=decision["prior_height"],
                posterior_height=self.q_height,
                prior_aff_1=decision["prior_aff_1"],
                prior_aff_2=decision["prior_aff_2"],
            )

            mode = self._mode_for_action(action)
            if mode == "exploratory":
                exploratory_actions += 1
            if mode == "exploitative":
                exploitative_actions += 1

            terms = decision["efe_terms"]
            logs.append(
                StepLogTwoObject(
                    step=step_idx,
                    action=PHASE1_5_ACTION_NAMES[action],
                    can_reach_obs="yes" if obs_idx == 1 else "no",
                    q_height_low=float(self.q_height[0]),
                    q_height_high=float(self.q_height[1]),
                    q_aff_1_unknown=float(self.q_aff_1[0]),
                    q_aff_1_climbable=float(self.q_aff_1[1]),
                    q_aff_1_not_climbable=float(self.q_aff_1[2]),
                    q_aff_2_unknown=float(self.q_aff_2[0]),
                    q_aff_2_climbable=float(self.q_aff_2[1]),
                    q_aff_2_not_climbable=float(self.q_aff_2[2]),
                    affordance_entropy_total=float(entropy(self.q_aff_1) + entropy(self.q_aff_2)),
                    q_pi_do_nothing=float(decision["q_pi"][0]),
                    q_pi_climb_object_1=float(decision["q_pi"][1]),
                    q_pi_climb_object_2=float(decision["q_pi"][2]),
                    g_do_nothing=float(terms[0]["G"]),
                    g_climb_object_1=float(terms[1]["G"]),
                    g_climb_object_2=float(terms[2]["G"]),
                    pragmatic_do_nothing=float(terms[0]["pragmatic"]),
                    pragmatic_climb_object_1=float(terms[1]["pragmatic"]),
                    pragmatic_climb_object_2=float(terms[2]["pragmatic"]),
                    epistemic_do_nothing=float(terms[0]["epistemic"]),
                    epistemic_climb_object_1=float(terms[1]["epistemic"]),
                    epistemic_climb_object_2=float(terms[2]["epistemic"]),
                    mode=mode,
                )
            )

            if done:
                success_step = step_idx + 1
                break

        return {
            "logs": logs,
            "success": success_step is not None,
            "steps_to_success": success_step if success_step is not None else max_steps,
            "exploratory_actions": exploratory_actions,
            "exploitative_actions": exploitative_actions,
            "final_aff_1_belief": self.q_aff_1.copy(),
            "final_aff_2_belief": self.q_aff_2.copy(),
            "final_height_belief": self.q_height.copy(),
            "b_kl_divergence": self.b_kl_divergence(),
        }
