"""Minimal active-inference agent in PyMDP style for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, TypedDict

import numpy as np

from agent.generative_model import expected_b_from_alpha, init_dirichlet_from_b, kl_b_tensors
from agent.inference_utils import (
    EfeTerms,
    expected_free_energy_terms,
    infer_posterior,
    normalize,
    predict_height,
    softmax,
)


@dataclass
class StepLog:
    """Per-step log of beliefs, policies, and expected free energy terms."""

    step: int
    action: str
    can_reach_obs: str
    q_height_low: float
    q_height_high: float
    q_aff_unknown: float
    q_aff_climbable: float
    q_aff_not_climbable: float
    q_pi_do_nothing: float
    q_pi_climb: float
    g_do_nothing: float
    g_climb: float
    pragmatic_do_nothing: float
    pragmatic_climb: float
    epistemic_do_nothing: float
    epistemic_climb: float


class PolicyPosterior(TypedDict):
    """Posterior over policies and their expected free energy terms."""

    q_pi: np.ndarray
    terms: List[EfeTerms]


class ActDecision(TypedDict):
    """Action selection decision with cached posteriors and priors."""

    action: int
    q_pi: np.ndarray
    efe_terms: List[EfeTerms]
    prior_height: np.ndarray
    prior_aff: np.ndarray


class EpisodeResult(TypedDict):
    """Summary outputs from a single episode rollout."""

    logs: List[StepLog]
    success: bool
    steps_to_success: int
    final_aff_belief: np.ndarray
    final_height_belief: np.ndarray
    b_kl_divergence: float
    b_target_distance: float | None


class ActiveInferenceAffordanceAgent:
    """Agent with explicit A/B/C/D and one-step EFE policy inference."""

    def __init__(
        self,
        model: Dict[str, np.ndarray],
        policy_precision: float = 5.0,
        enable_parameter_learning: bool = False,
        dirichlet_prior_strength: float = 8.0,
        learn_action_ids: Sequence[int] = (1,),
    ):
        self.model = model
        self.policy_precision = policy_precision
        self.enable_parameter_learning = enable_parameter_learning
        self.learn_action_ids = tuple(learn_action_ids)
        self._b_height_true = self.model["B_height"].copy()
        self._alpha_b_height: np.ndarray | None = None
        if self.enable_parameter_learning:
            self._alpha_b_height = init_dirichlet_from_b(
                self._b_height_true,
                prior_strength=dirichlet_prior_strength,
            )
            self.model["B_height"] = expected_b_from_alpha(self._alpha_b_height)

        self.q_height = self.model["D_height"].copy()
        self.q_aff = self.model["D_aff"].copy()
        self.reset_beliefs()

    def reset_beliefs(self, carry_affordance_belief: bool = True) -> None:
        """Reset posterior beliefs at the start of an episode."""
        self.q_height = self.model["D_height"].copy()
        if not carry_affordance_belief:
            self.q_aff = self.model["D_aff"].copy()

    def _policy_posterior(self) -> PolicyPosterior:
        """Compute the posterior over policies and associated EFE terms."""
        terms: List[EfeTerms] = []
        for action in (0, 1):
            t = expected_free_energy_terms(
                q_height=self.q_height,
                q_aff=self.q_aff,
                action=action,
                a=self.model["A"],
                b_height=self.model["B_height"],
                c=self.model["C"],
            )
            terms.append(t)

        g_values = np.array([terms[0]["G"], terms[1]["G"]], dtype=float)
        q_pi = softmax(-g_values, precision=self.policy_precision)

        return {"q_pi": q_pi, "terms": terms}

    def act(self) -> ActDecision:
        """Select an action from the current policy posterior."""
        out = self._policy_posterior()
        action = int(np.argmax(out["q_pi"]))

        prior_height = predict_height(
            q_height=self.q_height,
            q_aff=self.q_aff,
            b_height=self.model["B_height"],
            action=action,
        )

        return {
            "action": action,
            "q_pi": out["q_pi"],
            "efe_terms": out["terms"],
            "prior_height": prior_height,
            "prior_aff": self.q_aff.copy(),
        }

    def update(
        self,
        obs_idx: int,
        action: int,
        prior_height: np.ndarray,
        prior_aff: np.ndarray,
    ) -> None:
        """Update posterior beliefs based on observation and action."""
        post = infer_posterior(
            obs_idx=obs_idx,
            action=action,
            prior_height=prior_height,
            prior_aff=prior_aff,
            a=self.model["A"],
            b_height=self.model["B_height"],
        )
        self.q_height = normalize(post["q_height"])
        self.q_aff = normalize(post["q_aff"])

    def _update_b_dirichlet(
        self,
        action: int,
        prior_height: np.ndarray,
        posterior_height: np.ndarray,
        prior_aff: np.ndarray,
    ) -> None:
        """Level 1 parameter learning for transition reliability."""

        if not self.enable_parameter_learning or self._alpha_b_height is None:
            return
        if action not in self.learn_action_ids:
            return

        transition_counts = np.outer(posterior_height, prior_height)
        for aff_idx, aff_weight in enumerate(prior_aff):
            self._alpha_b_height[action, aff_idx] += aff_weight * transition_counts
        self.model["B_height"] = expected_b_from_alpha(self._alpha_b_height)

    def b_kl_divergence(self) -> float:
        """KL divergence between current and ground-truth transition tensors."""

        return kl_b_tensors(self._b_height_true, self.model["B_height"])

    def rollout_episode(
        self,
        world,
        max_steps: int = 10,
        carry_affordance_belief: bool = True,
    ) -> EpisodeResult:
        """Run a full episode rollout and collect per-step logs."""
        self.reset_beliefs(carry_affordance_belief=carry_affordance_belief)
        world.reset()

        logs: List[StepLog] = []
        success_step = None

        for step_idx in range(max_steps):
            decision = self.act()
            action = int(decision["action"])

            next_obs, done, _ = world.step(action)
            obs_idx = int(next_obs["can_reach"])
            self.update(
                obs_idx=obs_idx,
                action=action,
                prior_height=decision["prior_height"],
                prior_aff=decision["prior_aff"],
            )
            self._update_b_dirichlet(
                action=action,
                prior_height=decision["prior_height"],
                posterior_height=self.q_height,
                prior_aff=decision["prior_aff"],
            )

            terms = decision["efe_terms"]
            log = StepLog(
                step=step_idx,
                action="climb" if action == 1 else "do_nothing",
                can_reach_obs="yes" if obs_idx == 1 else "no",
                q_height_low=float(self.q_height[0]),
                q_height_high=float(self.q_height[1]),
                q_aff_unknown=float(self.q_aff[0]),
                q_aff_climbable=float(self.q_aff[1]),
                q_aff_not_climbable=float(self.q_aff[2]),
                q_pi_do_nothing=float(decision["q_pi"][0]),
                q_pi_climb=float(decision["q_pi"][1]),
                g_do_nothing=float(terms[0]["G"]),
                g_climb=float(terms[1]["G"]),
                pragmatic_do_nothing=float(terms[0]["pragmatic"]),
                pragmatic_climb=float(terms[1]["pragmatic"]),
                epistemic_do_nothing=float(terms[0]["epistemic"]),
                epistemic_climb=float(terms[1]["epistemic"]),
            )
            logs.append(log)
            if done:
                success_step = step_idx + 1
                break

        return {
            "logs": logs,
            "success": success_step is not None,
            "steps_to_success": success_step if success_step is not None else max_steps,
            "final_aff_belief": self.q_aff.copy(),
            "final_height_belief": self.q_height.copy(),
            "b_kl_divergence": self.b_kl_divergence(),
            "b_target_distance": None,
        }
