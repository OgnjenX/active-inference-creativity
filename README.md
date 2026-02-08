# Active Inference Creativity Sandbox

A research sandbox for studying how affordance learning, transfer, and later creativity-like behavior can emerge from **active inference**.

## Phases vs Levels (Important)

This repository distinguishes between two orthogonal dimensions:

- **Phases** describe the *task and environment structure* the agent is placed in.
- **Levels** describe the *learning capability of the agent* given that task.

Any Phase can be run at any Level.

|            | Level 0: State Inference | Level 1: Parameter Learning |
|------------|--------------------------|-----------------------------|
| Phase 1    | ✅ supported              | ✅ supported                 |
| Phase 1.5  | ✅ supported              | ✅ supported                 |

Phase 2 introduces **Level 2: structure learning** (model comparison) in
parallel scripts without changing Phase 1/1.5 behavior.


## Phase 1 Goal

Phase 1 focuses on a minimal, interpretable setup where an agent:

- infers whether an object affords climbing from action-outcome observations,
- transfers that inferred affordance to a new object instance,
- selects actions using expected free energy (not rewards).

## Phase 1.5 Goal

Phase 1.5 extends the setup to two competing affordance hypotheses:

- two objects are present and exactly one is climbable,
- the agent must disambiguate which object affords climbing via epistemic action,
- behavior should transition from uncertainty reduction to reliable exploitation.

## Level 1: Parameter Learning

Level 1 adds learning over transition reliability while keeping the affordance
concept fixed.

- learned: Dirichlet parameters over affordance-conditioned `B_height` dynamics
- not learned: latent-state definitions, action space, observation model `A`, preferences `C`
- interpretation: the agent learns how strongly climbability changes height, not
  whether climbability exists as a concept

## Level 2: Structure Learning (Concept Learning)

Level 2 adds model-family comparison instead of tuning a fixed model:

- **Family A (flat / object-specific)**: predicts outcomes per object identity with no shared latent factor.
- **Family B (factorized / conceptual)**: introduces a shared latent factor that explains outcomes across objects.

The agent runs inference under both families, accumulates variational free energy
(approximate model evidence), and updates posterior preference over model
families over time.

Important scope note:
- this phase supports **concept selection via structure learning** among
  predefined model families; it does not implement unconstrained concept invention.

## Why This Is Active Inference (Not RL)

This project does **not** use rewards, Q-values, policy gradients, or value functions.

Instead, the agent uses an explicit generative model:

- hidden states: `agent_height_state` and `object_affordance`
- observation model: `A`
- transition model: `B`
- preferences: `C` (preferred observations)
- priors: `D`

Actions are chosen by minimizing expected free energy, which combines:

- pragmatic value: expected alignment with preferred observations (`can_reach=yes`)
- epistemic value: expected information gain about hidden causes (object affordance)

## Modeling Assumptions

- affordances are static latent causes in each episode/environment instance
- the only action-controllable latent state is `agent_height_state`
- observations remain outcome-only (`can_reach`), so object identity is never observed directly
- Phase 1.5 worlds are instantiated with exactly one climbable object to force hypothesis disambiguation
- identity-swap transfer is a stress test: prior beliefs may initially conflict with swapped dynamics

## Environment Design

The environment is deterministic and object-based, with strict separation of concerns:

- environment internal state includes object height and agent height
- agent never sees heights or affordance flags
- exposed observation is only `can_reach in {yes, no}`

Phase 2 keeps observations outcome-only and intervention-based actions while
removing explicit affordance labels from both environment and agent.

## Project Layout

- `/env`: world dynamics and observation API
- `/agent`: generative model + active inference loop
- `/experiments`: training, transfer, and metrics
- `/notebooks`: debug notebook scaffold

## Install

```bash
pip install -r requirements.txt
```

## Run Experiment 1 (Training)

```bash
python -m experiments.phase1_train
```

Run with Level 1 parameter learning:

```bash
python -c "from experiments.phase1_train import run_training; run_training(enable_parameter_learning=True)"
```

Expected behavior:

- early uncertainty over object affordance
- increasing belief in `climbable`
- reliable `climb` action selection

## Run Experiment 2 (Transfer)

```bash
python -m experiments.phase1_transfer
```

Expected behavior:

- transferred agent starts with stronger climbable belief
- fewer exploratory steps compared with a fresh baseline agent
- successful reaching in the new object instance

## Run Phase 1.5 Experiment 1 (Disambiguation Training)

```bash
python -m experiments.phase1_5_train
```

Run with Level 1 parameter learning:

```bash
python -c "from experiments.phase1_5_train import run_phase1_5_training; run_phase1_5_training(enable_parameter_learning=True)"
```

Expected behavior:

- early probing of both climb actions while affordance beliefs are ambiguous,
- decreasing affordance entropy over episodes,
- convergence to the truly climbable object.

## Run Phase 1.5 Experiment 2 (Transfer)

```bash
python -m experiments.phase1_5_transfer
```

Expected behavior:

- transferred beliefs disambiguate object affordances faster than a fresh baseline,
- clear epistemic-to-pragmatic transition in action patterns,
- consistent successful reach once uncertainty is reduced.

To test identity swap explicitly:

```bash
python -c "from experiments.phase1_5_transfer import run_phase1_5_transfer; run_phase1_5_transfer(swap_identities=True)"
```

In swap mode, slower adaptation than non-swap transfer is expected because carried priors can start anti-aligned with the new world.

## Run Phase 2 Stage A Demo (Environment Only)

```bash
python3 -m experiments.phase2_env_demo
```

This validates JSON-loaded worlds with hidden primitive properties
(`height_delta`, `stability`, `stackability`) and outcome generation.

## Run Phase 2 Stage B (Level 2 Model Comparison Training)

```bash
python3 -m experiments.phase2_train_structure
```

Expected diagnostics:

- cumulative model evidence (`free energy`) curves for flat vs factorized families,
- normalized evidence (`mean free energy per step`) to account for variable episode length,
- explicit complexity (`Occam`) term per family and combined score = free energy + complexity,
- posterior preference over model families across episodes,
- selected family controlling EFE-based action choice.

Current Level 2 control detail:
- action selection uses hard model selection (MAP family) at each step;
  soft uncertainty-weighted model averaging is left for later extension.

## Run Phase 2 Stage B Transfer

```bash
python3 -m experiments.phase2_transfer
```

Expected behavior:

- transferred structure state yields fewer steps-to-success than a fresh baseline,
- model-family posterior adapts when moving to a new world instance.

## Run Phase 2 Model-Selection Sanity Check

```bash
python3 -m experiments.phase2_model_selection_sanity
```

This compares model preference across:

- structured world with reusable latent-factor structure (factorized should win),
- unstructured world with independent channel generation (flat should win).

## Outputs

Scripts print per-episode summaries and save belief plots to:

- `experiments/results/phase1_train_beliefs.png`
- `experiments/results/phase1_transfer_transferred_beliefs.png`
- `experiments/results/phase1_transfer_fresh_beliefs.png`
- `experiments/results/phase1_5_train_beliefs.png`
- `experiments/results/phase1_5_train_entropy.png`
- `experiments/results/phase1_5_transfer_transferred_beliefs.png`
- `experiments/results/phase1_5_transfer_fresh_beliefs.png`
- `experiments/results/phase1_5_transfer_transferred_entropy.png`
- `experiments/results/phase1_5_transfer_fresh_entropy.png`
- `experiments/results/phase1_train_level1_b_kl.png`
- `experiments/results/phase1_5_train_level1_b_kl.png`
- `experiments/results/phase2_train_model_evidence.png`
- `experiments/results/phase2_train_model_evidence_per_step.png`
- `experiments/results/phase2_train_model_posterior.png`
- `experiments/results/phase2_transfer_steps.png`
- `experiments/results/phase2_transfer_transferred_posterior.png`
- `experiments/results/phase2_transfer_fresh_posterior.png`
- `experiments/results/phase2_structured_model_posterior.png`
- `experiments/results/phase2_unstructured_model_posterior.png`

## Not in Phase 1

The following are intentionally deferred to later phases:

- stacking and compositional behavior
- raw perception pipelines
- relational graph reasoning
- performance optimization
