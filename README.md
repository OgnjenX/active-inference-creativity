# Active Inference Creativity Sandbox

A research sandbox for studying how affordance learning, transfer, and later creativity-like behavior can emerge from **active inference**.

## Phase 1 Goal

Phase 1 focuses on a minimal, interpretable setup where an agent:

- infers whether an object affords climbing from action-outcome observations,
- transfers that inferred affordance to a new object instance,
- selects actions using expected free energy (not rewards).

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

## Environment Design

The environment is deterministic and object-based, with strict separation of concerns:

- environment internal state includes object height and agent height
- agent never sees heights or affordance flags
- exposed observation is only `can_reach in {yes, no}`

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

## Outputs

Scripts print per-episode summaries and save belief plots to:

- `experiments/results/phase1_train_beliefs.png`
- `experiments/results/phase1_transfer_transferred_beliefs.png`
- `experiments/results/phase1_transfer_fresh_beliefs.png`

## Not in Phase 1

The following are intentionally deferred to later phases:

- stacking and compositional behavior
- raw perception pipelines
- relational graph reasoning
- performance optimization
