"""Frozen RL v0.01: semantic replay and placement-based DQN.

Import components from their modules; the existing smoke API stays independent.

Requires the preinstalled PyTorch runtime (CPU or CUDA). Example, with the
experiment budget and watchdog supplied explicitly by the caller::

    config = TrainingConfig(decision_budget=budget, algorithm='double')
    trainer = DQNTrainer(config)
    summary = train(env, trainer, episode_decision_limit=watchdog)

TrainingConfig lives in config, DQNTrainer in dqn, and train in training.
RLRunner accepts any select(state, geometric_mask) policy; FormalRandomPolicy
and DQNTrainer share that interface. Pass on_transition=replay.add to record a
Formal Random baseline. Use reward_mode='shaped' for exact PBRS.

Counters: decisions count selections; stored_experiences excludes system
failures; learning starts at stored experience 5,000; each accepted decision
permits at most one optimizer step; target hard sync counts optimizer steps.
Policy epsilon uses the stored-experience counter and the explicit budget.
Initial target copy is excluded from target_sync_count. Training stops on a
diagnostic. Default masks remain geometric-only. The explicit oracle condition
uses physical masks for acting and one-step DDQN bootstrap, with no additional
state features. See oracle.py and --reachability-mask oracle.
"""
