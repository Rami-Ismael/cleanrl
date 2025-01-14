import optuna

from cleanrl_utils.tuner import Tuner

env_list = [
"BreakoutNoFrameskip-v4"
]

for env in env_list:
    tuner = Tuner(
        script="cleanrl/ppo_atari.py",
        metric="charts/episodic_return",
        metric_last_n_average_window=50,
        direction="maximize",
        aggregation_type="average",
        storage = "sqlite:///cleanrl_hp_ppo_classic_control.db",
        target_scores={
            env: None
        },
        params_fn=lambda trial: {
            "cuda": True,
            "total-timesteps": 500000,
            "learning-rate": trial.suggest_uniform("learning-rate", 2.5e-5, 0.1),
            "num-step": trial.suggest_categorical("num-step", [16 , 1024]),
            "anneal-lr": trial.suggest_categorical("annealing-lr", [True, False]),
            "gae": trial.suggest_categorical("gae", [True, False]),
            "gamma": trial.suggest_uniform("gamma", 0.1, 0.999),\
            "gae-lambda": trial.suggest_uniform("gae-lambda", 0.1, 0.999),
            "num-minibatches": trial.suggest_int("num-mini-batch", 1, 1024),
            "update-epochs": trial.suggest_int("update-epochs", 1, 1024),
            "norm-adv": trial.suggest_categorical("norm-adv", [True, False]),
        },
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(),
        start_trial={
            "learning-rate": 2.5e-4,
            
        }
    )
    tuner.tune(
        num_trials=1,
        num_seeds=1,
    )


