import optuna

from cleanrl_utils.tuner import Tuner

env_list = [
"Acrobot-v1",
"CartPole-v1",
"Pendulum-v1",
"MountainCar-v0",
]

for env in env_list:
    tuner = Tuner(
        script="cleanrl/ppo.py",
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
            "anneal-lr": trial.suggest_categorical("annealing-lr", [True, False]),
            "gae": trial.suggest_categorical("gae", [True, False]),
            "gamma": trial.suggest_uniform("gamma", 0.1, 0.999),\
            "gae-lambda": trial.suggest_uniform("gae-lambda", 0.1, 0.999),
            "update-epochs": trial.suggest_int("update-epochs", 1, 1024),
            "norm-adv": trial.suggest_categorical("norm-adv", [True, False]),
            "clip-coef": trial.suggest_uniform("clip-coef", 0.1, 0.999),
            "clip-vloss":  True,
            "quantize": True,
        },
        pruner=optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(),
        start_trial={
            "learning-rate": 2.5e-4,
            "num-step": 128,
        }
    )
    tuner.tune(
        num_trials=1,
        num_seeds=1,
    )
