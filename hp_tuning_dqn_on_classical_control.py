import optuna

from cleanrl_utils.tuner import Tuner

env_list = [
"Acrobot-v1",
"CartPole-v1",
"Pendulum"
"MountainCar-v0",
]

for env in env_list:
    tuner = Tuner(
        script="cleanrl/dqn.py",
        metric="charts/episodic_return",
        metric_last_n_average_window=50,
        direction="maximize",
        aggregation_type="average",
        target_scores={
            env: None
        },
        params_fn=lambda trial: {
            "learning-rate": trial.suggest_loguniform("learning-rate", 0.0003, 0.003),
            "total-timesteps": 100000,
            "torch-deterministic": True,
            "buffer-size": trial.suggest_int("buffer-size", 10000, 100000),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
            "batch-size": trial.suggest_int("batch-size", 128, 256),
            "quantize": trial.suggest_categorical("quantized", [True, False]),
            "target-network-frequency": trial.suggest_int("target-network-frequency", 100 , 1000),
            "start-e": 1,
            "end-e": 0.05,
            "cuda": True
        },
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(),
    )
    tuner.tune(
        num_trials=100,
        num_seeds=3,
    )
