import optuna

from cleanrl_utils.tuner import Tuner

env_list = [
"HalfCheetahBulletEnv-v0",
"MinitaurBulletDuckEnv-v0",
"HumanoidBulletEnv-v0",
"ReacherBulletEnv-v0"
]

for env in env_list:
    tuner = Tuner(
        script="cleanrl/ppo_continuous_action.py",
        metric="charts/episodic_return",
        metric_last_n_average_window=50,
        direction="maximize",
        aggregation_type="average",
        storage = "sqlite:///cleanrl_hp_ppo_pybullet.db",
        target_scores={
            env: None
        },
        params_fn=lambda trial: {
            "cuda": True,
            "total-timesteps": 1000000,
            "learning-rate": trial.suggest_uniform("learning-rate", 2.5e-5, 0.1),
            "quantize": True,
        },
        pruner=optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(),
        start_trial={
            "learning-rate": 3e-4,
            "total-timesteps": 5,
        },
        wandb_kwargs={"project": "cleanrl", "tags": ["ppo", "pybullet"]},
    )
    tuner.tune(
        num_trials=10,
        num_seeds=3,
    )

