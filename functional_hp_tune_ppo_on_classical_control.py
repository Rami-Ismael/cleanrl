import optuna

from cleanrl_utils.tuner_functional import Tuner

env_list = [
"CartPole-v1",
"Acrobot-v1",
"Pendulum-v1",
"MountainCar-v0",
]

for env in env_list:
    tuner = Tuner(
        metric="charts/episodic_return",
        metric_last_n_average_window=50,
        direction="maximize",
        aggregation_type="average",
        storage = "sqlite:///cleanrl_hp_ppo_classic_control.db",
        target_scores={
            env: None
        },
        params_fn=lambda trial: {
            "learning_rate": trial.suggest_uniform("learning_rate", 2.5e-5, 0.1),
        },
        pruner=optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0),
        sampler=optuna.samplers.TPESampler(),
        start_trial={
            "learning-rate": 2.5e-4,
            "ent-coef": 0.01,
            "vf-coef": 0.5,
        },
        wandb_kwargs={"project": "cleanrl", "tags": ["ppo", "classic-controll"]},
        efficnet_timestep_algo = "binary_growth",
        max_total_timesteps = 500000,
        
    )
    tuner.tune(
        num_trials=100,
        num_seeds=1,
    )

