import optuna
import os
import runpy
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import optuna
import wandb
from rich import print
import numpy as np

## Goals it to have less head ache for developer
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MyTuner:
    def __init__(
        self,
        script: str,
        metric: str,
        target_scores: Dict[str, Optional[List[float]]],
        params_fn: Callable[[optuna.Trial], Dict],
        direction: str = "maximize",
        aggregation_type: str = "average",
        metric_last_n_average_window: int = 50,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        storage: str = "sqlite:///cleanrl_hpopt.db",
        study_name: str = "",
        wandb_kwargs: Dict[str, any] = {},
    ) -> None:
        self.script = script
        self.metric = metric
        self.target_scores = target_scores
        if len(self.target_scores) > 1:
            if None in self.target_scores.values():
                raise ValueError(
                    "If there are multiple environments, the target scores must be specified for each environment."
                )

        self.params_fn = params_fn
        self.direction = direction
        self.aggregation_type = aggregation_type
        if self.aggregation_type == "average":
            self.aggregation_fn = np.average
        elif self.aggregation_type == "median":
            self.aggregation_fn = np.median
        elif self.aggregation_type == "max":
            self.aggregation_fn = np.max
        elif self.aggregation_type == "min":
            self.aggregation_fn = np.min
        else:
            raise ValueError(f"Unknown aggregation type {self.aggregation_type}")
        self.metric_last_n_average_window = metric_last_n_average_window
        self.pruner = pruner
        self.sampler = sampler
        self.storage = storage
        self.study_name = study_name
        if len(self.study_name) == 0:
            self.study_name = f"tuner_{int(time.time())}"
        self.wandb_kwargs = wandb_kwargs
    def tune(
        self,
        num_trial:int,
        num_seeds:int
    ) -> None:
        params = self.params_fn(trial)
        run = None
        
        if len(self.wandb_kwargs.keys()) > 0:
            run = wandb.init(
                **self.wandb_kwargs,
                config=params,
                name=f"{self.study_name}_{trial.number}",
                group=self.study_name,
                save_code=True,
                reinit=True,
            )
        ## the script
        algo_command = [f"--{key}={value}" for key, value in params.items()]
        sys.argv = algo_command + [f"--env-id={"CartPolve-v1"}", f"--seed={42}", "--track=False"]
        
        with HiddenPrints():
            runpy.run_path(self.script, run_name="__main__")
        
        ## read metric from tensorboard
        
        ea = event_accumulator.EventAccumulator(f"./runs/{run.name}")
            