import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../cleanrl')

from cleanrl.functional.functional_ddpg_continuous_action import ddpg_functional

import os
import runpy
import sys
import time
from typing import Callable, Dict, List, Optional
import logging
from colorlog import log
import numpy as np
import optuna
import wandb
from rich import print
from tensorboard.backend.event_processing import event_accumulator

logging.basicConfig(filename="tests.log", level=logging.NOTSET,
                    format='%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s')

def objective(trial):
    print(f" Strarting trial {trial.number}")
    
    learning_rate = trial.suggest_uniform("learning_rate", 2.5e-6, 0.1)
    
    run , average_episode_return =  ddpg_functional(
        learning_rate = learning_rate , 
        total_time_steps = 100 , 
        track = True , 
        trial = trial
    )
    if run:
        run.finish()
    return average_episode_return
    


study = optuna.create_study(
    direction="maximize",
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler = optuna.samplers.TPESampler(),
)
start_trial = {
    "learning"
}

study.enqueue_trial(start_trial)

study.optimize(
    objective , 
    n_trials= 2,
)