import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../cleanrl')

from cleanrl.functional.functional_sac_continuous_action import sac_functional

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
    
    learning_starts = trial.suggest_uniform("learning_starts", 2.5e-6, 0.1)
    policy_lr = trial.suggest_uniform("policy_lr", 2.5e-6, 0.1)
    q_network_lr = trial.suggest_uniform("q_network_lr", 2.5e-6, 0.1)
    
    average_episode_return = sac_functional(
        learning_starts = learning_starts,
        policy_lr = policy_lr,
        q_lr = q_network_lr,
        track = True , 
        trial = trial
    )
    return average_episode_return
    


study = optuna.create_study(
    direction="maximize",
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler = optuna.samplers.TPESampler(),
)
start_trial = {
    "learning-starts": 5e3,
    "policy_lr": 3e-4,
    "q_network_lr": 1e-3,
}

study.enqueue_trial(start_trial)

study.optimize(
    objective , 
    n_trials= 2,
)