import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../cleanrl')

from cleanrl.functional.functional_ppo_classic_control import ppo_functional

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
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    
    run , average_episode_return = ppo_functional(
        
        track = True,
        learning_rate = learning_rate , 
        trial = trial
        
    )
    if run:
        run.finish()
    return average_episode_return
    


study = optuna.create_study(
    direction="maximize",
    pruner = optuna.pruners.MedianPruner(n_startup_trials=0 , 
                                         n_warmup_steps = 5),
    sampler = optuna.samplers.TPESampler(),
)
start_trial = {
    "learning_rate": .5,
}

study.enqueue_trial(start_trial)

study.optimize(
    objective , 
    n_trials= 1,
)