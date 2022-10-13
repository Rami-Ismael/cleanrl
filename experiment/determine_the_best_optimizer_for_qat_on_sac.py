from random import random
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../cleanrl')

from cleanrl.functional.only_weight_bias.functional_sac_continuous_action import sac_functional

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
    
    max_episode_return = -1000
    for optimizer in ["Adam","hAdam","Adan"]:
        policy_lr = trial.suggest_float("policy_lr", 2.5e-6, 1e-2)
        q_lr = trial.suggest_float("q_lr", 2.5e-6, 1e-2)
        seed = random() * 100 + 1
        for seed in range( 1 , 3):
            average_episode_return    = sac_functional(
                
                seed = int(seed),
                
                total_timesteps = 100_000,
                policy_lr = policy_lr,
                q_lr = q_lr,
                track = True ,  
                optimizer = optimizer, 
            )
            max_episode_return = max(max_episode_return , average_episode_return)
    return  max_episode_return
    
# have the seed to be random value between 0 and 10

study = optuna.create_study(
    direction="maximize",
    pruner = optuna.pruners.MedianPruner(n_startup_trials=0 , 
                                         n_warmup_steps = 5),
    sampler = optuna.samplers.TPESampler(),
)
'''
start_trial = {
    "policy_lr": 3e-4,
}
'''
#study.enqueue_trial(start_trial)

study.optimize(
    objective , 
    n_trials= 100,
)