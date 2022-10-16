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

for quantize_activation_bitwidth , quantize_weight_bitwidth in zip( [  8 , 6 ,  4, 2, 1] , [  8 , 6 , 4, 2, 1] ):
    for seed in range( 0 , 2):
        average_episode_return    = sac_functional(
            seed = int(seed),
            total_timesteps = 200_000,
            quantize_activation_bitwidth = quantize_activation_bitwidth,
            quantize_weight_bitwidth =  quantize_weight_bitwidth,
            track = True,  
            optimizer = "Adam"
        )