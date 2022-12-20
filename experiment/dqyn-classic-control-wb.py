from random import random
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../cleanrl')
from cleanrl.functional.only_weight_bias.functional_dqn import dqn_functional


'''
This code gaols it to train a DQN agent on the 
dicrete classic control environments and upload to hugging Face
'''

#dicrete_box_envs = [ "CartPole-v1" , "MountainCar-v0" , "Acrobot-v1" ]
dicrete_box_envs = [ "CartPole-v1" ]

#optimizers = [ "Adam" , "Adan" , "hAdam" ]
optimizers = ["Adam"]


for env_id in dicrete_box_envs:
    for optimizer in optimizers:
            output = dqn_functional(
                seed = 0,
                env_id=env_id,
                track = True,
                optimizer = optimizer , 
                quantize_weight = False , 
                quantize_activation = False,
                wandb_entity = "compress_rl" , 
                wandb_project="cleanrl",
                trial = None
            )