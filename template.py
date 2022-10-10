# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import logging
import os
import random
import time
from distutils.util import strtobool
import torch

from cleanrl.algos.opt import hAdam
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HumanoidBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    
    # Quantization specific arguments
    ## Quantize Weight
    parser.add_argument("--quantize-weight", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-weight-bitwidth", type=int, default=8)
    ## Quantize Activation
    parser.add_argument("--quantize-activation" , type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--quantize-activation-bitwidth", type=int, default=8)
    parser.add_argument("--quantize-activation-quantize-min", type=int, default= 0)
    parser.add_argument("--quantize-activation-quantize-max", type=int, default= 255)
    parser.add_argument("--quantize-activation-quantize-reduce-range", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--quantize-activation-quantize-dtype", type=str, default="quint8")
    
    ## Other papers algorithm and ideas
    parser.add_argument("--use-num-adam", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    ## Other papers algorithm and ideas
    parser.add_argument("--optimizer" , type=str, default="Adam")
    
 
    args = parser.parse_args()
    # fmt: on
    return args

def get_quantization_config(self):
    if self.quantize_weight:
        if self.quantize_activation:
            return torch.ao.quantization.QConfig(
                activation = torch.ao.quantization.FakeQuantize.with_args(
                    observer = torch.ao.quantization.MovingAverageMinMaxObserver(
                        dtype = self.quantize_activation_quantize_dtype,
                        reduce_range = self.quantize_activation_quantize_reduce_range,
                        quant_min = self.quantize_activation_quantize_min,
                        quant_max = self.quantize_activation_quantize_max,
                    )
                ),
                weight = torch.ao.quantization.FakeQuantize.with_args(
                    observer = torch.ao.quantization.MovingAverageMinMaxObserver(
                        dtype = torch.quint8,
                        quant_min = -128,
                        quant_max = 127,
                    )
                )
            )
        else:
            return torch.ao.quantization.QConfig(
                activation = torch.nn.Identity,
                weight = torch.ao.quantization.FakeQuantize.with_args(
                    observer = torch.ao.quantization.MovingAverageMinMaxObserver,
                        quant_min = -128 ,
                        quant_max = 127,
                        dtype = torch.qint8 , 
                    )
            )
def add_datatypes(args):
    if args.quantize_activation_quantize_dtype is not None:
        if args.quantize_activation_quantize_dtype == "torch.quint8":
            args.quantize_activation_quantize_dtype = torch.quint8
        elif args.quantize_activation_quantize_dtype == "torch.qint8":
            args.quantize_activation_quantize_dtype = torch.qint8
        else:
            raise ValueError(f"{args.quantize_activation_quantize_dtype} is not supported for quantization")
    return args

def select_optmizer(args):
    if args.optimizer == "ADAM":
        return torch.optim.Adam
    elif args.optimizer == "hAdam":
        return hAdam
    elif args.optimizer == "Adan":
        return Adan