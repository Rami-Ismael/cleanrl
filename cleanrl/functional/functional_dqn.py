# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
from cmath import log
import os
import random
import time
from distutils.util import strtobool
import logging


import gym
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from cleanrl.algos.opt import Adan, hAdam
from cleanrl.argument_utils import get_datatype

from template import get_quantization_config

logging.basicConfig(filename="tests.log", level=logging.NOTSET,
                    filemode='w',
                    format='%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s')
try:
    from quantize_methods import size_of_model
except ModuleNotFoundError as e:
    logging.error(e)

def size_of_model(model):
    name_file = "temp.pt"
    torch.save(model.state_dict(), name_file)
    size =  os.path.getsize(name_file)/1e6
    os.remove(name_file)
    return size

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
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    
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
    parser.add_argument("--optimizer" , type=str, default="Adam")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env , 
                 quantize_weight:bool = False,
                 quantize_weight_bitwidth:int = 8,
                 quantize_activation:bool = False,
                 quantize_activation_bitwidth:int = 8,
                 quantize_activation_quantize_min:int = 0,
                 quantize_activation_quantize_max:int = 255,
                 quanitize_activation_quantize_reduce_range:bool = False,
                 quantize_activation_quantize_dtype:torch.dtype = torch.quint8 , 
                 ):
        super().__init__()
        ## Save the Param
        ## Quantize Param
        ### Quantuze Param Weight
        self.quantize_weight = quantize_weight
        self.quantize_weight_bitwidth = quantize_weight_bitwidth
        ### Quantize Param Activation
        self.quantize_activation = quantize_activation
        self.quantize_activation_bitwidth = quantize_activation_bitwidth
        self.quantize_activation_quantize_min = quantize_activation_quantize_min
        self.quantize_activation_quantize_max = quantize_activation_quantize_max
        self.quanitize_activation_quantize_reduce_range = quanitize_activation_quantize_reduce_range
        self.quantize_activation_quantize_dtype = quantize_activation_quantize_dtype
        
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )
        logging.info(f"The model is {self.network} GB")  
        logging.info(f"The size of the model is {size_of_model(self.network)}")
        
        if self.quantize_weight or self.quantize_activation:
            self.network = nn.Sequential(
                torch.ao.quantization.QuantStub(),
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, env.single_action_space.n),
                torch.ao.quantization.DeQuantStub(),
            )
            ## Fuse your model
            self.fuse_model()
            logging.info("Fused model")
            logging.info(self.network)
            ## set the qconfig for the model
            self.network.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
            logging.info(f"Set qconfig { self.network.qconfig} for the model")
            ## Prepare the model for quantize aware trianing
            torch.ao.quantization.prepare_qat(self.network, inplace=True)
            logging.info("Prepared model for quantization aware training")
            logging.info(self.network)

    def forward(self, x):
        return self.network(x)
    ## Fuse the model
    def fuse_model(self):
        layers = list()
        for index in range( 1, len(self.network) - 2 , 2):
            layers.append([str(index) , str(index + 1)])
        logging.info(f"Layers to fuse {layers}")
        torch.ao.quantization.fuse_modules(self.network, layers, inplace=True)
    def get_quantize_configuration(self):
        if self.quantize_weight:
            if self.quantize_activation:
                return torch.ao.quantization.QConfig(
                    activation = torch.ao.quantization.FakeQuantize.with_args(
                        observer = torch.ao.quantization.MovingAverageMinMaxObserver,
                            dtype = self.quantize_activation_quantize_dtype,
                            reduce_range = self.quantize_activation_quantize_reduce_range,
                            quant_min = self.quantize_activation_quantize_min,
                            quant_max = self.quantize_activation_quantize_max,
                    ),
                    weight = torch.ao.quantization.FakeQuantize.with_args(
                        observer = torch.ao.quantization.MovingAverageMinMaxObserver ,
                            dtype = torch.quint8,
                            quant_min = -128,
                            quant_max = 127,
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def dqn_functional(
    exp_name: str = "dqn",
    seed:int = 42,
    torch_deterministic: bool = True,
    cuda: bool = False,
    track: bool = False,
    
    env_id: str = "CartPole-v1",
    total_timesteps: int = 500_000,
    learning_rate: float = 2.5e-4,
    buffer_size: int = 10000,
    gamma: float = 0.99,
    target_network_frequency: int = 500,
    batch_size: int = 128,
    start_e: float = 1,
    end_e: float = 0.05,
    exploration_fraction: float = 0.5,
    learning_starts: int = 10000,
    train_frequency: int = 10, 

    quantize_weight:bool = True,
    quantize_weight_bitwidth:int = 8,
    quantize_activation:bool = False,
    quantize_activation_bitwidth:int = 8,
    quantize_activation_quantize_min:int = 0,
    quantize_activation_quantize_max:int = 255,
    quanitize_activation_quantize_reduce_range:bool = False,
    quantize_activation_quantize_dtype:torch.dtype = torch.quint8 , 
    
    optimizer:str = "Adam",
   
   trial = optuna.trial.Trial,
):
    args = parse_args()
    # Experiment name
    args.seed = seed
    args.torch_deterministic = torch_deterministic
    args.cuda = cuda
    args.track = track
    args.env_id = env_id
    # Off_policy RL specific arguments  Deep Q Learning
    args.total_timesteps = total_timesteps
    args.learning_rate = learning_rate
    args.buffer_size = buffer_size
    args.gamma = gamma
    args.target_network_frequency = target_network_frequency
    args.batch_size = batch_size
    args.start_e = start_e
    args.end_e = end_e
    args.exploration_fraction = exploration_fraction
    args.learning_starts = learning_starts
    args.train_frequency = train_frequency
    ## Quantize
    ### Quantize Weight
    args.quantize_weight = quantize_weight
    args.quantize_weight_bitwidth = quantize_weight_bitwidth
    ### Quantize Activation
    args.quantize_activation = quantize_activation
    args.quantize_activation_bitwidth = quantize_activation_bitwidth
    args.quantize_activation_quantize_min = quantize_activation_quantize_min
    args.quantize_activation_quantize_max = quantize_activation_quantize_max
    args.quanitize_activation_quantize_reduce_range = quanitize_activation_quantize_reduce_range
    args.quantize_activation_quantize_dtype = quantize_activation_quantize_dtype
    
    args.optimizer = optimizer
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        
    
    max_episode_return =[]
    run = None
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config={ **vars(args) },
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"The device the DQN is running on: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    optimizer_of_choice =  None
    if args.optimizer == 'Adam':
        optimizer_of_choice = torch.optim.Adam
    elif args.optimizer == "hAdam":
        optimizer_of_choice = hAdam
    elif args.optimizer == 'Adan':
        optimizer_of_choice =   Adan
    q_network = QNetwork(
                        env = envs,
                        quantize_weight = args.quantize_weight,
                        quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                        quantize_activation = args.quantize_activation,
                        quantize_activation_bitwidth =  args.quantize_activation_bitwidth,
                         ).to(device)
    logging.info(f"QNetwork: {q_network} and the model is on the device: {next(q_network.parameters()).device}")
    optimizer = optimizer_of_choice(q_network.parameters(), lr=args.learning_rate)
    target_network =    QNetwork(
                        env = envs,
                        quantize_weight = args.quantize_weight,
                        quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                        quantize_activation = args.quantize_activation,
                        quantize_activation_bitwidth =  args.quantize_activation_bitwidth,
                         ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    logging.info(f"TargetNetwork: {target_network} and the model is on the device: {next(target_network.parameters()).device}")

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                max_episode_return.append(info["episode"]["r"])
                trial.report(info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 1000 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    
    ## Convert the model to 8 bit
    q_network.to("cpu")
    q_network.eval()
    torch.ao.quantization.convert(q_network, inplace=True)
    logging.info(f"Model converted to 8 bit and the size of the model is {size_of_model(q_network)}")
    logging.info(f"The q network is {q_network}")
    envs.close()
    writer.close()
    return run ,  np.average(max_episode_return)