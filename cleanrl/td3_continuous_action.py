# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import argparse
import logging
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
from algos.opt import Adan, hAdam
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(filename="tests.log", level=logging.NOTSET,
                    filemode='w',
                    format='%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s')


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
    parser.add_argument("--wandb-entity", type=str, default="compress_rl",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HopperBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    
    # Quantization specific arguments
    ## Quantize Weight
    parser.add_argument("--quantize-weight", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-weight-bitwidth", type=int, default=8)
    ## Quantize Activation
    parser.add_argument("--quantize-activation" , type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-activation-bitwidth", type=int, default=8)
    parser.add_argument("--quantize-activation-quantize-dtype", type=str, default="qint8")
    
    parser.add_argument("--optimizer", type=str, default="Adam")
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
                 quantize_activation_quantize_dtype:str = "quint8" , 
                 ):
        super().__init__()
        # Quantize Param Weight
        self.quantize_weight = quantize_weight
        self.quantize_weight_bitwidth = quantize_weight_bitwidth
        ## Quantize Param Activation
        self.quantize_activation = quantize_activation
        self.quantize_activation_bitwidth = quantize_activation_bitwidth
        if quantize_activation or quantize_weight:
            self.model = nn.Sequential(
                torch.quantization.QuantStub(),
                nn.Linear(env.observation_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256,1),
                torch.quantization.DeQuantStub(),
            )
            logging.info( self.model)
            logging.info("Quantize Model")
            
            self.fuse_modules()
            
            logging.info( self.model)
            logging.info("Quantize the model ")
            ## Fuse the model 
            
            ## Set the Quantization Configuration
            logging.info("Set the Quantization Configuration")
            logging.info(f"The Quantization Configuration is { self.get_quantization_config() }")
            self.model.qconfig = self.get_quantization_config()
            ## Prepare the QAT
            logging.info("Prepare the QAT")
            torch.ao.quantization.prepare_qat(self.model, inplace=True)
            logging.info(f"The model is {self.model}")
        else:
            self.model = nn.Sequential(
                nn.Linear(env.observation_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256,1),
            )

    def forward(self, x, a):
        return self.model(torch.cat([x, a], 1))
    def fuse_modules(self):
        if self.quantize_activation or self.quantize_weight:
            self.model = torch.quantization.fuse_modules( self.model ,  [["1" , "2"] , ["3" , "4"]] , inplace=True) 
            logging.info(f"Fuse layer , the model structure it's {self.model}")
    def get_quantization_config(self):
        activation = torch.nn.Identity()
        weight = torch.nn.Identity()
        if self.quantize_weight:
            fq_weights = torch.quantization.FakeQuantize.with_args(
                        observer = torch.quantization.MovingAverageMinMaxObserver , 
                        quant_min = -(2 ** self.quantize_weight_bitwidth) // 2,
                        quant_max=(2 ** self.quantize_weight_bitwidth) // 2 - 1,
                        dtype = torch.qint8,
                        qscheme=torch.per_tensor_affine, 
                        reduce_range=False)
            if self.quantize_activation:
                fq_activation = torch.quantization.FakeQuantize.with_args(
                        observer = torch.quantization.MovingAverageMinMaxObserver , 
                        quant_min = 0,
                        quant_max = (2 ** self.quantize_activation_bitwidth) - 1,
                        dtype =  torch.quint8 ,
                        qscheme=torch.per_tensor_affine, 
                        reduce_range = False
                    )
                return torch.ao.quantization.QConfig( activation = fq_activation, weight = fq_weights )
            else:
                return torch.ao.quantization.QConfig(activation = activation, weight = fq_weights)
    def size_of_model(self):
        name_file = "temp.pt"
        torch.save(self.model.state_dict(), name_file)
        size =  os.path.getsize(name_file)/1e6
        os.remove(name_file)
        return size
    def get_dtype(self):
        if self.quantize_activation_bitwidth <= 8:
            return getattr(torch, self.quantize_activation_quantize_dtype)
        elif self.quantize_activation_bitwidth == 16:
            return torch.int16


class Actor(nn.Module):
    def __init__(self, env, 
                quantize_weight:bool = False,
                 quantize_weight_bitwidth:int = 8,
                 quantize_activation:bool = False,
                 quantize_activation_bitwidth:int = 8,
                 quantize_activation_quantize_dtype:str = "qint8" , 
                 ):
        super().__init__()
        # Quantize Param Weight
        self.quantize_weight = quantize_weight
        self.quantize_weight_bitwidth = quantize_weight_bitwidth
        ## Quantize Param Activation
        self.quantize_activation = quantize_activation
        self.quantize_activation_bitwidth = quantize_activation_bitwidth
        self.quantize_activation_quantize_dtype = quantize_activation_quantize_dtype
        
        if quantize_activation or quantize_weight:
            self.model = nn.Sequential(
                torch.quantization.QuantStub(),
                nn.Linear(env.observation_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, env.action_space.shape[0]),
                nn.Tanh(),
                torch.quantization.DeQuantStub(),
            )
            logging.info( self.model)
            logging.info("Quantize the model ")
            ## Fuse the model 
            self.fuse_modules()
            ## Set the Quantization Configuration
            logging.info("Set the Quantization Configuration")
            logging.info(f"The Quantization Configuration is { self.get_quantization_config() }")
            self.model.qconfig = self.get_quantization_config()
            ## Prepare the QAT
            logging.info("Prepare the QAT")
            torch.ao.quantization.prepare_qat(self.model, inplace=True)
            logging.info(f"The model is {self.model}")
        else:
            self.model = nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, env.action_space.shape),
                nn.Tanh(),
            )
        self.register_buffer("action_scale", torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0))
        self.register_buffer("action_bias", torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0))

    def forward(self, x):
        return self.model(x) * self.action_scale + self.action_bias
    def fuse_modules(self):
        if self.quantize_activation or self.quantize_weight:
            self.model = torch.quantization.fuse_modules( self.model ,  [["1" , "2"] , ["3" , "4"]] , inplace=True) 
            logging.info(f"Fuse layer , the model structure it's {self.model}")
    def get_quantization_config(self):
        activation = torch.nn.Identity()
        weight = torch.nn.Identity()
        if self.quantize_weight:
            fq_weights = torch.quantization.FakeQuantize.with_args(
                        observer = torch.quantization.MovingAverageMinMaxObserver , 
                        quant_min = -(2 ** self.quantize_weight_bitwidth) // 2,
                        quant_max=(2 ** self.quantize_weight_bitwidth) // 2 - 1,
                        dtype = self.get_dtype(), 
                        qscheme=torch.per_tensor_affine, 
                        reduce_range=False)
            if self.quantize_activation:
                fq_activation = torch.quantization.FakeQuantize.with_args(
                        observer = torch.quantization.MovingAverageMinMaxObserver , 
                        quant_min = -(2 ** self.quantize_activation_bitwidth) // 2,
                        quant_max = (2 ** self.quantize_activation_bitwidth) // 2 - 1,
                        dtype =  self.get_dtype() ,
                        qscheme=torch.per_tensor_affine, 
                        reduce_range = False , 
                    )
                return torch.ao.quantization.QConfig( activation = fq_activation, weight = fq_weights )
            else:
                return torch.ao.quantization.QConfig(activation = activation, weight = fq_weights)
    def size_of_model(self):
        name_file = "temp.pt"
        torch.save(self.model.state_dict(), name_file)
        size =  os.path.getsize(name_file)/1e6
        os.remove(name_file)
        return size
    def get_dtype(self):
        if self.quantize_activation_bitwidth <= 8:
            return getattr(torch, self.quantize_activation_quantize_dtype)
        elif self.quantize_activation_bitwidth == 16:
            return torch.int16


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
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

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs,
                quantize_weight = args.quantize_weight,
                quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                quantize_activation = args.quantize_activation,
                quantize_activation_bitwidth = args.quantize_activation_bitwidth,
                quantize_activation_quantize_dtype=args.quantize_activation_quantize_dtype,
                  ).to(device)
    qf1 = QNetwork(envs, 
                  quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_bitwidth = args.quantize_activation_bitwidth,
                   ).to(device)
    qf2 = QNetwork(envs, 
                  quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_bitwidth = args.quantize_activation_bitwidth,
                   ).to(device)
    qf1_target = QNetwork(envs,
                quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_bitwidth = args.quantize_activation_bitwidth,
                          ).to(device)
    qf2_target = QNetwork(envs , 
                  quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_bitwidth = args.quantize_activation_bitwidth,
                          ).to(device)
    target_actor = Actor(envs , 
                  quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_bitwidth = args.quantize_activation_bitwidth,
                  quantize_activation_quantize_dtype=args.quantize_activation_quantize_dtype,
                         ).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    optimizer_of_choice = None
    if args.optimizer == "Adam":
        optimizer_of_choice = torch.optim.Adam
    elif args.optimizer == "hAdam":
        optimizer_of_choice = hAdam
    elif args.optizer == "Adan":
        optimizer_of_choice =  Adan
    q_optimizer = optimizer_of_choice(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optimizer_of_choice(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
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
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
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
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(torch.Tensor(actions[0])) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                )

                next_state_actions = (target_actor(data.next_observations) + clipped_noise.to(device)).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
