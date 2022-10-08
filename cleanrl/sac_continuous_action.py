# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import logging
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--env-id", type=str, default="HopperBulletEnv-v0",
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
class SoftQNetwork(nn.Module):
    def __init__(self, env , 
                 quantize_weight:bool = False,
                 quantize_weight_bitwidth:int = 8,
                 quantize_activation:bool = False,
                 quantize_activation_bitwidth:int = 8,
                 quantize_activation_quantize_min:int = 0,
                 quantize_activation_quantize_max:int = 255,
                 quantize_activation_quantize_reduce_range:bool = False,
                 quantize_activation_quantize_dtype:torch.dtype = torch.quint8 , 
                 backend:str = 'fbgemm',
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
        self.quanitize_activation_quantize_reduce_range = quantize_activation_quantize_reduce_range
        self.quantize_activation_quantize_dtype = quantize_activation_quantize_dtype
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        if self.quantize_weight or self.quantize_activation:
            self.model = nn.Sequential(
                torch.ao.quantization.QuantStub(),
                nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                torch.ao.quantization.DeQuantStub()
            )
            logging.info(self.model)
            logging.info("Quantize Model")
            ## Prepare Quantize
            ### Fuse the model because their is relu
            self.fuse_model()
            ## Set the Quantization Configuration
            logging.info("Set the Quantization Configuration")
            logging.info(f"The Quantization Configuration is { self.get_quantization_config()}")
            self.model.qconfig = self.get_quantization_config()
            ## Prepare the QAT
            logging.info("Prepare the QAT")
            torch.ao.quantization.prepare_qat(self.model, inplace=True)
            logging.info(f"The model is {self.model}")
        else:
            self.model = nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.model(x)
    def fuse_model(self):
        if self.quantize_weight or self.quantize_activation:
            torch.quantization.fuse_modules(self.model, [['1', '2'], ['3', '4']], inplace=True)
        else:
            torch.ao.quantization.fuse_modules(self.model, [['0', '1'], ['2', '3']], inplace=True)
    def get_quantize_configuration(self):
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
    def size_of_model(self):
        name_file = "temp.pt"
        torch.save(self.model.state_dict(), name_file)
        size =  os.path.getsize(name_file)/1e6
        os.remove(name_file)
        return size


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env , 
                 quantize_weight:bool = False,
                 quantize_weight_bitwidth:int = 8,
                 quantize_activation:bool = False,
                 quantize_activation_bitwidth:int = 8,
                 quantize_activation_quantize_min:int = 0,
                 quantize_activation_quantize_max:int = 255,
                 quantize_activation_quantize_reduce_range:bool = False,
                 quantize_activation_quantize_dtype:torch.dtype = torch.quint8 , 
          ):
        super().__init__()
        
        self.quantize_weight = quantize_weight
        self.quantize_weight_bitwidth = quantize_weight_bitwidth
        self.quantize_activation = quantize_activation
        self.quantize_activation_bitwidth = quantize_activation_bitwidth
        self.quantize_activation_quantize_min = quantize_activation_quantize_min
        self.quantize_activation_quantize_max = quantize_activation_quantize_max
        self.quanitize_activation_quantize_reduce_range = quantize_activation_quantize_reduce_range
        self.quantize_activation_quantize_dtype = quantize_activation_quantize_dtype
        
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer("action_scale", torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0))
        self.register_buffer("action_bias", torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0))
        
        if self.quantize_activation or self.quantize_weight:
            self.model = nn.Sequential(
                torch.ao.quantization.QuantStub(),
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            logging.info(self.model) 
            ##  Fuse the model
            self.fuse_model()
            logging.info(f"After the model being used" , self.model)
            ## Set the Quantization Configuration
            self.model.qconfig = self.get_quantization_config()
            self.fc_mean.qconfig = self.get_quantization_config()
            self.fc_logstd.qconfig = self.get_quantization_config()
            logging.info(f"The Quantization Configuration of the Model is {self.model.qconfig}\n")
            ## prepare QAT
            torch.ao.quantization.prepare_qat(self.model, inplace=True)
            torch.ao.quantization.prepare_qat(self.fc_mean, inplace=True)
            torch.ao.quantization.prepare_qat(self.fc_logstd, inplace=True)
            logging.info(f"The Model is {self.model}\n")
            logging.info(f"The fc_mean is {self.fc_mean}\n")
            logging.info(f"The fc_mean is {self.fc_logstd}\n")
        else:
            self.model = nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )


    def forward(self, x):
        x = self.model(x) 
        if self.quantize_activation or self.quantize_weight:
            mean = torch.ao.quantization.dequantize(self.fc_mean(x))
            log_std = self.fc_logstd(x)
            log_std = torch.ao.quantization(torch.tanh(log_std))
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
            return mean, log_std
        else:
            mean = self.fc_mean(x)
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
            return mean, log_std
    def get_action(self, x):
        mean, log_std = self(x)
        std =  log_std.exp()  ## Sample from a norrmal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    def get_quantize_configuration(self):
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
    def size_of_model(self):
        name_file = "temp.pt"
        torch.save(self.model.state_dict(), name_file)
        size =  os.path.getsize(name_file)/1e6
        os.remove(name_file)
        return size
    def fuse_model(self):
        if self.quantize_activation or self.quantize_weight:
            torch.ao.quantization.prepare_qat(self.model, [ ["1" , "2"] , [ "3" , "4"]] ,  inplace = True)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    ## Convert the string into a dtype
    if args.quantize_activation_quantize_dtype is not None:
        if args.quantize_activation_quantize_dtype == "quint8":
            args.quantize_activation_quantize_dtype = torch.quint8 
        elif args.quantize_activation_quantize_dtype == "qint8":
            args.quantize_activation_quantize_dtype = torch.qint8
        else:
            raise ValueError(f"Unknown dtype '{torch.dtype}'")
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

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs , 
                  quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_quantize_min = args.quantize_activation_quantize_min,
                  quantize_activation_quantize_max = args.quantize_activation_quantize_max,
                  quantize_activation_quantize_reduce_range = args.quantize_activation_quantize_reduce_range,
                  quantize_activation_quantize_dtype = args.quantize_activation_quantize_dtype,
                  ).to(device)
    qf1 = SoftQNetwork(envs , 
                  quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_quantize_min = args.quantize_activation_quantize_min,
                  quantize_activation_quantize_max = args.quantize_activation_quantize_max,
                  quantize_activation_quantize_reduce_range = args.quantize_activation_quantize_reduce_range,
                  quantize_activation_quantize_dtype = args.quantize_activation_quantize_dtype,
                  ).to(device)
    qf2 = SoftQNetwork(envs , 
                quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_quantize_min = args.quantize_activation_quantize_min,
                  quantize_activation_quantize_max = args.quantize_activation_quantize_max,
                  quanitize_activation_quantize_reduce_range = args.quanitize_activation_quantize_reduce_range,
                  quantize_activation_quantize_dtype = args.quantize_activation_quantize_dtype,
                       ).to(device)
    qf1_target = SoftQNetwork(envs , 
                    quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_quantize_min = args.quantize_activation_quantize_min,
                  quantize_activation_quantize_max = args.quantize_activation_quantize_max,
                  quanitize_activation_quantize_reduce_range = args.quanitize_activation_quantize_reduce_range,
                  quantize_activation_quantize_dtype = args.quantize_activation_quantize_dtype,
                              ).to(device)
    qf2_target = SoftQNetwork(envs , 
                quantize_weight = args.quantize_weight,
                  quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                  quantize_activation = args.quantize_activation,
                  quantize_activation_quantize_min = args.quantize_activation_quantize_min,
                  quantize_activation_quantize_max = args.quantize_activation_quantize_max,
                  quanitize_activation_quantize_reduce_range = args.quanitize_activation_quantize_reduce_range,
                  quantize_activation_quantize_dtype = args.quantize_activation_quantize_dtype,
                              ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

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
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

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
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
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
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
