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

from algos.opt import Adan, hAdam


logging.basicConfig(filename="test_sac_continous_action.log", level=logging.NOTSET,
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
    parser.add_argument("--env-id", type=str, default="AntBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100_000,
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
    
    # Quantization 

    parser.add_argument("--quantize-weight", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-weight-bitwidth", type=int, default=8)
    parser.add_argument("--quantize-weight-quantize-min", type=int, default = -128)
    parser.add_argument("--quantize-weight-quantize-max", type=int, default = 127)
    parser.add_argument("--quantize-weight-dtype", type=str, default="qint8")
    parser.add_argument("--quantize-weight-qscheme", type=str, default="per_tensor_symmetric")
    parser.add_argument("--quantize-weight-reduce-range", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False)
    parser.add_argument("--w_observer", type=str, default="moving_average_min_max")
    parser.add_argument("--w_fakequantize", type=str, default="fake_quantize")
    
    ## Quantize Activation
    
    parser.add_argument("--quantize-activation" , type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-activation-bitwidth", type=int, default=8)
    parser.add_argument("--quantize-activation-quantize-min", type=int, default= 0)
    parser.add_argument("--quantize-activation-quantize-max", type=int, default= ( 2 ** 8 ) - 1)
    parser.add_argument("--quantize-activation-qscheme", type=str, default="per_tensor_affine")
    parser.add_argument("--quantize-activation-quantize-dtype", type=str, default="quint8")
    parser.add_argument("--quantize-activation-reduce-range", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--a_observer", type=str, default="moving_average_min_max")
    parser.add_argument("--a_fakequantize", type=str, default="fake_quantize")
    
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
class SoftQNetwork(nn.Module):
    def __init__(self, 
                 env 
                 ):
        super().__init__()
        '''
        logging.info(self.model)
        logging.info("Quantize Model")
        ## Prepare Quantize
        ### Fuse the model because their is relu
        self.fuse_model()
        ## Set the Quantization Configuration
        logging.info("Set the Quantization Configuration")
        logging.info(f"The Quantization Configuration is { self.get_quantization_config() }")
        self.model.qconfig = self.get_quantization_config()
        ## Prepare the QAT
        logging.info("Prepare the QAT")
        torch.ao.quantization.prepare_qat(self.model, inplace=True)
        logging.info(f"The model is {self.model}")
        '''
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
        torch.quantization.fuse_modules(self.model, [['0', '1'], ['2', '3']], inplace=True)
    def get_quantization_config(self):
        activation = torch.nn.Identity()
        weight = torch.nn.Identity()
        if self.quantize_weight:
            fq_weights = torch.quantization.FakeQuantize.with_args(
                        observer = torch.quantization.MovingAveragePerChannelMinMaxObserver , 
                        quant_min=-(2 ** self.quantize_weight_bitwidth) // 2,
                        quant_max=(2 ** self.quantize_weight_bitwidth) // 2 - 1,
                        dtype=torch.qint8, 
                        qscheme=torch.per_tensor_affine, 
                        reduce_range=False)
            if self.quantize_activation:
                fq_activation = torch.quantization.FakeQuantize.with_args(
                        observer = torch.quantization.MovingAveragePerChannelMinMaxObserver , 
                        quant_min=self.quantize_activation_quantize_min,
                        quant_max=self.quantize_activation_quantize_max,
                        dtype = getattr(torch, self.quantize_activation_quantize_dtype), 
                        qscheme=torch.per_tensor_affine, 
                        reduce_range=self.quanitize_activation_quantize_reduce_range
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


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        '''
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer("action_scale", torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0))
        self.register_buffer("action_bias", torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0))
        
        self.model = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.dequantize = torch.ao.quantization.DeQuantStub()
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
        self.model = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
    '''


    def forward(self, x):
        x = self.model(x) 
        if self.quantize_activation or self.quantize_weight:
            mean =  self.dequantize(self.fc_mean(x) )
            log_std = self.fc_logstd(x)
            log_std = self.dequantize(torch.tanh(log_std))
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
    def get_quantization_config(self):
        activation = torch.nn.Identity()
        weight = torch.nn.Identity()
        if self.quantize_weight:
            fq_weights = torch.quantization.FakeQuantize.with_args(
                        observer = torch.quantization.MovingAveragePerChannelMinMaxObserver , 
                        quant_min=-(2 ** self.quantize_weight_bitwidth) // 2,
                        quant_max=(2 ** self.quantize_weight_bitwidth) // 2 - 1,
                        dtype=torch.qint8, 
                        qscheme=torch.per_tensor_affine, 
                        reduce_range=False)
            if self.quantize_activation:
                fq_activation = torch.quantization.FakeQuantize.with_args(
                    observer = torch.quantization.MovingAveragePerChannelMinMaxObserver , 
                        quant_min=self.quantize_activation_quantize_min,
                        quant_max=self.quantize_activation_quantize_max,
                        dtype = getattr(torch, self.quantize_activation_quantize_dtype), 
                        qscheme=torch.per_tensor_affine, 
                        reduce_range=self.quanitize_activation_quantize_reduce_range
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
    def fuse_model(self):
        if self.quantize_activation or self.quantize_weight:
            torch.ao.quantization.fuse_modules(self.model ,  [ ["1" , "2"], ["3","4"] ] ,inplace = True )

if __name__ == "__main__":
    print("Starting the training a SAC agent")
    args = parse_args()
    if args.quantize_activation_quantize_min > args.quantize_activation_quantize_max:
        raise ValueError(f"{args.quantize_activation_quantize_min} is greater than {args.quantize_activation_quantize_max}")
    ## Set the Quantzation Dtypes to the a Torch Dtypes 
    if args.quantize_activation_quantize_dtype is not None:
        if args.quantize_activation_quantize_dtype == "quint8":
            args.quantize_activation_quantize_dtype = torch.quint8
        elif args.quantize_activation_quantize_dtype == "qint8":
            args.quantize_activation_quantize_dtype = torch.qint8
        else:
            raise ValueError(f"{args.quantize_activation_quantize_dtype} is not supported for quantization")
    if args.quantize_weight_dtype is not None:
        if args.quantize_weight_dtype == "quint8":
            args.quantize_weight_dtype = torch.quint8
        elif args.quantize_weight_dtype == "qint8":
            args.quantize_weight_dtype = torch.qint8
        else:
            raise ValueError(f"{args.quantize_weight_dtype} is not supported for quantization")
    ## Set the Quantization Scheme to the Torch Quantization Scheme instead of a string which is the default
    if args.quantize_activation_qscheme is not None and isinstance( args.quantize_activation_qscheme , str):
        if args.quantize_activation_qscheme == "per_tensor_symmetric":
            args.quantize_activation_qscheme = torch.per_tensor_symmetric
        elif args.quantize_activation_qscheme == "per_tensor_affine":
            args.quantize_activation_qscheme = torch.per_tensor_affine
        else:
            raise ValueError(f"{args.quantize_activation_qscheme} is not supported for quantization")
    if args.quantize_weight_qscheme is not None and isinstance(args.quantize_weight_qscheme, str):
        if args.quantize_weight_qscheme == "per_tensor_symmetric":
            args.quantize_weight_qscheme = torch.per_tensor_symmetric
        elif args.quantize_weight_qscheme == "per_tensor_affine":
            args.quantize_weight_qscheme = torch.per_tensor_affine
        else:
            raise ValueError(f"{args.quantize_weight_qscheme} is not supported for quantization")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard = False  , ## IMPORTANT to set this to False otherwise wandb will override the tensorboard logs and google colab does not work 
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
    

    actor = Actor(envs).to(device)
    
    ## Create the Q Network 
    qf1 = SoftQNetwork(envs , 
                  ).to(device)
    logging.info(f" The SoftQNetwork {qf1}")
    ## Apply the Quantization to the Q Network
    if args.quantize_weight or args.quantize_activation:
        qf1.quantize = True
        ## Quant Wrapper is a wrapper around the Q Network which applies the Quantization
        qf1.model =  torch.ao.quantization.QuantWrapper(qf1.model)
        logging.info(f" The QuantWrapper SoftQNetwork {qf1}")
        #3 Fuse the layers 
        qf1.fuse_model()
        
    qf2 = SoftQNetwork(envs , 
                quantize_weight = args.quantize_weight,
                       ).to(device)
    qf1_target = SoftQNetwork(envs , 
                              ).to(device)
    qf2_target = SoftQNetwork(envs , 
                              ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    ## Select the optimzer of your choice
    optimizer_of_choice =  None
    if args.optimizer == 'Adam':
        optimizer_of_choice = torch.optim.Adam
    elif args.optimizer == "hAdam":
        optimizer_of_choice = hAdam
    elif args.optimizer == 'Adan':
        optimizer_of_choice =   Adan
    
    logging.info(f"The optimizer {optimizer_of_choice} is being used")    ## Optimizers

        
    q_optimizer = optimizer_of_choice(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optimizer_of_choice(list(actor.parameters()), lr=args.policy_lr)

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
    
    if args.track:
        run.save("test.log")
        run.finish()
