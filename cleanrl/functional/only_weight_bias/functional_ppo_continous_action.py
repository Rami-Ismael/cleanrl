# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import logging
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import optuna
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(filename="tests.log", level=logging.INFO,
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
    parser.add_argument("--env-id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    # Quantization specific arguments
    ## Quantize Weight
    parser.add_argument("--quantize-weight", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-weight-bitwidth", type=int, default=8)
    ## Quantize Activation
    parser.add_argument("--quantize-activation" , type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-activation-bitwidth", type=int, default=8)
    parser.add_argument("--quantize-activation-quantize-dtype", type=str, default="qint8")
    
    ## Optimizer
    parser.add_argument("--optimizer", type=str, default="Adam")
    
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, 
                 envs,
                 quantize_weight:bool = False,
                 quantize_weight_bitwidth:int = 8,
                 quantize_activation:bool = False,
                 quantize_activation_bitwidth:int = 8,
                 quantize_activation_quantize_dtype:str = "quint8" , 
                 ):
        super().__init__()
        # Quantize Param Weight
        self.envs = envs
        self.quantize_weight = quantize_weight
        self.quantize_weight_bitwidth = quantize_weight_bitwidth
        ## Quantize Param Activation
        self.quantize_activation = quantize_activation
        self.quantize_activation_bitwidth = quantize_activation_bitwidth
        self.model_size = - 1
        if quantize_activation or quantize_weight:
            self.critic = nn.Sequential(
                torch.ao.quantization.QuantStub(),
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
                torch.ao.quantization.DeQuantStub(),
            )
            self.actor_mean = nn.Sequential(
                torch.ao.quantization.QuantStub(),
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
                torch.ao.quantization.DeQuantStub(),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
            ## Convert the
            ## Set the Quantization Configuration
            logging.info("Set the Quantization Configuration")
            logging.info(f"The Quantization Configuration is { self.get_quantization_config() }")
            self.actor_mean.qconfig = self.get_quantization_config()
            self.critic.qconfig = self.get_quantization_config()
            logging.info(f" Set qConfig for actor and critic to {self.actor_mean.qconfig}")
            ## Prepare the model for quantize aware training
            logging.info("Prepare the model for quantize aware training")
            logging.info(self.actor_mean)
            logging.info(self.critic)
            torch.ao.quantization.prepare_qat(self.actor_mean, inplace=True)
            torch.ao.quantization.prepare_qat(self.critic, inplace=True)
            logging.info("After prepare_qat")
            logging.info(self.actor_mean)
            logging.info(self.critic)
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        #self.model_size = self.size_of_model( self.critic) + self.size_of_model( self.actor_mean)
        self.model_size = self.size_of_model( self.critic
                                             )  + self.size_of_model( self.actor_mean)
        logging.info(f"Model size: {self.model_size}")
        ## Print out the model 
        

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    def size_of_model( self , model:nn.Sequential  ):
        name_file = "temp.pt"
        torch.save(self.state_dict(), name_file)
        size = os.path.getsize(name_file)
        return size
    def get_size(self):
        self.model_size = self.size_of_model( self.critic) + self.size_of_model( self.actor_mean)
        return self.model_size
    def inference(self, x):
        x = torch.randint(
            np.array(self.envs.single_observation_space.shape).prod(), (1, 1), dtype=torch.float32
        )
        
        num_samples = 200
        
        with torch.no_grad():
            start_time = time.time()
            for _ in range(num_samples):
                self.get_action_and_value(x)
                end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_per_sample = elapsed_time / num_samples
        
        return elapsed_time_per_sample
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
    def get_dtype(self):
        if self.quantize_activation_bitwidth <= 8:
            return getattr(torch, self.quantize_activation_quantize_dtype)
        elif self.quantize_activation_bitwidth == 16:
            return torch.int16

def ppo_functional(
    seed:int = 0,
    exp_name: str = "ppo",
    track: bool = False,
   
    env_id: str = "HopperBulletEnv-v0",
    total_timesteps: int = 1000000,
    learning_rate: float = 3e-4,
    
    quantize_weight_bitwidth:int = 8,
    quantize_activation_bitwidth:int = 8,
    
    optimizer: str = "Adam",
):
    args = parse_args()
    args.seed = seed
    args.exp_name = exp_name
    args.track = track
    
    args.env_id = env_id
    
    args.total_timesteps = total_timesteps
    args.learning_rate = learning_rate
    
    args.quantize_weight_bitwidth = quantize_weight_bitwidth
    args.quantize_activation_bitwidth = quantize_activation_bitwidth
    
    args.optimizer = optimizer
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs , 
                quantize_weight = args.quantize_weight,
                quantize_weight_bitwidth = args.quantize_weight_bitwidth,
                quantize_activation = args.quantize_activation,
                quantize_activation_bitwidth = args.quantize_activation_bitwidth,
                quantize_activation_quantize_dtype=args.quantize_activation_quantize_dtype,
                  ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    if args.track:
                        wandb.log({"charts/episodic_return": item["episode"]["r"], "charts/episodic_length": item["episode"]["l"]})
                        run.log({"charts/episodic_return": item["episode"]["r"]}, global_step)
                        run.log({"charts/episodic_length": item["episode"]["l"]}, global_step) 
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print("SPS:", int(global_step / (time.time() - start_time)))
        if track:
            run.log({"losses/value_loss": v_loss.item()} , global_step)
            run.log({"losses/policy_loss": pg_loss.item()}, global_step)
            run.log({"losses/entropy": entropy_loss.item()}, global_step)
            run.log({"losses/old_approx_kl": old_approx_kl.item()}, global_step)
            run.log({"losses/approx_kl": approx_kl.item()}, global_step)
            run.log({"losses/clipfrac": np.mean(clipfracs)}, global_step)
            run.log({"losses/explained_variance": explained_var}, global_step)
            run.log({"charts/SPS": int(global_step / (time.time() - start_time))})
    ## Convert the model to 8 bit model 
    agent.to("cpu")
    agent.eval()
    torch.ao.quantization.convert(agent, inplace=True)
    logging.info(f"Model converted to 8 bit model and the size of the model  is {agent.get_size()}")
    logging.info(f"The model is {agent}")
    envs.close()
    if track:
        run.save("test.log")
        run.finish()