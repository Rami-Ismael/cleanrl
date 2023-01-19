# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
from cmath import log
import os
import random
import time
from distutils.util import strtobool
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from quantize_methods import get_eager_quantization


import gym
from algos.opt import Adan, hAdam
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.ao.quantization.fake_quantize import  default_weight_fake_quant
## Set up Warning # Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)





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
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
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
    parser.add_argument("--quantize-weight", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-weight-bitwidth", type=int, default=8)
    parser.add_argument("--quantize-weight-quantize-min", type=int, default=  ( 2 ** ( 8 ) ) // 2)
    parser.add_argument("--quantize-weight-quantize-max", type=int, default = ( 2 ** 8 ) // 2 - 1)
    parser.add_argument("--quantize-weight-dtype", type=str, default="qint8")
    parser.add_argument("--quantize-weight-qscheme", type=str, default="per_tensor_symmetric")
    parser.add_argument("--quantize-weight-reduce-range", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False)
    ## Quantize Activation
    parser.add_argument("--quantize-activation" , type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--quantize-activation-bitwidth", type=int, default=8)
    parser.add_argument("--quantize-activation-quantize-min", type=int, default= 0)
    parser.add_argument("--quantize-activation-quantize-max", type=int, default= ( 2 ** 8 ) - 1)
    parser.add_argument("--quantize-activation-qscheme", type=str, default="per_tensor_asymmetric")
    parser.add_argument("--quantize-activation-quantize-dtype", type=str, default="quint8")
    parser.add_argument("--quantize-activation-reduce-range", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    ## Other papers algorithm and ideas
    parser.add_argument("--optimizer" , type=str, default="Adam")
    args = parser.parse_args()
    # fmt: on
    return args
## A method to check if my code is runnning on google colab
import sys
def on_colab() -> bool:
    if 'google.colab' in sys.modules:
        return True
    else:
        return False

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
    def __init__(self, 
                 env ,
                 ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )
        self.quantize = False
        self.quantize_modules = torch.ao.quantization.QuantStub()
        self.dequantize_modules = torch.ao.quantization.DeQuantStub()
        logging.info(f"The model is {self.network} GB")  
        logging.info(f"The size of the model is {size_of_model(self.network)}")
    def forward(self, x ):
        return  self.dequantize_modules(
                        self.network(
                            self.quantize_modules(
                                x))) if self.quantize else self.network(x)
    ## Fuse the model
    def fuse_model(self):
        layers = list()
        for index in range( 0, len(self.network) - 2 , 2):
            layers.append([str(index) , str(index + 1)])
        logging.info(f"Layers to fuse {layers}")
        print(f"Layers to fuse {layers}")
        torch.ao.quantization.fuse_modules(self.network, layers, inplace=True)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    print("Starting the training a DQN agent")
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
        ## log the
        logging.info(f"Logging the model to wandb")
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
    ## Select the optimzer of your choice
    optimizer_of_choice =  None
    if args.optimizer == 'Adam':
        optimizer_of_choice = torch.optim.Adam
    elif args.optimizer == "hAdam":
        optimizer_of_choice = hAdam
    elif args.optimizer == 'Adan':
        optimizer_of_choice =   Adan
    
    logging.info(f"The optimizer {optimizer_of_choice} is being used")

    q_network = QNetwork(
                        env = envs,
                         )
    logging.info(f"QNetwork: {q_network} ")
    if args.quantize_weight or args.quantize_activation:
        q_network.quantize = True
        ### Eager Mode Quantization
        '''
        1. Fuse the model
        2. The Quantization Configuration for QAT
        3 . Call the Prepare function
        '''
        ## Fuse the layer you must do this before you call the prepare function and set the model to eval mode
        q_network.eval()
        q_network.fuse_model()
        ## Set the model to train mode to set the qat configuration of the model 
        q_network.train()
        print(args.quantize_activation)
        q_network.qconfig = get_eager_quantization(
            weight_quantize = args.quantize_weight,
            weight_quantization_min =  args.quantize_weight_quantize_min , 
            weight_quantization_max = args.quantize_weight_quantize_max,
            weight_quantization_dtype = args.quantize_weight_dtype,
            weight_reduce_range= args.quantize_weight_reduce_range,
            activation_quantize= args.quantize_activation,
            activation_quantization_min = args.quantize_activation_quantize_min,
            activation_quantization_max = args.quantize_activation_quantize_max,
            activation_quantization_dtype = args.quantize_activation_quantize_dtype,
            activation_quantization_qscheme = args.quantize_activation_qscheme,
            activation_reduce_range = args.quantize_activation_reduce_range,
        )
        ## inplace will modify the model in place memory. There is no need to create a new model and qat module will be added
        torch.ao.quantization.prepare_qat(q_network, inplace=True)
    optimizer = optimizer_of_choice(q_network.parameters(), lr=args.learning_rate)
    target_network =    QNetwork(
                        env = envs,
                         ).to(device)
    if args.quantize_weight or args.quantize_activation:
        target_network.quantize = True
        
        ### Eager Mode Quantization
        '''
        1. Fuse the model
        2. The Quantization Configuration for QAT
        3 . Call the Prepare function
        '''
        ## Fuse the layer you must do this before you call the prepare function and set the model to eval mode
        target_network.eval()
        target_network.fuse_model()
        ## Set the model to train mode to set the qat configuration of the model 
        target_network.train()
        print(args.quantize_activation)
        target_network.qconfig = get_eager_quantization(
            weight_quantize = args.quantize_weight,
            weight_quantization_min =  args.quantize_weight_quantize_min , 
            weight_quantization_max = args.quantize_weight_quantize_max,
            weight_quantization_dtype = args.quantize_weight_dtype,
            weight_reduce_range= args.quantize_weight_reduce_range,
            activation_quantize= args.quantize_activation,
            activation_quantization_min = args.quantize_activation_quantize_min,
            activation_quantization_max = args.quantize_activation_quantize_max,
            activation_quantization_dtype = args.quantize_activation_quantize_dtype,
            activation_quantization_qscheme = args.quantize_activation_qscheme,
            activation_reduce_range = args.quantize_activation_reduce_range,
        )
        ## inplace will modify the model in place memory. There is no need to create a new model and qat module will be added
        torch.ao.quantization.prepare_qat(target_network, inplace=True)       
    target_network.load_state_dict(q_network.state_dict())
    logging.info(f"TargetNetwork: {target_network} and the model is on the device: {next(target_network.parameters()).device}")
    ## Before the training start. I want to set the fake_quant and oberr to be enable. When iniltization scale in Fake Quantize are inf and -inf
    q_network.apply( torch.ao.quantization.enable_observer )
    q_network.apply( torch.ao.quantization.enable_fake_quant )
    target_network.apply( torch.ao.quantization.enable_observer )
    target_network.apply( torch.ao.quantization.enable_fake_quant )
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
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                ## write directly to weight and  bias. Because for some wierd reason sync tensorboard does not work yet  this is work around right now
                if on_colab() or ( args.track and True):
                    run.log(data = {"charts/episodic_return": info["episode"]["r"]}  , step = global_step)
                    run.log({"charts/episodic_length": info["episode"]["l"]  }, step = global_step)
                    run.log({"charts/epsilon": epsilon  }, step = global_step)
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

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if on_colab() or ( args.track and True):
                    run.log({"losses/td_loss": loss  }, step = global_step)
                    run.log({"losses/q_values": old_val.mean().item()  }, step = global_step)
                    run.log({"charts/SPS": int(global_step / (time.time() - start_time))  }, step = global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    ## Stop logging of Weight and Bias
    if args.track:
        wandb.finish()
        run.finish()
    ## Convert the model to 8 bit
    q_network.to("cpu")
    q_network.eval()
    if args.save_model:
        try:
            torch.ao.quantization.convert(q_network, inplace=True)
            logging.info(f"Model converted to 8 bit and the size of the model is {size_of_model(q_network)}")
            logging.info(f"The q network is {q_network}")
            model_path = os.path.join(args.save_path, "q_network.pt")
            torch.save(q_network.state_dict(), model_path)
            
            try:
                from cleanrl_utils.evals.dqn_eval import evaluate
            
                episodic_returns = evaluate(
                    model_path,
                    make_env,
                    args.env_id , 
                    eval_episodes=10,
                    run_name=f"{run_name}-eval",
                    Model=QNetwork,
                    device= device , 
                    epsilon = 0.05,
                )
                for idx, episodic_return in enumerate(episodic_returns):
                    writer.add_scalar("charts/eval_episodic_return", episodic_return, idx)
                if on_colab() or ( args.track and True):
                    run.log({"charts/eval_episodic_return": episodic_return  }, step = idx)
                if args.upload_model:
                    from cleanrl_utils.huggingface import push_to_hub

                    repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                    repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                    push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")
            except Exception as e:
                print(f"Eval did not happen\n: {e}")
            


                
        except Exception as e:
            logging.info(f"Conversion to 8 bit did not happen\n: {e}")
    envs.close()
    writer.close()
