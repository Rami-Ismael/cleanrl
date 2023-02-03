# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51py
import argparse
import os
import random
import time
from distutils.util import strtobool
import logging
from quantize_methods import get_eager_quantization

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# Optimizer 
from algos.opt import Adan, hAdam

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
    parser.add_argument("--wandb-entity", type=str, default="compress_rl",
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
    parser.add_argument("--n-atoms", type=int, default=101,
        help="the number of atoms")
    parser.add_argument("--v-min", type=float, default=-100,
        help="the number of atoms")
    parser.add_argument("--v-max", type=float, default=100,
        help="the number of atoms")
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
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]
    ## fuse the model 
    ## Fuse the model
    def fuse_model(self):
        layers = list()
        for index in range( 0, len(self.network) - 2 , 2):
            layers.append([str(index) , str(index + 1)])
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
            sync_tensorboard = False ,  ## Disable wandb sync with tensorboard for google colab 
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        ## log the a test.log file to wandb 
        try:
            wandb.save("test.log")
        except:
            print("Could not save the test.log file to wandb")
        print("Save the test.log file to wandb")
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
    
    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    logging.info(f"The Q Value Network is {q_network}")
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
            w_observer= args.w_observer,
            w_fakequantize= args.w_fakequantize,
            activation_quantize= args.quantize_activation,
            activation_quantization_min = args.quantize_activation_quantize_min,
            activation_quantization_max = args.quantize_activation_quantize_max,
            activation_quantization_dtype = args.quantize_activation_quantize_dtype,
            activation_quantization_qscheme = args.quantize_activation_qscheme,
            activation_reduce_range = args.quantize_activation_reduce_range,
            a_observer= args.a_observer,
            a_fakequantize= args.a_fakequantize,
        )
        ## inplace will modify the model in place memory. There is no need to create a new model and qat module will be added
        torch.ao.quantization.prepare_qat(q_network, inplace=True)
        logging.info(f"The Q Value Network is {q_network}")
    optimizer = optimizer_of_choice(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    ## Target Network
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    ## Eager Quantization
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
            w_observer= args.w_observer,
            w_fakequantize= args.w_fakequantize,
            activation_quantize= args.quantize_activation,
            activation_quantization_min = args.quantize_activation_quantize_min,
            activation_quantization_max = args.quantize_activation_quantize_max,
            activation_quantization_dtype = args.quantize_activation_quantize_dtype,
            activation_quantization_qscheme = args.quantize_activation_qscheme,
            activation_reduce_range = args.quantize_activation_reduce_range,
            a_observer= args.a_observer,
            a_fakequantize= args.a_fakequantize,
        )
        ## inplace will modify the model in place memory. There is no need to create a new model and qat module will be added
        torch.ao.quantization.prepare_qat(target_network, inplace=True)
        logging.info(f"The Q Value Network is {target_network}")
    target_network.load_state_dict(q_network.state_dict())

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
            actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                if args.track:
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
                _, next_pmfs = target_network.get_action(data.next_observations)
                next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                # projection
                delta_z = target_network.atoms[1] - target_network.atoms[0]
                tz = next_atoms.clamp(args.v_min, args.v_max)

                b = (tz - args.v_min) / delta_z
                l = b.floor().clamp(0, args.n_atoms - 1)
                u = b.ceil().clamp(0, args.n_atoms - 1)
                # (l == u).float() handles the case where bj is exactly an integer
                # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                d_m_l = (u + (l == u).float() - b) * next_pmfs
                d_m_u = (b - l) * next_pmfs
                target_pmfs = torch.zeros_like(next_pmfs)
                for i in range(target_pmfs.size(0)):
                    target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

            _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
            loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

            if global_step % 100 == 0:
                writer.add_scalar("losses/loss", loss.item(), global_step)
                old_val = (old_pmfs * q_network.atoms).sum(1)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.track:
                    run.log({"losses/loss": loss.item()  }, step = global_step)
                    run.log({"losses/q_values": old_val.mean().item()  }, step = global_step)
                    run.log({"charts/SPS": int(global_step / (time.time() - start_time))  }, step = global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()
    try:
            wandb.save("test.log")
    except:
            print("Could not save the test.log file to wandb")
    if args.save_model:
        if args.quantize_weight and args.quantize_activation and args.quantize_weight_bitwidth == 8 and args.quantize_activation_bitwidth == 8:
            torch.ao.quantization.convert(q_network, inplace=True)
            logging.info(f"Model converted to 8 bit and the size of the model is {size_of_model(q_network)}")
            logging.info(f"The q network is {q_network}")
            model_path = os.path.join(args.save_path, "q_network.pt")
            torch.save(q_network.state_dict(), model_path)
        elif args.quantize_weights == False  and args.quantize_activations == False:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            model_data = {
                "model_weights": q_network.state_dict(),
                "args": vars(args),
            }
            torch.save(model_data, model_path)
            print(f"model saved to {model_path}")
            from cleanrl_utils.evals.c51_eval import evaluate
            episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub

                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "C51", f"runs/{run_name}", f"videos/{run_name}-eval")
        
    ## Stop logging of Weight and Bias
    if args.track:
        wandb.finish()
        run.finish()
