--- 
language: en 
license: apache-2.0 
model-index: 
  - name: CartPole-v1__functional_dqn__0__1671479574 
---
DQN model applied to the this discrete environments CartPole-v1 
## Model Description 
The model was trained from the CleanRl library using the DQN algorithm 
## Intended Use & Limitation 
The model is intended to be used for the following environments CartPole-v1 
 and understand the implication of Quantization on this type of model from a pretrained state## Training Procdure 
### Training Hyperparameters 
The folloing hyperparameters were used during training: 
- exp_name: functional_dqn 
- seed: 0 
- torch_deterministic: True 
- cuda: False 
- track: True 
- wandb_project_name: cleanRL 
- wandb_entity: compress_rl 
- capture_video: False 
- env_id: CartPole-v1 
- total_timesteps: 500000 
- learning_rate: 0.00025 
- buffer_size: 10000 
- gamma: 0.99 
- target_network_frequency: 500 
- batch_size: 128 
- start_e: 1 
- end_e: 0.05 
- exploration_fraction: 0.5 
- learning_starts: 10000 
- train_frequency: 10 
- optimizer: Adan 
- wandb_project: cleanrl 
### Framework and version 
Pytorch 1.12.1+cu102 
gym 0.23.1 
Weights and Biases 0.13.3 
Hugging Face Hub 0.11.1 
