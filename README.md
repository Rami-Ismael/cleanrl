DQN model applied to the this discrete environments CartPole-v1 
## Model Description 
The model was trained from the CleanRl library using the DQN algorithm 
## Intended Use & Limitation 
The model is intended to be used for the following environments CartPole-v1 
 and understand the implication of Quantization on this type of model from a pretrained state## Training Procdure 
### Training Hyperparameters 
``` 
The folloing hyperparameters were used during training: 
- learning_rate: 0.00025 
- batch_size: 128 
- gamma: 0.99 
- learning_starts: 10000 
- train_frequency: 10 
- target_network_frequency: 500 
