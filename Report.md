# Udacity Deep Reinforcement Learning Nanodegree - Project 1: Navigation

## Description of the implementation

### Algorithms
In this project I examined the 4 deep reinforcement learning algorithms from lesson 2 "Deep Q-Networks":

* [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Double Deep Q-Network](https://arxiv.org/abs/1509.06461)
* [Dueling Q-Network](https://arxiv.org/abs/1511.06581)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

##### Deep Q-Network
I started with the Deep Q-Network (DQN) algorithm using the following Q-Network architecture:  

- Input layer with 37 nodes for the states
- First Fully-Connected layer with 128 nodes
- Second Fully-Connected layer with 32 nodes
- Output layer with 4 nodes for the possible actions (up/down/left/right)
  
![](./resources/DQN-Agent-Network.png) 

##### Double Deep Q-Network
Then, I implemented the Double Deep Q-Network algorithm (DDQN), using the same Q-Network architecture as for the Deep Q-Network.

##### Dueling Network
Next I implemented the Dueling Double Deep Q-Network algorithm, which required an expansion of the network architecture:

![](./resources/Dueling-DQN-Agent-Network.png)  

##### Prioritized Experience Replay
Last but not least, I implemented the Prioritized Experience Replay algorithm as a further optimization  

## Training & Evaluation

I trained one agent at a time using the 4 DQN algorithms mentioned above until they got an average score of **+13** in the last **100** consecutive episodes. 
I have also adjusted the hyperparameters to get the best training results.

### Hyperparameters
The learning process can be influenced by changing the following parameters:  

In the [Navigation.ipynb](Navigation.ipynb) file:  

Parameters for controlling the training length and the epsilon-greedy action selection:

|Parameter         |Value |Description|
|:-----------------|-----:|:----------|
|n_episodes        |  1000|Maximum number of training episodes|
|max_t             |   500|Maximum number of timesteps per episode|
|eps_start         |  0.10|Starting value of epsilon, for epsilon-greedy action selection|
|eps_end           |  0.01|Minimum value of epsilon|
|eps_decay         |  0.98|Multiplicative factor (per episode) for decreasing epsilon|

Parameters to select algorithms. Here as an example for the **Double DQN Agent without Prioritized Experience Replay**:

|Parameter         |Value |Description|
|:-----------------|-----:|:----------|
|double_dqn        |  True|Whether or not to use Double Deep Q-Network algorithm|
|dueling_network   | False|Whether or not to use Dueling Network algorithm|
|prioritized_replay| False|Whether or not to use Prioritized Experience Replay|

In the [dqn_agent.py](dqn_agent.py) file:

Common DQN parameters:

|Parameter                 |Value     |Description|
|:-------------------------|---------:|:----------|
|BUFFER_SIZE               |       1e5|Replay buffer size|
|BATCH_SIZE                |        64|Batch size|
|GAMMA                     |      0.99|Discount factor for expected rewards|
|TAU                       |      1e-3|Multiplicative factor for updating the target network weights|
|LR                        |      5e-4|Learning rate|
|LR_DECAY                  |    0.9999|Multiplicative factor of learning rate decay|
|UPDATE_EVERY              |         4|How often to update the network|
|hidden_layers             | [128, 32]|Number and size of the Deep Q-Network layers|

Additional parameters for the **Dueling Network Algorithm**:

|Parameter                 |Value     |Description|
|:-------------------------|---------:|:----------|
|hidden_state_value_layers |  [64, 32]|Number and size of the value network of the Dueling Network|

Additional parameters for the **Prioritized Experience Replay Algorithm**:

|Parameter                 |Value     |Description|
|:-------------------------|---------:|:----------|
|alpha                     |       0.6|Determines how much prioritization is used; α = 0 corresponding to the uniform case|
|beta                      |       0.4|Amount of importance-sampling correction; β = 1 fully compensates for the non-uniform probabilities|
|beta_scheduler            |       1.0|Multiplicative factor (per sample) for increasing beta (should be >= 1.0)|

Then I evaluated each of the agents over exactly 100 episodes and determined the average score.

### Results in comparison
The following table shows the evaluation results of the different agents:

Agent                                         | # Training Episodes | Average Score (Evaluation 100 Episodes) 
:---------------------------------------------| ------------------: | ---------------------------------------: 
DQN                                           | 388                 | 14.47 
Double DQN                                    | 289                 | 15.18  
Dueling DQN                                   | 237                 | 13.60  
Dueling DQN and Prioritized Experience Replay | 339                 | 14.08  


The results of the different agents are pretty close together, but the **Double DQN Agent** achieved the best result. 
With a relatively short training time of **289** episodes, this achieved an average score of **15.18** in the evaluation over **100** episodes.

I was a bit surprised that the comparatively simple Double Deep Q-Network algorithm outperformed the Double Deep Q-Network algorithm with Prioritized Experience Replay optimization!

### Plot of Rewards
This graph shows the rewards per episode within the training phase of the Double Deep Q-Network Agent, as well as the moving mean.  
It illustrates that the Agent is able to receive an average reward of at least +13 over 100 episodes.  

In this case, the Agent solved the environment after **289 episodes**.

![](./resources/Training_Result_Double_DQN_Agent.png)


### Evaluation result 
This graph shows the rewards per episode within the evaluation of the Double Deep Q-Network Agent over 100 episodes and the average score.

![](./resources/Evaluation_Result_Double_DQN_Agent.png)

## Ideas for Future Work

1. Further hyperparameter tuning may produce better results. 
   

2. The agent's training was stopped here when the target score of +13 was reached. The agents could be trained until there is no longer any significant improvement in the score, so higher score values could be achieved.  


3. Agents with other combinations of the different algorithms could achieve better results, e.g. Double DQN with Prioritized Experience Replay.


4. Use of other network architectures, e.g. more hidden layers or more or fewer nodes per layer.

