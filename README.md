# DQN_for_OpenAI
I create a DQN model for OpenAI environment. 
This model can 'play' games well.

## Environment
I use 'Enduro-v0' and 'Enduro-ram-v0' game environment in OpenAI gym.
For 'Enduro-ram-v0', I create two DQN, first one has two full connected DNN as Q_net and Target_net.
Second one has two 1-dim CNN as Q_net and Target_net.

For 'Enduro-v0', I create a 2-dim CNN with 2 conv_layers, 2 max_pooling layers and 2 hidden layers.

### DQN

#### Tricks 
I use Double DQN to solve 'overEstimated' problem and SumTree Memeory Structure to implement priority train sot that the model can be trained faster.

### How to use it.
1. install openai gym
2. download this source code 
3. execute 'training.py'
4. A playing game environment will be showed.  
