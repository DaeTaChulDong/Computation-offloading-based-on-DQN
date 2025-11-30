import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.1
EPISILO = 0.995
MEMORY_CAPACITY = 1000
Q_NETWORK_ITERATION = 100

NUM_ACTIONS = 66
NUM_STATES = 6

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.1)
                
class DuelingNet(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(DuelingNet, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(NUM_STATES, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.feature_layer.apply(init_weights)
        
        #  Value Stream-State value
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # V(s)
        )
        self.value_stream.apply(init_weights)
        
        # Advantage Stream-Action value
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS) # A(s, a)
        )
        self.advantage_stream.apply(init_weights)
        


    def forward(self,x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)        # V(s)
        advantages = self.advantage_stream(features) # A(s, a)
        
        # Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        q_vals = values + (advantages - advantages.mean(1, keepdim=True))
        
        return q_vals

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = DuelingNet(), DuelingNet()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.flag = 0
        self.epsilon = 0.9999

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= self.epsilon:# greedy policy
            action_value = self.eval_net.forward(state)
            #if self.memory_counter > MEMORY_CAPACITY:
             #   print(action_value)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] #if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            self.flag = 1
            self.epsilon = self.epsilon * 0.9999
            #print(self.epsilon)
            #if self.memory_counter > MEMORY_CAPACITY:
             #   print(action)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            self.flag = 0
            #action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action,self.epsilon


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        #print("1",self.eval_net(batch_state))
        #print("2",batch_action)
        #print("q_eval",q_eval)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        #print("batch_reward",batch_reward)
        #print("GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)",GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1))
        #print("q_target",q_target)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #for parms in self.eval_net.parameters():
         #   print('-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
