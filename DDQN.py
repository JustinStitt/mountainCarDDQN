import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

cuda_enabled = T.cuda.is_available()
print('cuda enabled? : ', cuda_enabled)

saved_model_path = 'saved_models/trained_model_'

class DoubleDQN(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DoubleDQN, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions

        #set up network model architecture
        self.fc1 = nn.Linear(*self.input_dims, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512, self.n_actions)
        #Linear model. 1 input layer, 2 hidden layers, 1 output layer w/ n_actions outputs

        #state optimizer and loss func / cost criterion
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        #if cuda capabilities are enabled, set device to cuda -- else cpu
        if cuda_enabled:#set default tensor data type if we have cuda capabilities
            T.set_default_tensor_type('torch.cuda.FloatTensor')#use gpu
        self.device = T.device('cuda:0' if (cuda_enabled) else 'cpu')#ternary
        #cast to device
        self.to(self.device)

    def forward(self, state):
        """
            Given a state, pass sequentially through the network and return
            the action space
        """
        x = F.relu(self.fc1(state))#use ReLu activation func for all layers except last (output layer)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions
    def save_model(self, type):
        print('...saving model...')
        path = saved_model_path + type + '.pt'
        T.save(self.state_dict(), path)
    def load_model(self, type):
        print('...loading model...')
        path = saved_model_path + type + '.pt'
        self.load_state_dict(T.load(path))

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                    max_mem_size = 50000, eps_end = 0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]#list comprehension
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0#keep track of how many iterations have been run (for replace)
        self.replace_target = 100#how many iterations till we replace our target network

        #local network is the one we are training
        self.Q_local = DoubleDQN(lr,n_actions = n_actions, input_dims = input_dims)
        #target network is what we use to determine the best action available (for bellman eq.)
        self.Q_target = DoubleDQN(lr,n_actions = n_actions, input_dims = input_dims)

        #setup numpy zeros-arrays with max mem size
        sz = self.mem_size
        dims = input_dims
        self.state_memory = np.zeros((sz,*dims), dtype = np.float32)
        self.new_state_memory = np.zeros((sz,*dims), dtype = np.float32)
        self.action_memory = np.zeros(sz, dtype = np.int64)#action's are ints (one-hot encoded??!)
        self.reward_memory = np.zeros(sz, dtype = np.float32)
        self.terminal_memory = np.zeros(sz, dtype = np.bool)#done flags

    def store_transition(self, state, action, reward, state_, terminal):
        """
            store a new memory.
            state is the current environment observation
            action is the action we performed
            reward is the reward we got
            state_ is the next state after our action/reward feedback
            terminal is a done flag if we finished the epsiode (99% sure ???)
        """
        index = self.mem_cntr % self.mem_size#handles wrap-around. We rewrite old memories after mem counter exceeds max_mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        #we added a memory so increase memory counter
        self.mem_cntr += 1

    def choose_action(self, observation):
        #epsilon-greedy algorithm
        if np.random.random() > self.epsilon:#exploit
            #convert observation to a state tensor and cast to device
            state = T.tensor([observation], dtype= T.float).to(self.Q_local.device)
            #evaluates Q-Values for each action given a state
            actions = self.Q_local.forward(state)
            #argmax() chooses best action (highest Q-Value)
            action = T.argmax(actions).item()#.item() to get index
        else:#explore
            action = np.random.choice(self.action_space)#random action in Agent's action space
        #return our chosen action (either exploited or explored )
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:#if we can't complete a batch, then dont learn
            return
        #zero out our gradient. (stops exploding gradient because torch remembers past gradient pos ??? I think...)
        self.Q_local.optimizer.zero_grad()
        #replace target network every replace_target iterations
        self.replace_target_network()
        max_mem = min(self.mem_cntr, self.mem_size)#since mem_cntr can be larger than our saved memory due to wrap-around hack, we must take the smaller one

        #define our batch stochastically
        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        #random.choice returns batch_size amount of values in the range of max_mem. randomly.
        #generate our batch_index using np.arange ex: b_i = [0,1,2,3,4,batch_size - 1]
        batch_index = np.arange(self.batch_size)

        #define our batches with each given metric (rule of 5)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_local.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_local.device)
        action_batch = T.tensor(self.action_memory[batch]).to(self.Q_local.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_local.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_local.device)

        #forward pass of local and target networks
        q_pred = self.Q_local.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_eval = self.Q_local.forward(new_state_batch)#no idea what this is for
        #i think q_eval is for finding the best predicted action for the new_state_batch but idk why we want it
        max_actions = T.argmax(q_eval, dim = 1)
        q_next[terminal_batch] = 0.0
        # estimated future reward is the self.gamma*q_next[batch_index, max_actions] part, where gamma discounts future rewards
        q_target = reward_batch + self.gamma*q_next[batch_index, max_actions]#bellman equation
        loss = self.Q_local.loss(q_target, q_pred).to(self.Q_local.device)#basically put, how good could our action have been(q-target), vs how good it actually was (q-pred)
        loss.backward()

        self.Q_local.optimizer.step()
        self.iter_cntr += 1
        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = max((self.epsilon - self.eps_dec), self.eps_min)

    def replace_target_network(self):
        if self.iter_cntr is not None and self.iter_cntr % self.replace_target == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())

    def save_agent(self):
        #save local and target network
        self.Q_local.save_model("local")
        self.Q_target.save_model("target")

    def load_agent(self):
        #load local and target network
        self.Q_local.load_model("local")
        self.Q_target.load_model("target")
