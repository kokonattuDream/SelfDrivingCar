import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from replayMemory import ReplayMemory
from network import Network

# Implementing Deep Q Learning

class Dqn():
    """
    Whole process of Deep Q Learning Algorithm
    
    """
    
    def __init__(self, input_size, nb_action, gamma):
        """
        Initialize Deep Q Learning 
        
        @param input_size: input size of Neural Network
        @param nb_action: possible actions
        @param gamma: gamma paramemter of Deep Q-learning equation
        """
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        
        #optimization algorithm
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        
        #unsqueeze(0) => torch tensor of size 1 x input_size
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        """
        Select the next action of the car. 
        Use softmax function to get the best action while exploring different actions
        
        1.Generate Q values for all of possible actions
        2.It generate the probability distribution of all Q values
        3.Choose the action according to the probability distribution of each action
        
        @param state: input state of neural network
        @return: final action to play
        """
        #Convert torch tensor state to torch variable
        # T=100 Temperature parameter
        # Higher temperature, softmax function increases the certainty 
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) 
        
        # Get random draw of the probability distribution for eacj state
        action = probs.multinomial()
        
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """
        Train the deep neural network
      
        1.Get the output by forward propagation
        2.Get the target
        3.Compare output and target to computer last error
        4.Back propagate the last error to neural network
        5.Use stochastic gradient descent to update the weight 
            according how much contribute the last error
        
        @param batch_state: current state
        @param batch_next_state: next state
        @param batch_reward: reward
        @param batch_action: action
        """
        #output of all possible actions
        #gather(1, batch_action.unsqueeze(1)) to choose the only one action selected by NN
        #Want a simple vector, we need to kill fake dimension
        #squeeze(1) removes size of 1 from the dimension
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #Detach
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        #Bellman equation
        target = self.gamma*next_outputs + batch_reward
        
        #Temporal Difference
        td_loss = F.smooth_l1_loss(outputs, target)
        
        #re-initialize the optimizer
        self.optimizer.zero_grad()
        #back-ward propagation
        #free memory by retain variables
        td_loss.backward(retain_variables = True)
        #Update the weight of neural network
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        """
        Update everything once AI reaches to new state
        
        @param reward: last reward
        @param new_signal: last signal
        @return action to play
        """
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        
        self.reward_window.append(reward)
        
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action
    
    def score(self):
        """
        Get the current score
        
        @return score
        """
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        """
        Save the current Neural Network into file
        
        """
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        """
        Load the existed Neural Network
        
        """
        
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")