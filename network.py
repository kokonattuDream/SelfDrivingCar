import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """
    Neural Network which is the heart of the AI.
    -It predicts the action to play.
    -Inherit the torch.nn.Module
    
    """
    
    def __init__(self, input_size, nb_action):
        """
        Initialize the Neural Network 

        @param input_size: input size of the neural network
        @param nb_action: possible actions
        """
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        #Connection with input layer and hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        #Connection with hidden layer and output layer
        self.fc2 = nn.Linear(30, nb_action)
    
    #Return Q-values of each action
    def forward(self, state):
        """
        Forward propagation 
        
        @param state: input of neural network
        @return q_values: q values of the predicted action
        """
        
        #Pass the input state to fully connected neural network
        #Use rectified function to activate neurons
        x = F.relu(self.fc1(state))
        #Get q-value from the output neurons of neural network
        q_values = self.fc2(x)
        return q_values