import random
import torch
from torch.autograd import Variable


class ReplayMemory(object):
    """
    Experience Replay that stores memory of past series of events
    
    -Help AI study long-term correlation
    -Helps make deep Q-learning process much better by keeping long-term memory
    
    """
    
    def __init__(self, capacity):
        """
        Initialize Experience Replay
     
        @param capacity: max number of events to keep in memory
        """
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        """
        Push new event to the memory. If it exceeds capacity, then delete the 
        oldest event
        
        event: four ele
        
        @param event: event that is consist of four elements
            (last state, new state, last action, last reward)
        """
        self.memory.append(event)
        
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        """
        Get random sample from memory.
        
        @param batch_size: size of random sample
        @return x: samples in Pytorch Variable
        """
        
        #Take random sample from memory that has batch_size
        #Reshape the list to one batch for each of the state
        #[(last states),(new states), (last actions), (last rewards)], 
        samples = zip(*random.sample(self.memory, batch_size))
        
        #Convert samples to Pytorch Variable
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
