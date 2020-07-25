
# Importing the libraries
import os
import random
import torch
import torch.nn as neural_network
import torch.nn.functional as function
import torch.optim as optim
from torch.autograd import Variable

class NeuralNetwork(neural_network.Module):
    
    # Number of hidden layer neurons.
    # I chose 45 neurons, but this can be altered to explore more outcomes.
    # Warning: Higher numbers of neurons may require significant computational power.
    HIDDEN_NEURONS_COUNT = 45    
    
    # Architecture of the neural network    

    # Initialise the neural network with input signals and output actions.
    def __init__(self, no_of_input_signal, no_of_output_action):
        super(NeuralNetwork, self).__init__()

        # Input layer:
        # no_of_input_signal = 5 (3 signals, +orientation, -orientation).
        self.no_of_input_signal = no_of_input_signal
        
        # Output layer:
        # no_of_output_action = 3 (move forward, turn right, turn left)
        self.no_of_output_action = no_of_output_action
        
        # Establish full connection between input layer and hidden layer 
        # Using 'Linear' function from PyTorch.
        # Note: 5 input layer neurons and 45 hidden layer neurons.
        self.full_connection_input = neural_network.Linear(no_of_input_signal, NeuralNetwork.HIDDEN_NEURONS_COUNT)

        # Establish full connection between hidden layer and output layer 
        # Using 'Linear' function from PyTorch.
        # Note: 45 hidden layer neurons and 3 output layer neurons.
        self.full_connection_output = neural_network.Linear(NeuralNetwork.HIDDEN_NEURONS_COUNT, no_of_output_action)
    
    # Forward propogation
    # Returning the Q-values of an action  
    def forward(self, state):
        # Activate the hidden neurons.
        # RELU = rectifier function (_/) from PyTorch.
        hidden_neuron = function.relu(self.full_connection_input(state))
        output_q_values = self.full_connection_output(hidden_neuron)
        # Return the Q value for each possicle action (move forward, turn left and turn right)
        return output_q_values

# Creating Experience Memory class
class ExperienceMemory(object):

    def __init__(self, capacity):
        # Initialise the memory capacity 
        # [This will be 100000 (MEMORY_CAPACITY defined in Deep_Q_Network class)].
        self.capacity = capacity
        # Initialise the memory list.
        self.memory = []
    
    def memorise(self, event):
        # Add the event to the memory.
        self.memory.append(event)
        # If the memory goes beyond the capacity, then remove the oldest event in memory.
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, sample_size):
        # Get random samples from the robot's memory.
        # The sample size is 100.
        # The zip function re-shapes the events in the memory.
        # This groups the actions states and rewards.
        samples = zip(*random.sample(self.memory, sample_size))
        # Map the samples to PyTorch variables that contain tensors and gradients.
        return map(lambda sample: Variable(torch.cat(sample, 0)), samples)

# Creating Deep Q Learning Network.
class Deep_Q_Network():
    
    # Initialise the class variables.
    MEMORY_CAPACITY = 100000
    LEARNING_RATE = 0.003
    SOFTMAX_TEMPERATURE = 1000

    def __init__(self, no_of_input_signal, no_of_output_action, discount):
        
        self.softmax_temperature = Deep_Q_Network.SOFTMAX_TEMPERATURE
        
        # discount: the gamma factor in the Bellman equation.
        self.discount = discount
        
        # Sliding window of the mean of the last 100 rewards 
        # Which is used to evaluate the evolution of the neural network's performance.
        self.rewards_over_time = []

        # no_of_input_signal: 5 input signals (3 signals, +orientation, -orientation).
        # no_of_output_action: 3 output actions (move forward, turn right, turn left).
        self.network_model = NeuralNetwork(no_of_input_signal, no_of_output_action)
        
        # Initialising the memory (an instance of the ExperienceMemory class).
        self.memory = ExperienceMemory(Deep_Q_Network.MEMORY_CAPACITY)
        
        # The optimiser performs stochastic gradient descent. 
        # optim is the PyTorch fuction to do stochastic gradient descent.
        # This optimiser is an object of the Adam optimiser.
        self.optimiser = optim.Adam(self.network_model.parameters(), lr = Deep_Q_Network.LEARNING_RATE)

         # Initialising the previous action.
        self.previous_action = 0

        # Initialising the previous reward.
        self.previous_reward = 0
       
        # Initialising the previous state of the robot with a PyTorch tensor.
        self.previous_state = torch.Tensor(no_of_input_signal).unsqueeze(0)
    
    def select_action(self, input_state):
        
        # Softmax Tempaerature is the measure of how sure the AI is in its decisions.
        # Generate a probability for an action.
        # This probability helps the robot to explore.
        probability = function.softmax(self.network_model(Variable(input_state, volatile = True)) * self.softmax_temperature)
        
        # Taking a random draw but according to the probabilities. 
        # The lower probabilites are unlikely to be chosen but 
        # the higher ones are likely to be chosen.
        selected_action = probability.multinomial()
        return selected_action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        # Get the output of the batch state and get the action out of the state.
        outputs = self.network_model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        # Get the maximum of next output batch.
        next_outputs = self.network_model(batch_next_state).detach().max(1)[0]

        # Target is the reward plus the discounted next output.
        target = batch_reward + next_outputs * self.discount
        
        # Calculate the temporal difference / loss.
        temporal_difference_loss = function.smooth_l1_loss(outputs, target)
        
        # Re-initialise the optimiser at each iteration of the loop.
        self.optimiser.zero_grad()
        
        # Perform backward propagation, using the backward function.
        # Set retain_variables to True to retain / free some memory and 
        # to improve the backward propagation.
        temporal_difference_loss.backward(retain_variables = True)
        
        # Update the weights after back propagation.
        self.optimiser.step()
    
    def update(self, current_reward, current_signal):

        # Get the current state with the current sensor signals.
        current_state = torch.Tensor(current_signal).float().unsqueeze(0)
        
        # Push the transition event into the memory.
        self.memory.memorise((self.previous_state, current_state, torch.LongTensor([int(self.previous_action)]), torch.Tensor([self.previous_reward])))
        
        # Select the action to play.
        current_action = self.select_action(current_state)
        
        # Making the AI learn from its action.
        # Note: self.memory is the object of the ExperienceMemory class 
        # which has an attribute called memory which is a list. 
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        # Updating the action, state and reward.
        self.previous_action = current_action
        self.previous_state = current_state
        self.previous_reward = current_reward
        
        # Add the current reward to the list of 1000 rewards over time.
        self.rewards_over_time.append(current_reward)
        
        # If the rewards_over_time goes over 1000, remove or pop the oldest reward.
        if len(self.rewards_over_time) > 1000:
            del self.rewards_over_time[0]
        
        # Return the current action.
        return current_action
    
    def score(self):
        # Calculate the mean of the rewards.
        # In order to avoid the division by zero error, I added + 1. float in the denominator.
        return sum(self.rewards_over_time) / (len(self.rewards_over_time) + 1.)
    
    def save(self):
        # Save the neural network model and the optimiser.
        torch.save(
            {
               'neural_network_model': self.network_model.state_dict(),
               'optimiser' : self.optimiser.state_dict(),
            }, 'MecClearen_Brain.pth'
        )
    
    def load(self):
        # Load the saved brain file.
        if os.path.isfile('MecClearen_Brain.pth'):
            print("Loading brain ...")
            loading_brain = torch.load('MecClearen_Brain.pth')
            # Load the neural network model.
            self.network_model.load_state_dict(loading_brain['neural_network_model'])
            # Load the optimiser.
            self.optimiser.load_state_dict(loading_brain['optimiser'])
            print("Saved brain loaded successfully!")
        else:
            print("Sorry, no brain found to load!")