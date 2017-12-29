# SelfDrivingCar

Self Driving Car Program


This is the modelled version of a car. The car will move toward the goal(top-left corner or bottom right corner). After the car reaches the goal, the goal will change to other corner.


It will learn how to drive itself. The key of this program is "learn" because the car will not be program to drive in the program before hand. The car will have to figure everything out by itself. 




We use Deep Q-learning to implement this solution.


1. The state of environment will be passed to the Neural Network as vectors

2. Neural Network try to predict actions should be played

3. Return the outputs as Q-value for each of the possible actions

4. The best action will be chosen by either taking the highest Q-value or by overlaying the Softmax function


Inputs => Input layer => Hidden layer => Output layer => Outputs(Q-values) => Softmax Function(Decide to use which q-value) => output (Seleted Q-value = Action)


Training Artificial Neural Network using Stochastic Gradient

1. Randomly initialize weights to small number close to 0.

2. Input the first observation of your dataset in the input layer, each featue in one input node

3. Forward-Propagation: the neurons are activated in a way that impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicated result y'. 

4. Compare predicated result to the actual result y.  c = 1/2(y' - y)^2

5. Back-Propagation: The error is back-propagated.Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.

6. Repeat 1 to 5 After each observation.
