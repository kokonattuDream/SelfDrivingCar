# SelfDrivingCar

Self Driving Car Program


This is the modelled version of a car. It will learn how to drive itself. The key of this program is "learn" because the car will not be program to drive in the program before hand. The car will have to figure everything out by itself.





We use Deep Q-learning to implement this solution.

Deep Q-learning = Q-learing + Artifical Neural Network

1. The state of environment will be passed to the Neural Network as vectors

2. Neural Network try to predict actions should be played

3. Return the outputs as Q-value for each of the possible actions

4. The best action will be chosen by either taking the highest Q-value or by overlaying the Softmax function
