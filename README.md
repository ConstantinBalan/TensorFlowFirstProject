# TensorFlowFirstProject
## Practicing with the Python TensorFlow library and Keras API

This was mostly done just by following the documentation on the TensorFlow and Keras websites.
Imports the fashion MNIST dataset(This prgoram skips over implementing an input function due to the data already being organized in a dataset). Creates a one layer sequential neural network, then trains it using an adam optimizer and a Sparse Categorical Crossentropy loss fucntion to determine the accuracy of the model. Then it passes the model through *n* number of epochs, and prints the accuracy of each. Finally, it compares the training accuracy to the test accuracy. This will allow you to determine if there is any overfitting with the model
