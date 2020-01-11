'''
File name: nuralNetwork.py
    Implemetation of a Neural Network, without using external NN libraries.
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 01.12.2019

Python Version: 3.6
'''

import numpy as np

class NeuralNetwork():
    '''
    Neural Network object.

    Args:
        input_nodes (int): 
            The size of the input layer of the ANN
        hidden_nodes (int): 
            The size of the hidden layer of the ANN
        output_nodes (int): 
            The size of the output layer of the ANN
        learning_rate (float): 
            The learning rate in (0, 1]
                
    Attributes:
        input_nodes (int): 
            The size of the input layer of the ANN
        hidden_nodes (int): 
            The size of the hidden layer of the ANN
        output_nodes (int): 
            The size of the output layer of the ANN
        lr (float): 
            The learning rate in (0, 1]
        activation_function (function):
            The activation function of the ANN (sigmoid)
        weights_input_to_hidden (numpy array 2D): 
            The wights from the input to the hidden layer.
        weights_hidden_to_output (numpy array 2D): 
            The wights from the hidden to the output layer.
                                
    Methods:
        train():
            Train the network on batch of features and targets. 
        __feedForwrad():
            Implement feed forward, calculate the hidden layer and final outputs
        __backpropagation():
            Backpropagation implementation.
        __update_weights():
            Update weights on gradient descent step.
        predict:
            Run a forward pass through the network with input features and return
            a prediction.
    '''

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        '''
        Class constructor.

        Args:
            input_nodes (int): 
                The size of the input layer of the ANN
            hidden_nodes (int): 
                The size of the hidden layer of the ANN
            output_nodes (int): 
                The size of the output layer of the ANN
            learning_rate (float): 
                The learning rate in (0, 1]

        Raises:
            -

        Returns:
            -
        '''

        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # Set self.activation_function to the sigmoid
        self.activation_function = lambda x : 1/(1+np.exp(-x))
                    

    def train(self, features, targets):
        '''
        Train the network on batch of features and targets. 

        Args:
            features (numpy array 2D): 
                Each row is one data record, each column is a feature
            targets (numpy array 1D): 
                Target values
            
        Raises:
            -

        Returns:
            -
        '''
        
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.__feedForwrad(X)

            delta_weights_i_h, delta_weights_h_o = self.__backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.__update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def __feedForwrad(self, X):
        '''
        Implement feed forward, calculate the hidden layer and final outputs

        Args:
            X (numpy array 2D): 
                Each row is one data record, each column is a feature

        Raises:
            -

        Returns:
            final_outputs  (numpy array 2D): 
                Signals from final output layer
            hidden_outputs (numpy array 2D): 
                Signals from hidden layer
        '''

        # Hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)   # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer, the activation function is f(x) = x
        
        return final_outputs, hidden_outputs


    def __backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        '''
        Backpropagation implementation.

        Args:
            final_outputs: 
                output from forward pass
            y: 
                target (i.e. label) batch
            delta_weights_i_h: 
                change in weights from input to hidden layers
            delta_weights_h_o: 
                change in weights from hidden to output layers

        Raises:
            -

        Returns:
            delta_weights_i_h (float): 
                Weight step (input to hidden)
            delta_weights_h_o (float): 
                Weight step (hidden to output)
        '''

        # Output error
        # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs

        # activation function of the output layer is f(x) = x, where f'(x) = 1
        output_error_term = error * 1.

        # Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += X[:, None] * hidden_error_term

        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term[:,None] * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o


    def __update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        '''
        Update weights on gradient descent step.

        Args:
            delta_weights_i_h: 
                Change in weights from input to hidden layers
            delta_weights_h_o: 
                Change in weights from hidden to output layers
            n_records: 
                Number of records

        Raises:
            -

        Returns:
            -
        '''

        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records

        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records


    def predict(self, features):
        '''
        Run a forward pass through the network with input features and return
        a prediction.

        Args:
            features (numpy array 2D): 
                Each row is one data record, each column is a feature

        Raises:
            -

        Returns:
            prediction (numpy array 1D): 
                Predictions for the input data.
        '''

        return self.__feedForwrad(features)[0]