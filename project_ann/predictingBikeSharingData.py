'''
File name: predictingBikeSharingData.py
    Udacity Deep Learning Nanodegree: Implementation of the 'Predicting bike 
    sharing data' project.
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 01.12.2019

Python Version: 3.6
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neuralNetwork as nn
import sys

def MSE(prediction, actual):
    '''
    Calculates mean squared error for a prediction and actual value.

    Args:
        prediction (numpy array 1D):
            Array with predictions.
        actual (numpy array 1D):
            Array with actual values.

    Raises:
        -

    Returns:
        mse (float):
            The mean squared error.    
    '''

    return np.mean((prediction - actual)**2)


def processData(input_file, verbose = False):
    '''
    Load and process the input data.

    Args:
        input_file (string):
            The csv file of the input data.
        verbose (boolean):
            If True, printing services are enabled.

    Raises:
        -

    Returns:
        data (pandas dataframe):
            Processed data.
        orig_data (pandas dataframe):
            Original data.
        scaled_features (dictionary):
            Scaled features mean and std
    '''

    # Load and prepare the data
    data = pd.read_csv(input_file)
    
    orig_data = data

    if verbose:
        print('Example of loaded data:\n', data.head())

    # Process data
    # Categorical to binary variables
    for field in ['season', 'weathersit', 'mnth', 'hr', 'weekday']:
        dummies = pd.get_dummies(data[field], prefix = field, drop_first = False)
        data = pd.concat([data, dummies], axis=1)

    # Drop unused fields
    data = data.drop(['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 
                      'workingday', 'hr'], axis = 1)

    if verbose:
        print('\nProcessed data:\n', data.head())

    # Standardize each of the continuous variables
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}

    for field in ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']:
        mean, std = data[field].mean(), data[field].std()
        scaled_features[field] = [mean, std]
        data.loc[:, field] = (data[field] - mean)/std

    if verbose:
        print('\nScaled data:\n', data.head())
        
    return data, orig_data, scaled_features


def splitData(data):
    '''
    Split the data into training, testing, and validation sets

    Args:
        data (pandas dataframe):
            Data to split.

    Raises:
        -

    Returns:
        train_features (pandas dataframe):
            Train features values.
        train_targets (pandas dataframe):
            Train labels.
        val_features (pandas dataframe):
            Validation features values.
        val_targets (pandas dataframe):
            Validation labels.
        test_features (pandas dataframe):
            Test features values.
        test_targets (pandas dataframe):
            Test labels.  
        test_data (pandas dataframe):
            Test data (features and labels).
    '''

    # Save data for approximately the last 21 days 
    test_data = data[-21*24:]

    # Now remove the test data from the data set 
    data = data[:-21*24]

    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    # Hold out the last 60 days or so of the remaining data as a validation set
    train_features, train_targets = features[:-60*24], targets[:-60*24]
    val_features, val_targets = features[-60*24:], targets[-60*24:]

    return train_features, train_targets, val_features, val_targets, test_features, test_targets, test_data


def trainANN(train_features, train_targets, val_features, val_targets):
    '''
    Train an ANN

    Args:
        train_features (pandas dataframe):
            Train features values.
        train_targets (pandas dataframe):
            Train labels.
        val_features (pandas dataframe):
            Validation features values.
        val_targets (pandas dataframe):
            Validation labels.

    Raises:
        -

    Returns:
        network (NeuralNetwork object):
            The trained neural network. 
    '''
    
    N_i = train_features.shape[1]
    network = nn.NeuralNetwork(N_i, hidden_nodes = 13, output_nodes = 1, learning_rate = 1.0)

    losses = {'train':[], 'validation':[]}
    
    for ii in range(4000):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['cnt']
                                
        network.train(X, y)
        
        # Printing out the training progress
        train_loss = MSE(network.predict(train_features).T, train_targets['cnt'].values)
        val_loss = MSE(network.predict(val_features).T, val_targets['cnt'].values)
        sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(4000)) + \
            "% ... Training loss: " + str(train_loss)[:5] + \
            " ... Validation loss: " + str(val_loss)[:5])
        sys.stdout.flush()
        
        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)
    
    # Plot training and validation loss
    plt.title('Training and Validation Loss', fontsize = 12, fontweight = 'bold')
    plt.xlabel('Iterations', fontsize = 12, fontweight = 'bold')
    plt.ylabel('Loss', fontsize = 12, fontweight = 'bold')
    plt.plot(losses['train'], label = 'Training loss', color = 'rosybrown')
    plt.plot(losses['validation'], label = 'Validation loss', color = 'brown')
    plt.legend()
    plt.ylim()
    plt.tight_layout()
    plt.show()
    
    return network


def testANN(network, orig_data, scaled_features, test_features, test_targets, test_data):
    '''
    Test a trained ANN

    Args:
        network (NeuralNetwork object):
            A trained neural network.
        orig_data (pandas dataframe):
            Original data.
        scaled_features (dictionary):
            Scaled features mean and std
        test_features (pandas dataframe):
            Test features values.
        test_targets (pandas dataframe):
            Test labels.
        test_data (pandas dataframe):
            Test data (features and labels).

    Raises:
        -

    Returns:
        -    
    '''

    fig, ax = plt.subplots(figsize=(8,4))
    
    mean, std = scaled_features['cnt']
    predictions = network.predict(test_features).T*std + mean
    ax.plot(predictions[0], label = 'Prediction', color = 'brown')
    ax.plot((test_targets['cnt']*std + mean).values, label = 'Actuals', color = 'rosybrown')
    ax.set_xlim(right=len(predictions))
    ax.legend()
    
    dates = pd.to_datetime(orig_data.iloc[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    ax.set_xticklabels(dates[12::24], rotation=45)
    
    plt.title('Prediction vs Actuals on the Test Data', fontsize = 12, fontweight = 'bold')
    plt.xlabel('Date', fontsize = 12, fontweight = 'bold')
    plt.ylabel('Total Rental Bikes', fontsize = 12, fontweight = 'bold')
    plt.tight_layout()
    plt.show()
    

# Execute project flow
print('Project Predicting bike sharing data using an ANN')

# Load and process the input data.
data, orig_data, scaled_features = processData(input_file = './bike_sharing_dataset/hour.csv', verbose = False)

# Split the data into training, testing, and validation sets
train_features, train_targets, val_features, val_targets, test_features, test_targets, test_data = splitData(data)

# Train an ANN
network = trainANN(train_features, train_targets, val_features, val_targets)

# Test ANN
testANN(network, orig_data, scaled_features, test_features, test_targets, test_data)
