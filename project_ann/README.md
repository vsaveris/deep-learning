# Project ANN
In this project, a neural network is built from scratch to carry out a prediction problem on the Bike Sharing Dataset (reference [1]). By building a neural network from the ground up, we can have a much better understanding of gradient descent, backpropagation, and other concepts that are important to know before we move to higher level tools such as PyTorch.

## Components
The project consists of two python files:
```
o neuralNetwork.py: Implements an Artificial Neural Network object of a single hidden layer.
o predictingBikeSharingData.py: Implements the project flow (load and process input data, split data in training, validation and test sets, train an ANN, test the trained ANN) 
``` 
The input data are in a *.csv file format and are stored in the `./bike_sharing_dataset` directory of the project folder.

## Executing the project
```
$python predictingBikeSharingData.py

Project Predicting bike sharing data using an ANN
Progress: 100.0% ... Training loss: 0.073 ... Validation loss: 0.149
``` 

During the execution the below graphs are shown:

### Training and Validation loss during all the training iterations
![](/images/train_validation_loss.png?raw=true)

### Predictions vs Actuals benchmark on the trained ANN
![](/images/prediction_vs_actuals.png?raw=true)

## Prerequisites
1. [python 3.6](https://www.python.org/downloads/release/python-369/)
2. [Numpy](https://numpy.org/)
3. [Pandas](https://pandas.pydata.org/)
4. [Matplotlib](https://matplotlib.org/)

## References
1. *[1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.*