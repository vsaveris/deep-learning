'''
File name: utils.py
    Utilities script where common project functions are implemented.
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 09.01.2020

Python Version: 3.7
'''

import torchvision.transforms as transforms
import torch
import numpy as np
import time
from torchvision import datasets
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# cite: https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def plotTrainPerformance(train_losses, validation_losses, best_model_epoch, save_graph_path):
    '''
    Plots a graph showing the training and validation losses during network
    training. 

    Args:
        train_losses (list of floats):
            Training loss for each epoch.
        validation_losses (list of floats):
            Validation loss for each epoch.
        best_model_epoch (integer):
            Epoch where the best validation score was reached (saved model).
        save_graph_path (string):
            Path in which the graph should be saved.
        
    Raises:
        -

    Returns:
        -
    '''
    
    print('Plot Training Performance: ', save_graph_path, sep = '')
    
    # Plot training and validation loss
    plt.title('Training and Validation Loss', fontsize = 11, fontweight = 'bold')
    plt.xlabel('Epochs', fontsize = 11, fontweight = 'bold')
    plt.ylabel('Loss', fontsize = 11, fontweight = 'bold')
    plt.plot(range(1, len(train_losses) + 1), train_losses, label = 'Training loss', color = 'seagreen')
    plt.plot(range(1, len(train_losses) + 1), validation_losses, label = 'Validation loss', color = 'mediumvioletred')
    plt.axvline(x = best_model_epoch, color = 'rosybrown', label = 'Best validation loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
    plt.legend()
    plt.ylim()
    plt.tight_layout()
    plt.savefig(save_graph_path)
    plt.clf()


def train(max_epochs, stop_criterion, loaders, model, optimizer, criterion, cuda, 
    save_model_path, save_graph_path):
    '''
    Train the network on batch of features and targets. 

    Args:
        max_epochs (integer):
            Maximum epochs to train the network.
        stop_criterion (integer):
            Number of concecutive epochs in which if the validation loss
            is not improved, the training ends.
        loaders (dict of torch.utils.data.DataLoader):
            Data loaders dictionary, containing training (train), test (test) 
            and validation (valid) data.
        model (torch.nn.Module()):
            The cnn model.
        optimizer (torch.optim):
            Optimizer to be used during the training.
        criterion (torch.nn):
            Loss criterion to be used during the training and validation.
        cuda (boolean):
            If True execute the training on the GPU.
        save_model_path (string):
            Path in which the trained model should be saved.
        save_graph_path (string):
            Path in which the graph should be saved.
        
    Raises:
        -

    Returns:
        model (torch.nn.Module()): The trained model.
    '''

    print('\nTraining model: max_epochs = ', max_epochs, ', stop_criterion = ', stop_criterion,
          ', save_model_path = ', save_model_path, ', save_graph_path = ', save_graph_path,
          ', cuda = ', cuda, sep = '')
    
    # Initialize the best validation loss to inf
    best_validation_loss = np.Inf 
    
    # Keep track of losses, for creating a graph
    train_losses = [] 
    validation_losses = []
    best_model_epoch = 0
    
    # Count epochs where validation loss is not getting improved
    # When counter is equal to the stop criterion then the training ends
    not_improvement_counter = 0
    
    for epoch in range(max_epochs):
        start_time = time.time()

        # Reset loses for each epoch
        training_loss = 0.0
        validation_loss = 0.0

        # Model training
        model.train()
        for batch, (input_data, target_values) in enumerate(loaders['train']):

            if cuda:
                input_data, target_values = input_data.cuda(), target_values.cuda()
            
            optimizer.zero_grad()
            loss = criterion(model(input_data), target_values)
            loss.backward()
            optimizer.step()
            training_loss = training_loss + ((1 / (batch + 1)) * (loss.data - training_loss))
            
        # Model Validation
        model.eval()
        for batch, (input_data, target_values) in enumerate(loaders['valid']):

            if cuda:
                input_data, target_values = input_data.cuda(), target_values.cuda()

            loss = criterion(model(input_data), target_values)
            validation_loss = validation_loss + ((1 / (batch + 1)) * (loss.data - validation_loss))
       
        train_losses.append(training_loss)
        validation_losses.append(validation_loss)
        
        print('Epoch: {:3d}, Training Loss: {:.6f}, Validation Loss: {:.6f}, Duration: {:.3f}'.\
              format(epoch + 1, training_loss, validation_loss, time.time() - start_time), end = '')
        
        # Check if model performance improved (validation loss decreased) 
        if validation_loss <= best_validation_loss:         
            torch.save(model.state_dict(), save_model_path)
            best_validation_loss = validation_loss
            not_improvement_counter = 0
            best_model_epoch = epoch + 1
            print(', Not Improvement Counter: ', not_improvement_counter, '. Model saved.', sep = '')
                  
        else:
            not_improvement_counter += 1
            print(', Not Improvement Counter: ', not_improvement_counter, sep = '')
            
        if not_improvement_counter == stop_criterion:
            plotTrainPerformance(train_losses, validation_losses, best_model_epoch, save_graph_path)
            return model
            
    # Maximum epochs reached
    plotTrainPerformance(train_losses, validation_losses, best_model_epoch, save_graph_path)
    return model

   
def test(loaders, model, criterion, cuda):
    '''
    Test the network on batch of features and targets. 

    Args:
        loaders (dict of torch.utils.data.DataLoader):
            Data loaders dictionary, containing training (train), test (test) 
            and validation (valid) data.
        model (torch.nn.Module()):
            The cnn model.
        criterion (torch.nn):
            Loss criterion to be used during the training and validation.
        cuda (boolean):
            If True execute the training on the GPU.
        
    Raises:
        -

    Returns:
        -
    '''
    
    print('\nTesting model: cuda = ', cuda, sep = '')
    
    # Initializations
    testing_loss = 0.0
    correctly_classified = 0
    total_tests = 0
    start_time = time.time()

    model.eval()
    for batch, (input_data, target_values) in enumerate(loaders['test']):

        if cuda:
            input_data, target_values = input_data.cuda(), target_values.cuda()
        
        output = model(input_data)
        loss = criterion(output, target_values)
        testing_loss = testing_loss + ((1 / (batch + 1)) * (loss.data - testing_loss))
        
        # Convert output probabilities to predicted class
        predictions = output.data.max(1, keepdim = True)[1]
 
        correctly_classified += np.sum(np.squeeze(predictions.eq(target_values.data.view_as(predictions))).cpu().numpy())
        total_tests += input_data.size(0)
            
    print('Test finished in {:.3f} seconds. Testing Loss: {:.6f}'.format(time.time() - start_time, testing_loss))

    print('Test Accuracy: {:.2f}% ({}/{})'.format(100. * correctly_classified / total_tests, correctly_classified, total_tests))
        

def dataLoader(batch_size, image_size, train_data_path, test_data_path, validation_data_path,
               affine_degrees = 0, translate = None, scale = None, shear = None, 
               norm_mean = [0.50, 0.50, 0.50], norm_std = [0.50, 0.50, 0.50]):
    '''
    Create data loaders for training, validation and testing data. 

    Args:
        batch_size (integer):
            The size of each data batch.
        image_size (tuple of integers):
            The new size of the loaded images.
        train_data_path (string):
            The path of the training data.
        test_data_path (string):
            The path of the testing data. 
        validation_data_path (string):
            The path of the validation data.
        affine_degrees (integer):
            Rotation degrees to be used in the Affine Transformation.
        translate (2D tuple of floats):
            Maximum absolute fraction for horizontal and vertical translations.        
        scale (tuple floats):
            Scale to be used in the Affine Transformation.
        shear (sequence or float or int):
            Range of degrees to select from. If shear is a number, a shear parallel 
            to the x axis in the range (-shear, +shear) will be apllied. Else if shear 
            is a tuple or list of 2 values a shear parallel to the x axis in the range 
            (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 
            values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], 
            shear[3]) will be applied. Will not apply shear by default.
            cite: https://pytorch.org/docs/stable/torchvision/transforms.html
        norm_mean (3D list):
            Normalization mean.
        norm_std (3D list):
            Normalization std.
        
    Raises:
        -

    Returns:
        loaders (dict of torch.utils.data.DataLoader):
            Data loaders dictionary, containing training (train), test (test) 
            and validation (valid) data.
    '''
    
    print('\nPrepare data loaders: batch_size = ', batch_size, ', image_size = ', image_size,
          ', train_data_path = ', train_data_path, ', test_data_path = ', test_data_path, 
          ', validation_data_path = ', validation_data_path, ', affine_degrees = ', affine_degrees,
          ', translate = ', translate, ', scale = ', scale, ', shear = ', shear, 
          ', norm_mean = ', norm_mean, ', norm_std = ', norm_std, sep = '')
    
    # Images transformation 
    image_tranform = transforms.Compose([transforms.Resize(image_size), 
        transforms.RandomAffine(degrees = affine_degrees, 
            translate = translate, 
            scale = scale,
            shear = shear, 
            resample = False, 
            fillcolor = 0),
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean, std=norm_std)])

    # Datasets (train, test and validation)
    train_dataset = datasets.ImageFolder(root = train_data_path, 
        transform = image_tranform, 
        target_transform = None, 
        loader = Image.open)

    test_dataset = datasets.ImageFolder(root = test_data_path, 
        transform = image_tranform, 
        target_transform = None, 
        loader = Image.open)

    validation_dataset = datasets.ImageFolder(root = validation_data_path, 
        transform = image_tranform, 
        target_transform = None, 
        loader = Image.open)

    # Data loaders
    train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        sampler = None, 
        num_workers = 0)

    validation_data_loader = torch.utils.data.DataLoader(dataset = validation_dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        sampler = None, 
        num_workers = 0)

    test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        sampler = None, 
        num_workers = 0)

    return {'train': train_data_loader, 'valid': validation_data_loader, 'test':test_data_loader}
    
        
        