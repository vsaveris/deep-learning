'''
File name: cnnTL.py
    Implementation of a Convolutional Neural Network for dog's breed
    classification with Transfer Learning, using PyTorch and CUDA.
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 09.01.2020

Python Version: 3.7
'''

import torch.nn as nn
import torch
import utils as ut
import torch.optim as optim
import torchvision.models as models
  
print('Convolutional Neural Network implementation for dog\'s breed classification with Transfer Learning.')

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
print('CUDA availability (', use_cuda, '): Training on ', 'GPU.' if use_cuda else 'CPU.', sep = '')

# Load pre-trained model
cnnTL_model = models.vgg16(pretrained = True)
print('\nPre-trained CNN model description:\n', cnnTL_model)

# Disable training for the pre-trained layers
for param in cnnTL_model.features.parameters():
    param.requires_grad = False

# Change the last linear layer to match the classification problem
cnnTL_model.classifier[6] = nn.Linear(cnnTL_model.classifier[6].in_features, 133)
print('\nModified pre-trained CNN model description:\n', cnnTL_model)

if use_cuda:
    cnnTL_model.cuda()
    print('cnnTL_model moved to GPU.')
    
# Create the data loaders for training, validation and test
data_loaders = ut.dataLoader(batch_size = 20, image_size = (224, 224), 
                             train_data_path      = './data/dog_images/train', 
                             test_data_path       = './data/dog_images/test', 
                             validation_data_path = './data/dog_images/valid')

# Train the model
ut.train(max_epochs = 500, stop_criterion = 10, loaders = data_loaders, model = cnnTL_model, 
    optimizer = optim.SGD(cnnTL_model.classifier.parameters(), lr = 0.001), 
    criterion = nn.CrossEntropyLoss(), cuda = use_cuda, 
    save_model_path = './cnn_trained_models/cnnTL_trained_model.pt',
    save_graph_path = './graphs/cnnTL_train_losses.png')

# Load the model which had the best validation accuracy
cnnTL_model.load_state_dict(torch.load('./cnn_trained_models/cnnTL_trained_model.pt'))

# Test the model
ut.test(loaders = data_loaders, model = cnnTL_model, criterion = nn.CrossEntropyLoss(), cuda = use_cuda)