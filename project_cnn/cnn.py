'''
File name: cnn.py
    Implementation of a Convolutional Neural Network for dog's breed
    classification, using PyTorch and CUDA.
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 09.01.2020

Python Version: 3.7
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import utils as ut
import torch.optim as optim

class CNN(nn.Module):
    '''
    Convolutional Neural Network object. Implements a CNN of a specific 
    architecture.

    Args:
        -
                
    Attributes:
        conv* (torch.nn.Conv2d): 
            Convolutional layers
        pool (torch.nn.MaxPool2d):
            Max pool layer.
        dropout (torch.nn.Dropout):
            Dropout layer.
        bn* (nn.BatchNorm2d):
            Batch normalization layers.
        fc* (torch.nn.Linear):
            Linear layers.
                                
    Methods:
        forward():
            Executes a feed forward pass through the defined CNN. 
            Returns the output of the feed forward pass.

    '''

    def __init__(self):
        
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels =   3, out_channels =  16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels =  16, out_channels =  32, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels =  32, out_channels =  64, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels =  64, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)
        
        # Max Pool layer, between each two convolutional layers
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(p = 0.25, inplace = False)
        
        # Batch normalization layers, between each two convolutional layers
        self.bn1 = nn.BatchNorm2d(num_features =  16)
        self.bn2 = nn.BatchNorm2d(num_features =  32)
        self.bn3 = nn.BatchNorm2d(num_features =  64)
        self.bn4 = nn.BatchNorm2d(num_features = 128)
        self.bn5 = nn.BatchNorm2d(num_features = 256)
        self.bn6 = nn.BatchNorm2d(num_features = 512)
        
        # Linear layers (fully connected)
        self.fc1 = nn.Linear(in_features = 512 * 4 * 4, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 512)
        self.fc3 = nn.Linear(in_features =  512, out_features = 256)
        self.fc4 = nn.Linear(in_features =  256, out_features = 133)
                   
    
    def forward(self, x):
        '''
        Feed forward the model.
    
        Args:
            x (torch.Tensor): Input data.
                    
        Raises:
            -

        Returns:
            x (torch.Tensor): Output of the feed forward execution.
    
        '''

        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = self.bn3(self.pool(F.relu(self.conv3(x))))
        x = self.bn4(self.pool(F.relu(self.conv4(x))))
        x = self.bn5(self.pool(F.relu(self.conv5(x))))
        x = self.bn6(self.pool(F.relu(self.conv6(x))))
        
        # Flatten the input image
        x = x.view(-1, 512 * 4 * 4) 
        
        # Add dropout layer
        x = self.dropout(x)
        
        # Add first hidden layer
        x = F.relu(self.fc1(x))
        
        # Add dropout layer
        x = self.dropout(x)
        
        # Add second hidden layer
        x = F.relu(self.fc2(x))
        
        # Add dropout layer
        x = self.dropout(x)
        
        # Add third hidden layer
        x = F.relu(self.fc3(x))
        
        # Add dropout layer
        x = self.dropout(x)
        
        # Add fourth hidden layer
        x = self.fc4(x)

        return x


if __name__ == '__main__':
    
    print('Convolutional Neural Network implementation for dog\'s breed classification.')
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print('CUDA availability (', use_cuda, '): Training on ', 'GPU.' if use_cuda else 'CPU.', sep = '')
    
    # Create an instance of the CNN 
    cnn_model = CNN()
    print('\nCNN model description:\n', cnn_model)
    
    if use_cuda:
        cnn_model.cuda()
        print('cnn_model moved to GPU.')
        
    # Create the data loaders for training, validation and test
    data_loaders = ut.dataLoader(batch_size = 20, image_size = (256, 256), 
                                 train_data_path      = './data/dog_images/train', 
                                 test_data_path       = './data/dog_images/test', 
                                 validation_data_path = './data/dog_images/valid',
                                 scale = (0.5, 2.0))
    
    # Train the model
    ut.train(max_epochs = 500, stop_criterion = 10, loaders = data_loaders, model = cnn_model, 
        optimizer = optim.SGD(cnn_model.parameters(), lr = 0.01), criterion = nn.CrossEntropyLoss(), 
        cuda = use_cuda, save_model_path = './cnn_trained_models/cnn_trained_model.pt',
        save_graph_path = './graphs/cnn_train_losses.png')

    # Load the model which had the best validation accuracy
    cnn_model.load_state_dict(torch.load('./cnn_trained_models/cnn_trained_model.pt'))
    
    # Test the model
    ut.test(loaders = data_loaders, model = cnn_model, criterion = nn.CrossEntropyLoss(), cuda = use_cuda)

    