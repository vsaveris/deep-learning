# Project CNN
In this project, a Convolutional Neural Network (CNN) pipeline is built. Given an image of a dog, the model will identify an estimate of the canine’s breed. Two CNN classifiers are built. One classifier from the scratch and one using transfer learning (VGG16 reference [1]).
In this project PyTorch, an open source machine learning framework that accelerates the path from research prototyping to production deployment, is used. The training of the models is executed in a GPU (NVIDIA GeForce RTX 2060 SUPER) using CUDA (reference [2]).

## Components
The project consists of the below python files:
* **cnn.py:** Implementation of a Convolutional Neural Network for dog's breed classification, using PyTorch and CUDA. 
* **cnnTL.py:** Implementation of a Convolutional Neural Network for dog's breed classification with Transfer Learning, using PyTorch and CUDA.
* **utils.py:** Utilities script where common project functions are implemented.
* **determineBreed.py:** Application which selects 'n' random dog images ('n' is given as an input parameter) from the dataset (test images), and classifies them (estimate of the canine’s breed) using a saved trained model produced by the execution of the cnn.py or cnnTL.py scripts.

## Executing the project
### CNN from scratch
```
$python cnn.py
Convolutional Neural Network implementation for dog's breed classification.
CUDA availability (True): Training on GPU.

CNN model description:
 CNN(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv6): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.25, inplace=False)
  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn6): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=8192, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=256, bias=True)
  (fc4): Linear(in_features=256, out_features=133, bias=True)
)
cnn_model moved to GPU.

Prepare data loaders: batch_size = 20, image_size = (256, 256), train_data_path = ./data/dog_images/train, test_data_path = ./data/dog_images/test, validation_data_path = ./data/dog_images/valid, affine_degrees = 0, scale = (0.5, 2.0), norm_mean = [0.5, 0.5, 0.5], norm_std = [0.5, 0.5, 0.5]

Training model: max_epochs = 500, stop_criterion = 10, save_model_path = ./cnn_trained_models/cnn_trained_model.pt, save_graph_path = ./graphs/cnn_train_losses.png, cuda = True
Epoch:   1, Training Loss: 4.888078, Validation Loss: 4.874681, Duration: 66.742, Not Improvement Counter: 0. Model saved.
Epoch:   2, Training Loss: 4.859412, Validation Loss: 4.825675, Duration: 64.016, Not Improvement Counter: 0. Model saved.
Epoch:   3, Training Loss: 4.759161, Validation Loss: 4.646235, Duration: 64.975, Not Improvement Counter: 0. Model saved.
Epoch:   4, Training Loss: 4.569113, Validation Loss: 4.436183, Duration: 64.025, Not Improvement Counter: 0. Model saved.
Epoch:   5, Training Loss: 4.396616, Validation Loss: 4.263655, Duration: 64.077, Not Improvement Counter: 0. Model saved.
Epoch:   6, Training Loss: 4.260268, Validation Loss: 4.207516, Duration: 63.811, Not Improvement Counter: 0. Model saved.
Epoch:   7, Training Loss: 4.133730, Validation Loss: 4.168678, Duration: 64.020, Not Improvement Counter: 0. Model saved.
Epoch:   8, Training Loss: 4.034443, Validation Loss: 3.986879, Duration: 63.673, Not Improvement Counter: 0. Model saved.
Epoch:   9, Training Loss: 3.938194, Validation Loss: 3.875120, Duration: 63.912, Not Improvement Counter: 0. Model saved.
Epoch:  10, Training Loss: 3.852243, Validation Loss: 3.850490, Duration: 63.634, Not Improvement Counter: 0. Model saved.
Epoch:  11, Training Loss: 3.760509, Validation Loss: 3.738414, Duration: 63.988, Not Improvement Counter: 0. Model saved.
Epoch:  12, Training Loss: 3.684166, Validation Loss: 3.687783, Duration: 63.555, Not Improvement Counter: 0. Model saved.
Epoch:  13, Training Loss: 3.595892, Validation Loss: 3.712724, Duration: 63.493, Not Improvement Counter: 1
Epoch:  14, Training Loss: 3.485414, Validation Loss: 3.599949, Duration: 63.921, Not Improvement Counter: 0. Model saved.
Epoch:  15, Training Loss: 3.421779, Validation Loss: 3.560970, Duration: 64.178, Not Improvement Counter: 0. Model saved.
Epoch:  16, Training Loss: 3.359030, Validation Loss: 3.585441, Duration: 63.762, Not Improvement Counter: 1
Epoch:  17, Training Loss: 3.277555, Validation Loss: 3.496234, Duration: 63.907, Not Improvement Counter: 0. Model saved.
Epoch:  18, Training Loss: 3.198224, Validation Loss: 3.341575, Duration: 64.115, Not Improvement Counter: 0. Model saved.
Epoch:  19, Training Loss: 3.121265, Validation Loss: 3.403527, Duration: 63.780, Not Improvement Counter: 1
Epoch:  20, Training Loss: 3.049275, Validation Loss: 3.274054, Duration: 64.065, Not Improvement Counter: 0. Model saved.
Epoch:  21, Training Loss: 2.982371, Validation Loss: 3.352093, Duration: 63.958, Not Improvement Counter: 1
Epoch:  22, Training Loss: 2.885857, Validation Loss: 3.224485, Duration: 63.734, Not Improvement Counter: 0. Model saved.
Epoch:  23, Training Loss: 2.819812, Validation Loss: 3.185515, Duration: 63.858, Not Improvement Counter: 0. Model saved.
Epoch:  24, Training Loss: 2.741909, Validation Loss: 3.253900, Duration: 64.363, Not Improvement Counter: 1
Epoch:  25, Training Loss: 2.660525, Validation Loss: 3.037000, Duration: 61.515, Not Improvement Counter: 0. Model saved.
Epoch:  26, Training Loss: 2.586457, Validation Loss: 3.141257, Duration: 60.655, Not Improvement Counter: 1
Epoch:  27, Training Loss: 2.501433, Validation Loss: 3.048705, Duration: 60.865, Not Improvement Counter: 2
Epoch:  28, Training Loss: 2.421862, Validation Loss: 3.056936, Duration: 60.925, Not Improvement Counter: 3
Epoch:  29, Training Loss: 2.362612, Validation Loss: 2.901979, Duration: 60.877, Not Improvement Counter: 0. Model saved.
Epoch:  30, Training Loss: 2.300475, Validation Loss: 3.013498, Duration: 60.672, Not Improvement Counter: 1
Epoch:  31, Training Loss: 2.205402, Validation Loss: 3.050148, Duration: 60.891, Not Improvement Counter: 2
Epoch:  32, Training Loss: 2.153845, Validation Loss: 2.989233, Duration: 60.504, Not Improvement Counter: 3
Epoch:  33, Training Loss: 2.058978, Validation Loss: 3.006618, Duration: 60.211, Not Improvement Counter: 4
Epoch:  34, Training Loss: 2.012426, Validation Loss: 2.866553, Duration: 60.625, Not Improvement Counter: 0. Model saved.
Epoch:  35, Training Loss: 1.939273, Validation Loss: 2.901061, Duration: 61.426, Not Improvement Counter: 1
Epoch:  36, Training Loss: 1.894807, Validation Loss: 2.871165, Duration: 60.588, Not Improvement Counter: 2
Epoch:  37, Training Loss: 1.826866, Validation Loss: 2.879071, Duration: 60.567, Not Improvement Counter: 3
Epoch:  38, Training Loss: 1.762301, Validation Loss: 2.875085, Duration: 60.573, Not Improvement Counter: 4
Epoch:  39, Training Loss: 1.704604, Validation Loss: 3.057007, Duration: 60.764, Not Improvement Counter: 5
Epoch:  40, Training Loss: 1.628982, Validation Loss: 2.744302, Duration: 61.788, Not Improvement Counter: 0. Model saved.
Epoch:  41, Training Loss: 1.549830, Validation Loss: 3.076204, Duration: 61.415, Not Improvement Counter: 1
Epoch:  42, Training Loss: 1.526312, Validation Loss: 2.975544, Duration: 62.057, Not Improvement Counter: 2
Epoch:  43, Training Loss: 1.487823, Validation Loss: 2.925201, Duration: 62.364, Not Improvement Counter: 3
Epoch:  44, Training Loss: 1.433959, Validation Loss: 2.856759, Duration: 61.991, Not Improvement Counter: 4
Epoch:  45, Training Loss: 1.358746, Validation Loss: 2.920357, Duration: 61.834, Not Improvement Counter: 5
Epoch:  46, Training Loss: 1.311032, Validation Loss: 2.810886, Duration: 61.950, Not Improvement Counter: 6
Epoch:  47, Training Loss: 1.266692, Validation Loss: 2.901157, Duration: 62.095, Not Improvement Counter: 7
Epoch:  48, Training Loss: 1.209524, Validation Loss: 2.911021, Duration: 62.389, Not Improvement Counter: 8
Epoch:  49, Training Loss: 1.157245, Validation Loss: 2.939894, Duration: 62.171, Not Improvement Counter: 9
Epoch:  50, Training Loss: 1.094446, Validation Loss: 2.917616, Duration: 61.453, Not Improvement Counter: 10
Plot Training Performance: ./graphs/cnn_train_losses.png

Testing model: cuda = True
Test finished in 6.195 seconds. Testing Loss: 2.741305
Test Accuracy: 32.66% (273/836)
``` 

![](./graphs/cnn_train_losses.png?raw=true)

### CNN with Transfer Learning
```
$python cnnTL.py
Convolutional Neural Network implementation for dog's breed classification with Transfer Learning.
CUDA availability (True): Training on GPU.

Pre-trained CNN model description:
 VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

Modified pre-trained CNN model description:
 VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=133, bias=True)
  )
)
cnnTL_model moved to GPU.

Prepare data loaders: batch_size = 20, image_size = (224, 224), train_data_path = ./data/dog_images/train, test_data_path = ./data/dog_images/test, validation_data_path = ./data/dog_images/valid, affine_degrees = 0, scale = None, norm_mean = [0.5, 0.5, 0.5], norm_std = [0.5, 0.5, 0.5]

Training model: max_epochs = 500, stop_criterion = 10, save_model_path = ./cnn_trained_models/cnnTL_trained_model.pt, save_graph_path = ./graphs/cnnTL_train_losses.png, cuda = True
Epoch:   1, Training Loss: 3.949235, Validation Loss: 2.603596, Duration: 58.619, Not Improvement Counter: 0. Model saved.
Epoch:   2, Training Loss: 1.969346, Validation Loss: 1.226156, Duration: 70.804, Not Improvement Counter: 0. Model saved.
Epoch:   3, Training Loss: 1.220818, Validation Loss: 0.880295, Duration: 72.098, Not Improvement Counter: 0. Model saved.
Epoch:   4, Training Loss: 0.948121, Validation Loss: 0.739325, Duration: 75.547, Not Improvement Counter: 0. Model saved.
Epoch:   5, Training Loss: 0.799986, Validation Loss: 0.676853, Duration: 72.477, Not Improvement Counter: 0. Model saved.
Epoch:   6, Training Loss: 0.697763, Validation Loss: 0.630785, Duration: 72.234, Not Improvement Counter: 0. Model saved.
Epoch:   7, Training Loss: 0.618199, Validation Loss: 0.596388, Duration: 73.109, Not Improvement Counter: 0. Model saved.
Epoch:   8, Training Loss: 0.569979, Validation Loss: 0.580781, Duration: 71.625, Not Improvement Counter: 0. Model saved.
Epoch:   9, Training Loss: 0.523631, Validation Loss: 0.567822, Duration: 68.531, Not Improvement Counter: 0. Model saved.
Epoch:  10, Training Loss: 0.489073, Validation Loss: 0.566467, Duration: 75.297, Not Improvement Counter: 0. Model saved.
Epoch:  11, Training Loss: 0.457797, Validation Loss: 0.541855, Duration: 72.862, Not Improvement Counter: 0. Model saved.
Epoch:  12, Training Loss: 0.425600, Validation Loss: 0.533512, Duration: 68.985, Not Improvement Counter: 0. Model saved.
Epoch:  13, Training Loss: 0.403577, Validation Loss: 0.523998, Duration: 74.909, Not Improvement Counter: 0. Model saved.
Epoch:  14, Training Loss: 0.376547, Validation Loss: 0.525483, Duration: 75.656, Not Improvement Counter: 1
Epoch:  15, Training Loss: 0.353527, Validation Loss: 0.512719, Duration: 71.844, Not Improvement Counter: 0. Model saved.
Epoch:  16, Training Loss: 0.340287, Validation Loss: 0.512059, Duration: 72.594, Not Improvement Counter: 0. Model saved.
Epoch:  17, Training Loss: 0.319083, Validation Loss: 0.517960, Duration: 74.141, Not Improvement Counter: 1
Epoch:  18, Training Loss: 0.308840, Validation Loss: 0.502366, Duration: 75.156, Not Improvement Counter: 0. Model saved.
Epoch:  19, Training Loss: 0.285639, Validation Loss: 0.508719, Duration: 75.266, Not Improvement Counter: 1
Epoch:  20, Training Loss: 0.276967, Validation Loss: 0.517260, Duration: 75.828, Not Improvement Counter: 2
Epoch:  21, Training Loss: 0.258572, Validation Loss: 0.506284, Duration: 72.889, Not Improvement Counter: 3
Epoch:  22, Training Loss: 0.239888, Validation Loss: 0.516426, Duration: 72.063, Not Improvement Counter: 4
Epoch:  23, Training Loss: 0.236688, Validation Loss: 0.517926, Duration: 72.531, Not Improvement Counter: 5
Epoch:  24, Training Loss: 0.228411, Validation Loss: 0.503276, Duration: 71.016, Not Improvement Counter: 6
Epoch:  25, Training Loss: 0.225552, Validation Loss: 0.512144, Duration: 70.688, Not Improvement Counter: 7
Epoch:  26, Training Loss: 0.210367, Validation Loss: 0.505074, Duration: 75.031, Not Improvement Counter: 8
Epoch:  27, Training Loss: 0.200302, Validation Loss: 0.510839, Duration: 70.406, Not Improvement Counter: 9
Epoch:  28, Training Loss: 0.195067, Validation Loss: 0.515491, Duration: 74.040, Not Improvement Counter: 10
Plot Training Performance: ./graphs/cnnTL_train_losses.png

Testing model: cuda = True
Test finished in 10.641 seconds. Testing Loss: 0.538141
Test Accuracy: 83.85% (701/836)
```

![](./graphs/cnnTL_train_losses.png?raw=true)

### Determine Breed Application
```

```

## Prerequisites
1. [python 3.7](https://www.python.org/downloads/release/python-376/)
2. [Matplotlib 3.1.2](https://matplotlib.org/)
3. [Numpy 1.17.4](https://numpy.org/)
4. [Pillow 6.2.1](https://pillow.readthedocs.io/en/stable/)
5. [PyTorch 1.3.1](https://pytorch.org/)

## References
1. *arXiv:1409.1556v6 [cs.CV]*
2. *[PyTorch CUDA API](https://pytorch.org/docs/stable/notes/cuda.html)*
