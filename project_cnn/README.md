# Project CNN
In this project, a Convolutional Neural Network (CNN) pipeline is built. Given an image of a dog, the model will identify an estimate of the canine’s breed. Two CNN classifiers are built. One classifier from the scratch and one using transfer learning (VGG16 reference [1]).
In this project PyTorch, an open source machine learning framework that accelerates the path from research prototyping to production deployment, is used. The training of the models is executed in a GPU (NVIDIA GeForce RTX 2060 SUPER) using CUDA (reference [2]).

## Components
The project consists of the below python files:
* **cnn.py:** Implementation of a Convolutional Neural Network for dog's breed classification, using PyTorch and CUDA. 
* **cnnTL.py:** Implementation of a Convolutional Neural Network for dog's breed classification with Transfer Learning, using PyTorch and CUDA.
* **utils.py:** Utilities script where common project functions are implemented.
* **determineBreed.py:** Application which selects 'n' random dog images ('n' is given as an input parameter) from the dataset (test images), and classifies them (estimate of the canine’s breed) using a saved trained model produced by the execution of the cnnTL.py script.

## Executing the project
### CNN from scratch
Below is the execution details of the cnn.py script. The script creates a CNN model from scratch, it trains the model using the training and validation input images, and finally it tests the trained model using the testing input images. The training is done under a predefined set of hyperparameters values came from a model complexity analysis applied during the model design (not included in the script).
The script outputs a graph showing the training and validation losses during the training execution (./graphs/cnn_train_losses.png) and saves the best performing trained model in a file (./cnn_trained_models/cnn_trained_model.pt).

```
$python cnn.py
Convolutional Neural Network implementation for dog's breed classification.
CUDA availability (True): Training on GPU.

cnn_model moved to GPU.

Prepare data loaders: batch_size = 20, image_size = (256, 256), train_data_path = ./data/dog_images/train, test_data_path = ./data/dog_images/test, validation_data_path = ./data/dog_images/valid, affine_degrees = 0, scale = (0.5, 2.0), norm_mean = [0.5, 0.5, 0.5], norm_std = [0.5, 0.5, 0.5]

Training model: max_epochs = 500, stop_criterion = 10, save_model_path = ./cnn_trained_models/cnn_trained_model.pt, save_graph_path = ./graphs/cnn_train_losses.png, cuda = True
Epoch:   1, Training Loss: 4.888078, Validation Loss: 4.874681, Duration: 66.742, Not Improvement Counter: 0. Model saved.
Epoch:   2, Training Loss: 4.859412, Validation Loss: 4.825675, Duration: 64.016, Not Improvement Counter: 0. Model saved.
...
Epoch:  49, Training Loss: 1.157245, Validation Loss: 2.939894, Duration: 62.171, Not Improvement Counter: 9
Epoch:  50, Training Loss: 1.094446, Validation Loss: 2.917616, Duration: 61.453, Not Improvement Counter: 10
Plot Training Performance: ./graphs/cnn_train_losses.png

Testing model: cuda = True
Test finished in 6.195 seconds. Testing Loss: 2.741305
Test Accuracy: 32.66% (273/836)
```  

![](./graphs/cnn_train_losses.png?raw=true)

The model's accuracy is relatively low but improved drastically comparing with the requirement of the original project (accuracy requirement was > 10%). Further improvements can be done by using a more complex CNN and by using additional transformations for the input images. These improvements will be applied in a later version of the project.

#### UPDATE 1, 12.01.2020:
Model's perfomance improved further (from ***32.66%*** to  ***46.77%***) by introducing additional data transformations like images translation and shear. The network architecture remains the same.

```
python cnn.py
Convolutional Neural Network implementation for dog's breed classification.
CUDA availability (True): Training on GPU.

cnn_model moved to GPU.

Prepare data loaders: batch_size = 20, image_size = (256, 256), train_data_path = ./data/dog_images/train, test_data_path = ./data/dog_images/test, validation_data_path = ./data/dog_images/valid, affine_degrees = 0, scale = (0.5, 2.0), norm_mean = [0.5, 0.5, 0.5], norm_std = [0.5, 0.5, 0.5]

Training model: max_epochs = 500, stop_criterion = 10, save_model_path = ./cnn_trained_models/cnn_trained_model.pt, save_graph_path = ./graphs/cnn_train_losses.png, cuda = True
Epoch:   1, Training Loss: 4.886535, Validation Loss: 4.876630, Duration: 146.193, Not Improvement Counter: 0. Model saved.
Epoch:   2, Training Loss: 4.864435, Validation Loss: 4.843610, Duration: 61.092, Not Improvement Counter: 0. Model saved.
...
Epoch: 109, Training Loss: 1.065152, Validation Loss: 2.089342, Duration: 61.016, Not Improvement Counter: 9
Epoch: 110, Training Loss: 1.088742, Validation Loss: 2.018300, Duration: 61.587, Not Improvement Counter: 10
Plot Training Performance: ./graphs/cnn_train_losses.png

Testing model: cuda = True
Test finished in 15.051 seconds. Testing Loss: 2.046723
Test Accuracy: 46.77% (391/836)
```

![](./graphs/cnn_train_losses_Update_1.png?raw=true)

Further improvements will be explored in later versions.

### CNN with Transfer Learning
Below is the execution details of the cnnTL.py script. The script creates a CNN model using transfer learning (VGG16 torchvision model, reference [1]), it trains the model using the training and validation input images, and finally it tests the trained model using the testing input images. The training is done under a predefined set of hyperparameters values came from a model complexity analysis applied during the model design (not included in the script).
The script outputs a graph showing the training and validation losses during the training execution (./graphs/cnnTL_train_losses.png) and saves the best performing trained model in a file (./cnn_trained_models/cnnTL_trained_model.pt).

```
$python cnnTL.py
Convolutional Neural Network implementation for dog's breed classification with Transfer Learning.
CUDA availability (True): Training on GPU.

cnnTL_model moved to GPU.

Prepare data loaders: batch_size = 20, image_size = (224, 224), train_data_path = ./data/dog_images/train, test_data_path = ./data/dog_images/test, validation_data_path = ./data/dog_images/valid, affine_degrees = 0, scale = None, norm_mean = [0.5, 0.5, 0.5], norm_std = [0.5, 0.5, 0.5]

Training model: max_epochs = 500, stop_criterion = 10, save_model_path = ./cnn_trained_models/cnnTL_trained_model.pt, save_graph_path = ./graphs/cnnTL_train_losses.png, cuda = True
Epoch:   1, Training Loss: 3.949235, Validation Loss: 2.603596, Duration: 58.619, Not Improvement Counter: 0. Model saved.
Epoch:   2, Training Loss: 1.969346, Validation Loss: 1.226156, Duration: 70.804, Not Improvement Counter: 0. Model saved.
...
Epoch:  27, Training Loss: 0.200302, Validation Loss: 0.510839, Duration: 70.406, Not Improvement Counter: 9
Epoch:  28, Training Loss: 0.195067, Validation Loss: 0.515491, Duration: 74.040, Not Improvement Counter: 10
Plot Training Performance: ./graphs/cnnTL_train_losses.png

Testing model: cuda = True
Test finished in 10.641 seconds. Testing Loss: 0.538141
Test Accuracy: 83.85% (701/836)
```

![](./graphs/cnnTL_train_losses.png?raw=true)

The model's accuracy is very high mainly because of the VGG16 pre-trained model use (very deep learning model). Further improvements will be explored in a later version of the project.

### Determine Breed Application
The script selects 'n' random dogs images and predicts their breed. The trained model used is the CNN TL. Because the file is > 500MB is not included in the repository, but it can be generated in advance by executing the cnnTL.py script.
The script requires the below input parameters.

```
python determineBreed.py -h
Determine dog's breed application.
usage: determineBreed.py [-h] -p images_path -n images_number -m CNN TL
                         trained_model

Determine dog's breed given an image, application.

optional arguments:
  -h, --help            show this help message and exit
  -p images_path        images path
  -n images_number      number of random test images (1 to 10)
  -m CNN TL trained_model
                        CNN TL trained model to load
```

An execution example of the script can be found below.

```
$python determineBreed.py -n 10 -p ./data/dog_images/*/*/* -m ./cnn_trained_models/cnnTL_trained_model.pt

Determine dog's breed application.
images_path = ./data/dog_images/*/*/*, number_of_images = 10, trained_model_file = ./cnn_trained_models/cnnTL_trained_model.pt
CUDA availability (True): Running on GPU.

Test file: ./data/dog_images\train\115.Papillon\Papillon_07478.jpg, Predicted Breed: Papillon, Classification Result: Correct
Test file: ./data/dog_images\train\061.English_cocker_spaniel\English_cocker_spaniel_04357.jpg, Predicted Breed: English_cocker_spaniel, Classification Result: Correct
Test file: ./data/dog_images\valid\034.Boxer\Boxer_02392.jpg, Predicted Breed: Boxer, Classification Result: Correct
Test file: ./data/dog_images\train\067.Finnish_spitz\Finnish_spitz_04656.jpg, Predicted Breed: Finnish_spitz, Classification Result: Correct
Test file: ./data/dog_images\train\082.Havanese\Havanese_05600.jpg, Predicted Breed: Havanese, Classification Result: Correct
Test file: ./data/dog_images\train\089.Irish_wolfhound\Irish_wolfhound_06026.jpg, Predicted Breed: Irish_wolfhound, Classification Result: Correct
Test file: ./data/dog_images\train\091.Japanese_chin\Japanese_chin_06198.jpg, Predicted Breed: Japanese_chin, Classification Result: Correct
Test file: ./data/dog_images\train\122.Pointer\Pointer_07837.jpg, Predicted Breed: Pointer, Classification Result: Correct
Test file: ./data/dog_images\train\064.English_toy_spaniel\English_toy_spaniel_04504.jpg, Predicted Breed: English_toy_spaniel, Classification Result: Correct
Test file: ./data/dog_images\train\002.Afghan_hound\Afghan_hound_00138.jpg, Predicted Breed: Afghan_hound, Classification Result: Correct

Prediction Accuracy: 100.00%
```

The expected accuraccy for the vast majority of the script executions is > 90%.

## Prerequisites
1. [python 3.7](https://www.python.org/downloads/release/python-376/)
2. [Matplotlib 3.1.2](https://matplotlib.org/)
3. [Numpy 1.17.4](https://numpy.org/)
4. [Pillow 6.2.1](https://pillow.readthedocs.io/en/stable/)
5. [PyTorch 1.3.1](https://pytorch.org/)

## References
1. *arXiv:1409.1556v6 [cs.CV]*
2. *[PyTorch CUDA API](https://pytorch.org/docs/stable/notes/cuda.html)*
