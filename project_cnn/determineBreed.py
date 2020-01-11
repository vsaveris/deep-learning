'''
File name: determineBreed.py
    Application which selects 'n' random dog images ('n' is given as an input
    parameter) from the dataset (test images), and classifies them (estimate of 
    the canine’s breed) using a saved trained model produced by the execution of 
    the cnn.py or cnnTL.py scripts. The model to be used is given as an input
    parameter.
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 11.01.2020

Python Version: 3.7
'''

if __name__ == '__main__':
    
    print('Determine dog\'s breed application.')
    
    # Read input parameters
    try:
        images_path, number_of_images, trained_model = getInputParameters()
    except:
        print('')
        
    # Load the test images
    

#Step 5: Write your Algorithm
#Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. Then,

#if a dog is detected in the image, return the predicted breed.
#if a human is detected in the image, return the resembling dog breed.
#if neither is detected in the image, provide output that indicates an error.
#You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the face_detector and human_detector functions developed above. You are required to use your CNN from Step 4 to predict dog breed.

#Some sample output for our algorithm is provided below, but feel free to design your own user experience!
import numpy as np
from glob import glob
import cv2
from torchvision import datasets
import torchvision.transforms as transforms
human_files = np.array(glob("./data/lfw/*/*"))
dog_files = np.array(glob("./data/dog_images/*/*/*"))
from PIL import Image
import torch
use_cuda = False #torch.cuda.is_available()
import cv2                
import matplotlib.pyplot as plt    
import torchvision.models as models
import torch.nn as nn
# define VGG16 model
VGG16 = models.vgg16(pretrained=True)
# list of class names by index, i.e. a name can be accessed like class_names[0]
# Datasets (train, test and validation)
image_tranform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
                                         transforms.Normalize(mean=[0.50, 0.50, 0.50], std=[0.50, 0.50, 0.50])])
train_dataset = datasets.ImageFolder(root = './data/dog_images/train', 
                                     transform = image_tranform, 
                                     target_transform = None, loader = Image.open)
class_names = [item[4:].replace("_", " ") for item in train_dataset.classes]

model_transfer = models.vgg16(pretrained = True)           
# Change the last linear layer to match the classification problem
output_layer = nn.Linear(model_transfer.classifier[6].in_features, 133)
model_transfer.classifier[6] = output_layer
print(model_transfer)         
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
 
from PIL import Image

import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    # Create the tranforms required for the model, as defined in PyTorch http://pytorch.org/docs/stable/torchvision/models.html
    image_tranform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Load the image and apply the tranforms in one step
    image_tensor = image_tranform(Image.open(img_path))
    if use_cuda:
        image_tensor = image_tensor.to('cuda')
    
    # Put the model in evaluation mode (no training for the given images)
    VGG16.eval()
    
    # Predict the input image (use unsqeeze for adding one more dimension since the model expects a batch)
    return VGG16(image_tensor.unsqueeze(0)).argmax()  # predicted class index


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    
    category = VGG16_predict(img_path)
    
    return True if category >= 151 and category <= 268 else False

 
def predict_breed_transfer(img_path):
    
    image_tranform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
                                         transforms.Normalize(mean=[0.50, 0.50, 0.50], std=[0.50, 0.50, 0.50])])
    
    image_tensor = image_tranform(Image.open(img_path))
    if use_cuda:
        image_tensor = image_tensor.to('cuda')
    
    # Put the model in evaluation mode
    model_transfer.eval()
    
    # Predict the input image (use unsqeeze for adding one more dimension since the model expects a batch)
    return class_names[model_transfer(image_tensor.unsqueeze(0)).argmax()]
    
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])

# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def run_app(img_path):
    
    import cv2
    import matplotlib.pyplot as plt                        

    
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.show()

    if face_detector(img_path):
        print('Image loaded is from a human.')
        print('If you were a dog, you would be a: ', predict_breed_transfer(img_path))
    elif dog_detector(img_path):
        print('Image loaded is from a dog.')
        print('The predicted breed is: ', predict_breed_transfer(img_path))
    else:
        print('Error! The loaded image is not from a human neither from a dog.')
        
## Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

import random

for _ in range(10):
    run_app(human_files[random.randint(0,13232)])
    
for _ in range(10):
    run_app(dog_files[random.randint(0,8350)])
    
