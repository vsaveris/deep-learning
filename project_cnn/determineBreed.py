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

import argparse, os, glob, random
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Constants
C_CLASS_NAMES = ['Affenpinscher','Afghan_hound','Airedale_terrier','Akita',
    'Alaskan_malamute','American_eskimo_dog','American_foxhound',
    'American_staffordshire_terrier','American_water_spaniel','Anatolian_shepherd_dog',
    'Australian_cattle_dog','Australian_shepherd','Australian_terrier','Basenji',
    'Basset_hound','Beagle','Bearded_collie','Beauceron','Bedlington_terrier',
    'Belgian_malinois','Belgian_sheepdog','Belgian_tervuren','Bernese_mountain_dog',
    'Bichon_frise','Black_and_tan_coonhound','Black_russian_terrier','Bloodhound',
    'Bluetick_coonhound','Border_collie','Border_terrier','Borzoi','Boston_terrier',
    'Bouvier_des_flandres','Boxer','Boykin_spaniel','Briard','Brittany','Brussels_griffon',
    'Bull_terrier','Bulldog','Bullmastiff','Cairn_terrier','Canaan_dog','Cane_corso',
    'Cardigan_welsh_corgi','Cavalier_king_charles_spaniel','Chesapeake_bay_retriever',
    'Chihuahua','Chinese_crested','Chinese_shar-pei','Chow_chow','Clumber_spaniel',
    'Cocker_spaniel','Collie','Curly-coated_retriever','Dachshund','Dalmatian',
    'Dandie_dinmont_terrier','Doberman_pinscher','Dogue_de_bordeaux','English_cocker_spaniel',
    'English_setter','English_springer_spaniel','English_toy_spaniel','Entlebucher_mountain_dog',
    'Field_spaniel','Finnish_spitz','Flat-coated_retriever','French_bulldog','German_pinscher',
    'German_shepherd_dog','German_shorthaired_pointer','German_wirehaired_pointer','Giant_schnauzer',
    'Glen_of_imaal_terrier','Golden_retriever','Gordon_setter','Great_dane','Great_pyrenees',
    'Greater_swiss_mountain_dog','Greyhound','Havanese','Ibizan_hound','Icelandic_sheepdog',
    'Irish_red_and_white_setter','Irish_setter','Irish_terrier','Irish_water_spaniel',
    'Irish_wolfhound','Italian_greyhound','Japanese_chin','Keeshond','Kerry_blue_terrier',
    'Komondor','Kuvasz','Labrador_retriever','Lakeland_terrier','Leonberger','Lhasa_apso',
    'Lowchen','Maltese','Manchester_terrier','Mastiff','Miniature_schnauzer','Neapolitan_mastiff',
    'Newfoundland','Norfolk_terrier','Norwegian_buhund','Norwegian_elkhound','Norwegian_lundehund',
    'Norwich_terrier','Nova_scotia_duck_tolling_retriever','Old_english_sheepdog','Otterhound',
    'Papillon','Parson_russell_terrier','Pekingese','Pembroke_welsh_corgi','Petit_basset_griffon_vendeen',
    'Pharaoh_hound','Plott','Pointer','Pomeranian','Poodle','Portuguese_water_dog','Saint_bernard',
    'Silky_terrier','Smooth_fox_terrier','Tibetan_mastiff','Welsh_springer_spaniel','Wirehaired_pointing_griffon',
    'Xoloitzcuintli','Yorkshire_terrier']
    

def getInputParameters():
    '''
    Parsing input arguments.
    
    Args:
        -
                
    Raises:
        -

    Returns:
        images_path (string):
            The images pool path.
        number_of_images (integer):
            The number of images to be selected randomly from the images pool.
        trained_model (string):
            The saved trained model file.
            
    '''

    description_message = 'Determine dog\'s breed given an image, application.'

    args_parser = argparse.ArgumentParser(description = description_message,
        formatter_class = argparse.RawTextHelpFormatter)
    
    # Input arguments
    args_parser.add_argument('-p', action = 'store', required = True, 
        help = 'images path', metavar = 'images_path')
    args_parser.add_argument('-n', action = 'store', required = True, 
        help = 'number of random test images (1 to 10)', metavar = 'images_number')
    args_parser.add_argument('-m', action = 'store', required = True, 
        help = 'CNN TL trained model to load', metavar = 'CNN TL trained_model')
    
    args = args_parser.parse_args()
    
    # Validate arguments
    try:
        if int(args.n) < 1 or int(args.n) > 10:
            raise ValueError('number of random test images should be between 1 and 10.')
    except:
        raise ValueError('number of random test images should be an integer value.')
        
    if not os.path.exists(args.p.replace('/*', '')):
        raise ValueError('images path \'' + args.p + '\' does not exist.')

    if not os.path.isfile(args.m):
        raise ValueError('trained model file \'' + args.m + '\' does not exist.')
        
    return args.p, int(args.n), args.m
    

def selectTestImages(images_path, number_of_images):
    '''
    Create a list with random images.
    
    Args:
        images_path (string):
            The path of the pool of images.
        number_of_images (integer):
            The number of the randomly selected images.
                
    Raises:
        -

    Returns:
        images_list (list of strings):
            List containing the randomly selected image file names.
    '''
    
    return random.sample(list(np.array(glob.glob(images_path))), number_of_images) 


def predictBreed(test_image, trained_model, use_cuda):
    '''
    Predict breed of the input image using the input model.
    
    Args:
        test_image (string): 
            Test image file name.
        trained_model (torchvision.models):
            Trained model to be used for the prediction.
        use_cuda (boolean):
            If True the prediction will be executed on the GPU, 
            otherwise on the CPU.
            
                
    Raises:
        -

    Returns:
        predicted_breed (string):
            The predicted breed. A string from the C_CLASS_NAMES list.
        test_result (string):
            'Correct' or 'Incorrect' based on the classification result.
    '''

    image_tranform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
        transforms.Normalize(mean=[0.50, 0.50, 0.50], std=[0.50, 0.50, 0.50])])
    
    image_tensor = image_tranform(Image.open(test_image))
    if use_cuda:
        image_tensor = image_tensor.to('cuda')
        trained_model.cuda()
    
    # Put the model in evaluation mode
    trained_model.eval()
    
    # Predict the breed
    predicted_breed = C_CLASS_NAMES[trained_model(image_tensor.unsqueeze(0)).argmax()]
    test_result = predicted_breed in test_image
    
    return predicted_breed, 'Correct' if test_result else 'Incorrect'
    

if __name__ == '__main__':
    
    print('Determine dog\'s breed application.')
    
    # Read input parameters
    try:
        images_path, number_of_images, trained_model_file = getInputParameters()
    except ValueError as e:
        print('Error in the input parameters: ', e, sep = '')
        
    print('images_path = ', images_path, ', number_of_images = ', number_of_images,
        ', trained_model_file = ', trained_model_file, sep = '')
        
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print('CUDA availability (', use_cuda, '): Running on ', 'GPU.' if use_cuda else 'CPU.', sep = '')
    
    # Load the test images
    test_images = selectTestImages(images_path, number_of_images)
    
    # Create the CNN TL network and load the trained model
    cnn_tl = models.vgg16(pretrained = True)            
    cnn_tl.classifier[6] = torch.nn.Linear(cnn_tl.classifier[6].in_features, 133)        
    try:
        cnn_tl.load_state_dict(torch.load(trained_model_file))
    except:
        print('Error, the loaded trained model file is not compatible with the CNN TL.')
        exit()

    # Predict breed
    correct_classifications = 0
    for test_image in test_images:
        predicted_breed, classification_result = predictBreed(test_image, cnn_tl, use_cuda)
        if classification_result == 'Correct':
            correct_classifications += 1
            
        print('Test file: ', test_image, ', Predicted Breed: ', predicted_breed,
            ', Classification Result: ', classification_result, sep = '')
            
    print('\nPrediction Accuracy: {:.2f}%'.format(100.*correct_classifications/number_of_images))
         
