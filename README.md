# Face Verification System with Liveness Detection
## Introduction
In our digital world, secure identity verification is vital. Face verification, a part of biometric authentication, offers convenience and accuracy but faces threats from spoofing attacks. These attacks aim to deceive the system with static images or videos, undermining its ability to differentiate living users from impersonators.

Liveness detection is crucial for security and trust in face verification systems, ensuring users are real individuals. Our paper introduces a solution that integrates liveness detection, enhancing system security. We aim to explore the methodologies, technologies, and benefits of this system.

## Features
CNNs: Used for deep learning-based feature extraction from facial images.

LBP and HOG: Traditional techniques for extracting features from facial characteristics.

MTCNN: A face detection algorithm that locates and identifies faces in images.

Triplet Loss: Employed for learning embeddings to distinguish authorized and unauthorized individuals.

LRN: Enhances feature quality through local contrast enhancement.

Transfer Learning: Leverages pre-trained models like VGGFace or ResNet for face recognition.

Siamese and Triplet Networks: Architectures used to differentiate between faces for verification.

SVMs: Classify face features into authorized and unauthorized categories for improved security.

Deep Metric Learning: Learns similarity metrics to compare and verify faces.

Data Pre-processing and Augmentation: Enhances model performance through data manipulation.

Custom Layers and Loss Functions: Tailored for specific requirements like face recognition and liveness detection.

Feature Embeddings: Learn and compare key characteristics of faces for identity verification.

Data Handling and Storage: Manages training data, templates, and access logs.

Face Recognition and Liveness Detection Models: Essential for decision-making in the system.

Image Pre-processing: Improves model accuracy through image enhancements.

Custom Utility Functions and Libraries: Support core program features with additional functionalities and algorithmic details.

## Requirements
#### HARDWARE REQUIREMENTS
 
✓ NVIDIA GeForce GTX 1650

✓ 8 GB RAM 

✓ 12 Gen Intel Core i5 – 1240P

#### SOFTWARE REQUIREMENTS

✓ Python 3.8

✓ Anaconda

✓ Tensorflow

✓ Open CV

## Architecture Diagram/Flow

![3-Figure1-1](https://github.com/SarankumarJ/Face-Verification-System-with-Liveness-Detection/assets/94778101/c16d1a76-cc37-40b8-9718-f3129404b5f5)

## Program
```py
# Import the prerequisite libraries
import os
import numpy as np
import cv2
import math
import random
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Conv2D,
    ZeroPadding2D,
    Activation,
    Input,
    BatchNormalization,
    MaxPooling2D,
    AveragePooling2D,
    Lambda,
    Flatten,
    Dense,
)
from keras.layers import concatenate
from keras.layers import Layer
from keras import backend as K
from numpy import genfromtxt
import pandas as pd
from utils import LRN2D
from mtcnn import MTCNN
import utils

%load_ext autoreload
%autoreload 2

# Load weights from csv files
weights = utils.weights
weights_dict = utils.load_weights()

# Set layer weights of the model
for name in weights:
  if model.get_layer(name) != None:
    model.get_layer(name).set_weights(weights_dict[name])
  elif model.get_layer(name) != None:
    model.get_layer(name).set_weights(weights_dict[name])

def check_right(righteye,lefteye,nose,orig_eye_dist,orig_nose_x):
    dist = math.sqrt((righteye[0]-lefteye[0])**2 + (righteye[1]-lefteye[1])**2)
    if dist<=orig_eye_dist*0.52 and nose[0]<orig_nose_x:
        return True
    else:
        return False
    
def check_left(righteye,lefteye,nose,orig_eye_dist,orig_nose_x):
    dist = math.sqrt((righteye[0]-lefteye[0])**2 + (righteye[1]-lefteye[1])**2)
    if dist<=orig_eye_dist*0.52 and nose[0]>orig_nose_x:
        return True
    else:
        return False
    
def check_smile(mouth_right,mouth_left,orig_mouth_dist):
    dist = math.sqrt((mouth_right[0]-mouth_left[0])**2 + (mouth_right[1]-mouth_left[1])**2)
    #print(dist)
    if dist>=orig_mouth_dist*1.3:
        return True
    else:
        return False
    
def check_pout(mouth_right,mouth_left,orig_mouth_dist):
    dist = math.sqrt((mouth_right[0]-mouth_left[0])**2 + (mouth_right[1]-mouth_left[1])**2)
    #print(dist)
    if dist<=orig_mouth_dist-10:
        return True
    else:
        return False

# main
choice = input("Are you a Registered User: ")
if choice == "no":
    name = input("Enter your Name and be ready infront of camera: ")
    store_sample(name)
    print("Registration Successful")
    
ch = input("Would you like to open the device: ")
if ch == 'yes':
    input_embeddings = create_input_image_embeddings()
    if input_embeddings != {}:
        print("Face Verification in process....")
        recognize_faces_in_cam_50(input_embeddings)
    else:
        print("No Registered User Found")

```

## Output

![image](https://github.com/SarankumarJ/Face-Verification-System-with-Liveness-Detection/assets/94778101/e483067e-87e4-43df-8ff6-49baaa9f4b88)

![image](https://github.com/SarankumarJ/Face-Verification-System-with-Liveness-Detection/assets/94778101/daf56acb-0fef-4e88-9822-ebe3869c1ad7)

#### Real User

![image](https://github.com/SarankumarJ/Face-Verification-System-with-Liveness-Detection/assets/94778101/bff4e884-6ea8-45f1-a6c0-7fc9853d0efb)

#### Fake User

![image](https://github.com/SarankumarJ/Face-Verification-System-with-Liveness-Detection/assets/94778101/1e4c582c-c7fc-4029-997e-3ff4aedff661)

## Result

The proposed Face Verification System with Liveness Detection represents a significant leap 
forward in the field of identity verification. This advanced system combines state-of-the-art 
face recognition and liveness detection technologies to provide a comprehensive and secure 
solution for ensuring the authenticity of individuals in various applications, from access 
control to online identity verification.

In conclusion, the Face Verification System with Liveness Detection is poised to set new 
standards in identity verification. It offers security and reliability in a world where privacy 
and security are paramount concerns. With its sophisticated technology and comprehensive 
features, it represents a significant step toward a safer and more efficient future for identity 
verification in a wide range of applications. This system has the potential to revolutionize the 
way we establish and confirm identities, making it an invaluable asset for organizations and 
individuals seeking advanced security and trust in an increasingly digital and interconnected 
world.
