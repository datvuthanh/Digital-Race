'''
Copyright (c) Dat Vu, 2020
All Rights Reserved
Email: stephen.t.vu@hotmail.com
'''
#################  Import Libraries  #################
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from model import cnn_model

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import random
import cv2
import math
from preprocessing_image import preprocessing

def getCalssName(classNo):
    if   classNo == 0: return 'Stop'
    elif classNo == 1: return 'Right'
    elif classNo == 2: return 'Left'
    elif classNo == 3: return 'Straight'
    elif classNo == 4: return 'No Left'
    elif classNo == 5: return 'No Right'

########################### Evalutation ###########################
model = cnn_model()
model.load_weights('/content/weight.h5') 
threshold=0.75
imgOrignal = cv2.imread('./image.jpg')
img = np.asarray(imgOrignal)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
img = img.reshape(1, 32, 32, 1)
#cv2.putText(imgOrignal, "CLASS: " , (20, 35), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (0, 0, 255), 2, cv2.LINE_AA)
# PREDICT IMAGE
predictions = model.predict(img)
classIndex = model.predict_classes(img)
probabilityValue =np.amax(predictions)
print("SIGN: ",getCalssName(classIndex),"\tProbability: ", probabilityValue*100, "%")
# if probabilityValue > threshold:
#     cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (0, 0, 255), 2, cv2.LINE_AA)