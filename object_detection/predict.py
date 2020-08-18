'''
MIT License

Copyright (c) 2020 Dat Vu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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