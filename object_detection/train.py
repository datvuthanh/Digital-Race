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
from preprocessing_image import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import random
import cv2
import math
################# Parameters #####################
path = "Train" # folder with all the class folders
imageDimesions = (32,32,3)
testRatio = 0.2    # if 1000 images split will 200 for testing
validationRatio = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation
##################################################

############################### Importing of the Images ###############################
count = 0
images = []
classNo = []
myList = os.listdir(path)
print(myList)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))
        #cv2_imshow(curImg)
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

############################### Split Data
X_train, X_validation, y_train, y_validation = train_test_split(images, classNo, test_size=testRatio)
total_images_train = X_train.shape[0]
total_images_test = X_validation.shape[0]
batch_size = 32
steps_per_epoch_train = math.ceil(total_images_train / batch_size)
print("Steps per epoch:",steps_per_epoch_train)
steps_per_epoch_test = math.ceil(total_images_test / batch_size)

############################### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("Data Shapes")
print("Train",end = "");print(X_train.shape,y_train.shape)
print("Validation",end = "");print(X_validation.shape,y_validation.shape)
#print("Test",end = "");print(X_validation.shape,y_validation.shape)
assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
#assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert(X_train.shape[1:]==(imageDimesions))," The dimesions of the Training images are wrong "
assert(X_validation.shape[1:]==(imageDimesions))," The dimesionas of the Validation images are wrong "
#assert(X_test.shape[1:]==(imageDimesions))," The dimesionas of the Test images are wrong"



X_train=np.array(list(map(preprocessing,X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
X_validation=np.array(list(map(preprocessing,X_validation)))
#X_test=np.array(list(map(preprocessing,X_test)))

############################### ADD A DEPTH OF 1
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
#X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

############################### AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
dataGen= ImageDataGenerator(featurewise_center=False,
                            featurewise_std_normalization=False,
                            width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)
                            # brightness_range=[0.8,1.2])  # DEGREES
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=batch_size)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
X_batch,y_batch = next(batches)
 
# TO SHOW AGMENTED IMAGE SAMPLES
fig,axs=plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()
 
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0],imageDimesions[1]))
    axs[i].axis('off')
plt.show()
 
 
y_train = to_categorical(y_train,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
#y_test = to_categorical(y_test,noOfClasses)

############################### CONVOLUTION NEURAL NETWORK MODEL


############################### TRAIN
model = cnn_model()
model.summary()

epochs = 100

model_checkpoint = ModelCheckpoint(mode='auto', filepath='./object_detection-{epoch:03d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5', 
                      monitor='val_loss', 
                      save_best_only='True', 
                      save_weights_only='True', 
                      period=1,
                      verbose=1)
early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=0.0,
                                patience=10,
                                verbose=1)
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.2,
                                          patience=6,
                                          verbose=1,
                                          epsilon=0.001,
                                          cooldown=0,
                                          min_lr=0.00001)
callbacks = [model_checkpoint, early_stopping ,reduce_learning_rate]

print("Batch size",batch_size,"Steps per Epoch: ",steps_per_epoch_train)

history = model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batch_size),steps_per_epoch=steps_per_epoch_train,epochs=epochs,validation_data=(X_validation,y_validation),shuffle=True,callbacks=callbacks)