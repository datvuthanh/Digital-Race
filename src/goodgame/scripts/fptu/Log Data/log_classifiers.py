#!/usr/bin/env python3

import glob
import random
import time
import numpy as np
import cv2
import math 
from numpy import linalg as LA
import json
import os
import numpy as np
from std_msgs.msg import Header,String,Float32,Int8,Bool
from sensor_msgs.msg import CompressedImage,Image
import rospy

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
import keras.backend as K
from numba import jit
from fptu.Handle.speed_up import angle_calculator,sign_detection,remove_noise_matrix,remove_lane_lines,compute_centroid,noright,noleft,getclassIndex,sign_density_check,getClassName
from fptu.Reader.get_frames import read_input
from fptu.Reader.lcd_publish import lcd_print
from fptu.Preprocessing.preprocessing import pre_processing
from fptu.Controller.error import PID,Fuzzy
from cv_bridge import CvBridge, CvBridgeError
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats


class detection:
    def __init__(self):

        K.clear_session() # Clear previous models from memory.
        
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
    
        self.graph = tf.get_default_graph()
        
        self.sess = tf.InteractiveSession(config=config)

        self.sess.graph.as_default()
        
        K.set_session(self.sess) 

        self.classify_model = self.cnn_model()

        f = gfile.FastGFile("/home/goodgame/catkin_ws/src/semi_final/scripts/fptu/Handle/Utils/Tensor_RT Models/TensorRT_CNN_model_0506.pb", 'rb')
        
        graph_def = tf.GraphDef()
        
        # Parses a serialized binary message into the current message.
        
        graph_def.ParseFromString(f.read())
        
        f.close()
        
        self.sess = tf.InteractiveSession(config=config)

        self.sess.graph.as_default()
        
        K.set_session(self.sess) 

        with self.sess.as_default():       
            with self.graph.as_default(): 
            # Import a serialized TensorFlow `GraphDef` protocol buffer
            # and place into the current default `Graph`.
                tf.import_graph_def(graph_def)

        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)

    def grayscale(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img
    def equalize(self,img):
        img = cv2.equalizeHist(img)
        return img
    def preprocessing(self,img):
        img = self.grayscale(img)
        img = self.equalize(img)
        img = img/255
        return img

    def processing(self,frame):
        img = cv2.resize(frame,(32,32))
        
        img = self.preprocessing(img)

        img = img.reshape(1,32,32,1)

        return img      

    def cnn_model(self):
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), padding='same',
        input_shape=(32, 32, 1),
        activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(64, (3, 3), padding='same',
        activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(128, (3, 3), padding='same',
        activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))
        #model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    
        return model

def localization(centroids):
    global count_cnn

    centroid_x = int(centroids.data[0])
    centroid_y = int(centroids.data[1])
    sign_density = int(centroids.data[2])   

    real_message = int(centroids.data[3])

    image = read.frame
    image = cv2.resize(image,(144,144))
    
    # cv2.imshow("Raw",image)
    with session.as_default():
        with session.graph.as_default():
            check,image_cnn = sign_density_check(image,sign_density,centroid_x,centroid_y)
            cv2.imwrite('/media/goodgame/3332-323312/Log/0212/CNN/Data_CNN_0212_'+str(count_cnn)+'.png', image_cnn)
            count_cnn += 1
            # cv2.imshow("frame_CNN",image_cnn)
            if(check == True):
                image_cnn = detect.processing(image_cnn)

                softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
                
                predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(image_cnn)})

                index, value = getclassIndex(predict_one[0])
                
                rospy.loginfo("Traffic Sign:\t" + getClassName(index) + "\t" + str(round(value*100,3)))

                if value > 0.9 and real_message == 1: # Khong phai fake message
                    lcd.update_message(getClassName(index),2,2)
                    traffic_sign_publish.publish(index)
    #cv2.waitKey(1)


if __name__ == '__main__':
    
    count_cnn = 0

    rospy.init_node('log_classifiers', anonymous=True)
    
    read = read_input() 

    detect = detection()

    lcd = lcd_print("Goodgame",1,1) # Init LCD

    session = detect.sess

    rospy.Subscriber("/localization", numpy_msg(Floats), localization)

    traffic_sign_publish = rospy.Publisher("/traffic_sign_id",Int8,queue_size = 1)    
    
    ## The first time we need call model to predict because numba need to complie the first time.

        # image_fake = cv2.imread('/home/goodgame/catkin_ws/src/semi_final/scripts/fptu/Handle/Fake_images/goodgame_data_sign207.png')
        # image_fake = cv2.resize(image_fake,(144,144))
    img_dir = '/home/goodgame/catkin_ws/src/semi_final/scripts/fptu/Log Data/Fake_images'
    img_list = os.listdir(img_dir)
    
    count_fake = 0

    for i in range(len(img_list)):
        with session.as_default():
            with session.graph.as_default():
                    if i == 0:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,200,72,72) #TH 250>=sign_density>0
                    if i == 1:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,450,72,72) #TH 500>=sign_density>250
                    if i == 2:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,800,72,72) #TH 1000>=sign_density>500
                    if i == 3:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,1500,72,72) #TH 2000>=sign_density>1000
                    if i == 4:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,3000,72,72) #TH 4000>sign_density>2000
                    if i == 5:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,200,135,72) #TH Dac biet 250>=sign_density>0 
                    if i == 6:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,450,130,72) #TH Dac biet500>=sign_density>250
                    if i == 7:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,800,125,72) #TH Dac biet1000>=sign_density>500
                    if i == 8:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,1500,120,72) #TH Dac biet 2000>=sign_density>1000
                    if i == 9:
                        image_fake = cv2.imread(img_dir+'/'+img_list[i])
                        check,image_fake = sign_density_check(image_fake,3000,111,72) #TH Dac biet 4000>sign_density>2000
                    print("Dang o day: ", i)
                    # # # Resize to 32x32 after convert to grayscale
                    image_fake = detect.processing(image_fake)
                    softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
                    predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(image_fake)})
                    index, value = getclassIndex(predict_one[0])
                    print("Classssssssssssssssssssssssssssss:" ,getClassName(index),value)
    ###################################################################################################        
    while not rospy.is_shutdown():
        if read.frame is not None:
            # cv2.imshow("hello",read.frame)
            if count_fake == 0:
                image = cv2.resize(read.frame,(144,144))
                check,image = sign_density_check(image,3000,111,72) #TH Dac biet 4000>sign_density>2000
                #cv2.imshow("testew",image)
                image = detect.processing(image)
                softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
                predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(image)})
                index, value = getclassIndex(predict_one[0])
                print("Predict real-time fakeeeee:" ,getClassName(index),value)
                count_fake += 1
        cv2.waitKey(1)           

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    cv2.destroyAllWindows()
