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
from fptu.Handle.speed_up import angle_calculator,sign_detection,remove_noise_matrix,remove_lane_lines,compute_centroid,noright,noleft,getclassIndex
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

        # with self.sess.as_default():       
        #     with self.graph.as_default(): 
        #         self.classify_model.load_weights('/home/goodgame/catkin_ws/src/fptu_architecture/scripts/fptu/Model/Classifier_models/Seg_Local-021_loss-0.0052_val_loss-0.0005.h5')
        #         self.classify_model._make_predict_function()
        f = gfile.FastGFile("/home/goodgame/catkin_ws/src/fptu_architecture/scripts/fptu/Model/Classifier_models/Seg_Local-021_loss-0.0052_val_loss-0.0005.pb", 'rb')
        
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
    ############################### CONVOLUTION NEURAL NETWORK MODEL
    def myModel(self):
        no_Of_Filters=32
        size_of_Filter=(5,5) # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
                            # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
        size_of_Filter2=(3,3)
        size_of_pool=(2,2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
        no_Of_Nodes = 1024   # NO. OF NODES IN HIDDEN LAYERS
        model= Sequential()
        model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(32,32,1),activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
        model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
        model.add(MaxPooling2D(pool_size=size_of_pool)) # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
    
        model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))
        model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
        model.add(MaxPooling2D(pool_size=size_of_pool))
        model.add(Dropout(0.5))
    
        model.add(Flatten())
        model.add(Dense(no_Of_Nodes,activation='relu'))
        model.add(Dropout(0.5)) # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
        model.add(Dense(6,activation='softmax')) # OUTPUT LAYER
        # COMPILE MODEL
        #model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        return model                

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
    global image
    global centroid_x_after
    global centroid_y_after
    global sign_density

    centroid_x_after = int(centroids.data[0])
    centroid_y_after = int(centroids.data[1])
    sign_density = int(centroids.data[2])   


    #print("HEREHRJEHSARJHSEKJRHJKSRHKJWAHRKAWHRJHWAJKRHWA")

        #cv2.waitKey(1)
def sign_density_check(image_localization,sign_density,centroid_x,centroid_y):
    if 4000>sign_density>2000: 
        image_localization = image_localization[centroid_y - 30 : centroid_y + 30, centroid_x - 30 : centroid_x + 30]
        print("Truong hop 4000>sign_density>2000")
    if 2000>sign_density>1000:
        image_localization = image_localization[centroid_y - 25 : centroid_y + 30, centroid_x - 25 : centroid_x + 25]
        print("Truong hop 2000>sign_density>1000")
    if 1000>sign_density>500:
        image_localization = image_localization[centroid_y - 20 : centroid_y + 25, centroid_x - 20 : centroid_x + 20]
        print("Truong hop 1000>sign_density>500")
    if 500>sign_density>250:
        image_localization = image_localization[centroid_y - 18 : centroid_y + 22, centroid_x - 18 : centroid_x + 20]
        print("Truong hop 500>sign_density>250")
    if 250>sign_density>0:
        image_localization = image_localization[centroid_y - 12 : centroid_y + 12, centroid_x - 12 : centroid_x + 12]
        print("Truong hop 250>sign_density>0")
    return image_localization
 

def getClassName(classNo):
    if classNo == 0: return "Stop"
    if classNo == 1: return "Right"
    if classNo == 2: return "Left"
    if classNo == 3: return "Straight"
    if classNo == 4: return "No Left"
    if classNo == 5: return "No Right"

def convert_to_np(ros_data):

    np_app = np.fromstring(ros_data.data,np.uint8)

    frame = cv2.imdecode(np_app,cv2.IMREAD_COLOR)

    return frame


def callback_rgb(ros_data):

    global centroid_x
    global centroid_x_after
    global centroid_y
    global centroid_y_after
    global sign_density
    global detect

    frame = convert_to_np(ros_data)

    if frame is not None:    
        #read.btn.led_send_message(True)
        image = cv2.resize(frame,(144,144))
        #cv2.imshow("frame",image)

        # image_cnn = sign_density_check(image,300,30,30)
        # cv2.imshow("frame",image_cnn)
        # #cv2.imwrite('/media/goodgame/3332-32336/CNN/image_cnn_test_2705/Image_CNN'+str(count_cnn)+'.png',image_cnn)
        # # # # Resize to 32x32 after convert to grayscale
        # image_cnn = detect.processing(image_cnn)
        # #count_cnn +=1
        # with session.as_default():
        #     with session.graph.as_default():
        #         softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
                
        #         predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(image_cnn)})        # output = np.zeros( onehot.shape[:2]+(3,) )
        # for k in colormap.keys():
        #   output[single_layer==k] = colormap[k]
        # cv2.imshow("Output",output)
            #     fake_image = cv2.cvtColor(fake_image,cv2.COLOR_BGR2GRAY)
            #     fake_image = cv2.resize(fake_image,(32,32))
            #     fake_image = fake_image.reshape(1,32,32,1)
            #     with session.as_default():
            #         with session.graph.as_default():
            #             for i in range(0,20):
            #                 softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
            #                 predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(fake_image)})         
            #     centroid_x_after = 0
            #     centroid_y_after = 0    
        try:
            if centroid_x_after != centroid_x or centroid_y_after != centroid_y:
                with session.as_default():
                    with session.graph.as_default():
                        image_cnn = sign_density_check(image,sign_density,centroid_x_after,centroid_y_after)
                        #cv2.imshow("frame",image_cnn)
                        #cv2.imwrite('/media/goodgame/3332-32336/CNN/image_cnn_test_2705/Image_CNN'+str(count_cnn)+'.png',image_cnn)
                        # # # Resize to 32x32 after convert to grayscale
                        image_cnn = detect.processing(image_cnn)
                        # #count_cnn +=1
                        softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
                        
                        predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(image_cnn)})

                        index, value = getclassIndex(predict_one[0])
                        
                        print("Classssssssssssssssssssssssssssss:" ,getClassName(index),value)

                        if value > 0.9:
                            lcd.update_message(getClassName(index),1,1)
                            steer_bool.publish(index)
                
                centroid_x = centroid_x_after
                centroid_y = centroid_y_after
        except Exception as e:
            print(str(e))
    #cv2.waitKey(1)
    return frame

if __name__ == '__main__':

    rospy.init_node('classifiers', anonymous=True)
    
    #read = read_input() 
    centroid_x = 0
    centroid_y = 0
    sign_density = 0
    centroid_x_after = 0
    centroid_y_after = 0
    count_cnn = 0
    detect = detection()
    lcd = lcd_print("Goodgame",1,1) # Init LCD

    session = detect.sess

    rospy.Subscriber("/localization", numpy_msg(Floats), localization)

    sign_pub = rospy.Publisher('/localization', numpy_msg(Floats),queue_size=1)

    steer_bool = rospy.Publisher("/steer_bool",Int8,queue_size = 1)    

    get_rgb = rospy.Subscriber("/camera/rgb/image_raw/compressed",
                                    CompressedImage,
                                    callback_rgb,
                                    queue_size=1,buff_size=2**24)



    ## The first time we need call model to predict 
    image_fake = np.zeros((32,32))
    image_fake = image_fake.reshape(1,32,32,1)
    with session.as_default():
        with session.graph.as_default():
            softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
            predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(image_fake)})
    #pre_fake = model.predict(image_fake)
    #print("Loaded Model",pre_fake)
    #print(model.summary())

    ##################################################

    steer_bool_data = Int8()

    # while not rospy.is_shutdown():
    #     print("hah")
        # if read.frame is not None:    
        #     read.btn.led_send_message(True)

        #     image = read.frame
        #     image = cv2.resize(read.frame,(144,144))
            
        #     try:
        #         # if centroid_x_after == -1 and centroid_y_after == -1 and sign_density == -1:
        #         #     print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEe")
        #         #     fake_image = image
        #         #     # cv2.imshow("Test2",fake_image)
        #         #     fake_image = cv2.cvtColor(fake_image,cv2.COLOR_BGR2GRAY)
        #         #     fake_image = cv2.resize(fake_image,(32,32))
        #         #     fake_image = fake_image.reshape(1,32,32,1)
        #         #     with session.as_default():
        #         #         with session.graph.as_default():
        #         #             for i in range(0,20):
        #         #                 softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
        #         #                 predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(fake_image)})         
        #         #     centroid_x_after = 0
        #         #     centroid_y_after = 0    

        #         if centroid_x_after != centroid_x or centroid_y_after != centroid_y:
        #             image_cnn = sign_density_check(image,sign_density,centroid_x_after,centroid_y_after)
        #             cv2.imshow("frame",image_cnn)
        #             #cv2.imwrite('/media/goodgame/3332-32336/CNN/image_cnn_test_2705/Image_CNN'+str(count_cnn)+'.png',image_cnn)
        #             # # # Resize to 32x32 after convert to grayscale
        #             image_cnn = detect.processing(image_cnn)
        #             #count_cnn +=1

        #             with session.as_default():
        #                 with session.graph.as_default():
        #                     softmax_tensor = session.graph.get_tensor_by_name('import/dense_2/Softmax:0')
                            
        #                     predict_one = session.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(image_cnn)})

        #                     index, value = getclassIndex(predict_one[0])
                            
        #                     print("Classssssssssssssssssssssssssssss:" ,getClassName(index),value)

        #                     if value > 0.9:
        #                         lcd.update_message(getClassName(index),1,1)
                    
        #             centroid_x = centroid_x_after
        #             centroid_y = centroid_y_after
        #         #cv2.imshow("frame",read.frame)
        #     except Exception as e:
        #         print(str(e))
        # cv2.waitKey(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    #cv2.destroyAllWindows()
