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
from fptu.Handle.speed_up import angle_calculator,sign_detection,remove_noise_matrix,remove_lane_lines,compute_centroid
from fptu.Reader.get_frames import read_input
from fptu.Reader.lcd_publish import lcd_print
from fptu.Preprocessing.preprocessing import pre_processing
from fptu.Controller.error import PID,Fuzzy
class segment:
    def __init__(self):

        K.clear_session() # Clear previous models from memory.
        
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
        
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.graph = tf.get_default_graph()
        
        f = gfile.FastGFile("/home/goodgame/Downloads/PSP_Dice_0.5_Focal_Data_None_Lib_loss-0.8959_val_loss-0.8947/PSP_Dice_0.5_Focal_Data_None_Lib_loss-0.8959_val_loss-0.8947.pb", 'rb')
        
        graph_def = tf.GraphDef()
        
        # Parses a serialized binary message into the current message.
        
        graph_def.ParseFromString(f.read())
        
        f.close()
        
        self.sess = tf.InteractiveSession(config=config)

        self.sess.graph.as_default()
        
        K.set_session(self.sess) 

        with self.sess.as_default():        # img_trab = cv2.imread('/home/goodgame/Desktop/hello.png') 
        # anh_lidar= np.zeros((768,768,3),np.uint8)
        # print('HERRRREEEEEEEEEEEEEEEEEEEEEEEEEEE1111111111111111111111', matran_tradinh_X)
        # print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee2222222222222222' ,matran_tradinh_Y)
        # for i in range(0,len(matran_tradinh_X)):
        #     if 0<matran_tradinh_X[i]<768 and 0< matran_tradinh_Y[i]<768:
        #         cv2.circle(anh_lidar,(768-matran_tradinh_X[i],768-matran_tradinh_Y[i]),2,(255,255,0), -1)
        # cv2.rectangle(anh_lidar,(384-5,384-11),(384+5,384+11),(0,0,255),-1)
        # cv2.imshow('hello',anh_lidar)
        # cv2.waitKey(1)
            with self.graph.as_default(): 
            # Import a serialized TensorFlow `GraphDef` protocol buffer
            # and place into the current default `Graph`.
                tf.import_graph_def(graph_def)

        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)

        self.label_codes, self.label_names = zip(*[self.parse_code(l) for l in open("/home/goodgame/catkin_ws/src/fptu_architecture/scripts/fptu/Model/label_colors.txt")])
        
        self.label_codes, self.label_names = list(self.label_codes), list(self.label_names)
        
        #label_codes[:5], label_names[:5]

        print(self.label_codes, self.label_names)

        self.code2id = {v:k for k,v in enumerate(self.label_codes)}
        self.id2code = {k:v for k,v in enumerate(self.label_codes)}

        self.name2id = {v:k for k,v in enumerate(self.label_names)}
        self.id2name = {k:v for k,v in enumerate(self.label_names)}


    def parse_code(self,l):
        '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
        '''
        if len(l.strip().split("\t")) == 2:
            a, b = l.strip().split("\t")
            return tuple(int(i) for i in a.split(' ')), b
        else:
            a, b, c = l.strip().split("\t")
            return tuple(int(i) for i in a.split(' ')), c

    def onehot_to_rgb(self,onehot, colormap):

        #Call global variables
        global bamlan
        '''Function to decode encoded mask labels
            Inputs: 
                onehot - one hot encoded image matrix (height x width x num_classes)
                colormap - dictionary of color to label id
            Output: Decoded RGB image (height x width x 3) 
        '''
        single_layer = np.argmax(onehot, axis=-1)
        output = np.zeros( onehot.shape[:2]+(3,) )
        for k in colormap.keys():
          output[single_layer==k] = colormap[k]
        #cv2.imshow('Frame',output)
        return np.uint8(output)
        
    def predict(self,frame):
        with self.sess.as_default():
            with self.graph.as_default():

                img = cv2.resize(frame,(144,144))
                
                # The code is implemented by Dat Vu on 28/04
                '''
                We can get line of road without using deep learning by opencv method (useful)
                We want to find line of road by convert image to HSV color
                Get line by HSV range 
                '''
                img = (img[...,::-1].astype(np.float32)) / 255.0
                
                img = np.reshape(img, (1, 144, 144, 3))

                softmax_tensor = self.sess.graph.get_tensor_by_name('import/softmax/truediv:0')
                
                predict_one = self.sess.run(softmax_tensor, {'import/input_1:0': np.array(img)})

                
                # In here, I've just add a parameter into onehot_to_rgb method.
                # Before the method only have two parameters (predict_one[0], self.id2code)
                image = self.onehot_to_rgb(predict_one[0],self.id2code)
                
                #Test for spped up

                #image = onehot_to_rgb_speedup(predict_one[0])

                #cv2.imshow("Prediction",image) # If you use local machine cv2.imshow instead of cv2_imshow  
                
                return image 

if __name__ == '__main__':

    rospy.init_node('deep_realtime', anonymous=True)

    segmentation = segment()
    read = read_input() 

    while not rospy.is_shutdown():
        start_time = time.time()
        if read.frame is not None:
            pre_processing_in = pre_processing(read.frame)
            try:
                #cv2.imshow('Framdde',read.frame)
                ouput = segmentation.predict(read.frame)
            except Exception as e:
                print(str(e))
        fps = 1 // (time.time() - start_time)
        rospy.loginfo("FPS IN RGB FRAME: " + str(fps))
        cv2.waitKey(1)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    #cv2.destroyAllWindows()
