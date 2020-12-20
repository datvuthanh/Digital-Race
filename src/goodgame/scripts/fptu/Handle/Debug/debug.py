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
from fptu.Handle.speed_up import angle_calculator,sign_detection,remove_noise_matrix,remove_lane_lines,compute_centroid,noright,noleft,compute_sign,find_biggest_components,speed_up_no,count_line,cut_road_speedup,cut_road_speedup_for_mid
from fptu.Reader.get_frames import read_input
from fptu.Reader.lcd_publish import lcd_print
from fptu.Preprocessing.preprocessing import pre_processing
from fptu.Controller.error import PID,Fuzzy
from cv_bridge import CvBridge, CvBridgeError
# from fptu.Controller.control_pass_sign import pass_sign
# Add new
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

# Add new 05/06
import roslaunch
from datetime import datetime
import threading

class segmentation:
    
    def __init__(self):

        K.clear_session() # Clear previous models from memory.
        
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
        
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.graph = tf.get_default_graph()
        
        # url = "/home/goodgame/catkin_ws/src/semi_final/scripts/fptu/Handle/Utils/Tensor_RT Models/Model_2109_144_rb.pb"
        url = "/home/goodgame/catkin_ws/src/semi_final/scripts/fptu/Model/PSPNET_3classes/seg_datvu_0812_8846_9342.pb"

        graph_def = load_model(url)
        
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

        self.id2code = class_return()

        # Publish centroid of traffic sign to classifiers.py
        self.sign_pub = rospy.Publisher('/localization', numpy_msg(Floats),queue_size=1)
        

    def onehot_to_rgb(self,onehot, colormap):

        '''Function to decode encoded mask labels
            Inputs: 
                onehot - one hot encoded image matrix (height x width x num_classes)
                colormap - dictionary of color to label id
            Output: Decoded RGB image (height x width x 3) 
        '''
        
        single_layer = np.argmax(onehot, axis=-1) # Dat Vu convert numpy to tensorflow

        return single_layer

    def predict(self,frame):
        with self.sess.as_default():
            with self.graph.as_default():
                
                img = frame[220:,:]
                img = cv2.resize(img,(144,144))
                
                # test = img.copy()
                
                # test = test[20:,:]
                # cv2.imshow("Frame",img)
                
                # cv2.imwrite('/home/goodgame/Desktop/image/road.png',img)
                #img_cnn = img.copy()
                # The code is implemented by Dat Vu on 28/04
                '''
                We can get line of road without using deep learning by opencv method (useful)
                We want to find line of road by convert image to HSV color
                Get line by HSV range 
                '''
                # ## Pre-processing
                # lower = np.array([0, 0, 215]) #### ---> Modify here
                # upper = np.array([179, 255, 255])
                # # Create HSV Image and threshold into a range.
                # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # mask = cv2.inRange(hsv, lower, upper)
                # cv2.imshow("LINE OPENCV", mask)
                # line_opencv = mask[64:,:] # We want to get half of image below 
                ############################################
                image = (img[...,::-1].astype(np.float32)) / 255.0
                
                image = np.reshape(image, (1, 144, 144, 3))

                softmax_tensor = self.sess.graph.get_tensor_by_name('import/softmax/truediv:0')
                
                predict_one = self.sess.run(softmax_tensor, {'import/input_1:0': np.array(image)})
                
                # print(predict_one[0])
                # In here, I've just add a parameter into onehot_to_rgb method.
                # Before the method only have two parameters (predict_one[0], self.id2code)
                image = self.onehot_to_rgb(predict_one[0],self.id2code)
                
                #Test for spped up

                #image = onehot_to_rgb_speedup(predict_one[0])

                #cv2.imshow("Prediction",image) # If you use local machine cv2.imshow instead of cv2_imshow  
                
                return image 


def load_model(url):
    f = gfile.FastGFile(url, 'rb')
    
    graph_def = tf.GraphDef()
    
    # Parses a serialized binary message into the current message.
    
    graph_def.ParseFromString(f.read())
    
    f.close()

    return graph_def


def parse_code(l):
    '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
    '''
    if len(l.strip().split("\t")) == 2:
        a, b = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c


def class_return():
    
    label_codes, label_names = zip(*[parse_code(l) for l in open("/home/goodgame/catkin_ws/src/semi_final/scripts/fptu/Model/PSPNET_3classes/label_colors.txt")])

    label_codes, label_names = list(label_codes), list(label_names)

    id2code = {k:v for k,v in enumerate(label_codes)}

    return id2code

def line_cut(road,line,bamlan):
    index = 0 # Cuoi cung cua line duong
    row_save = 0
    for i in range(144,0,-1):
        for j in range(144,0,-1):
            if bamlan == 1: # Bam lan phai, cat duong lan phai
                if line[i][j] == 255:
                    return i,j

def process(single_layer,cut_road=False):
    global bamlan

    global sign_id

    '''
    If we want to visualize all classes on frame
    Uncomment these line
    Of course, when we visualize frame which 'll make our algorithm slower (decrease fps)
    '''"""  """
    ############################
    output = np.zeros( single_layer.shape[:2]+(3,) )
    for k in model.id2code.keys():
       output[single_layer==k] = model.id2code[k]
    cv2.imshow("Output",output)
    # out.write(output)
    ############################

    '''
    When we want to visualize only one class
    Uncomment these line
    '''
    ############################
    line = np.zeros(single_layer.shape[:2])
    line[single_layer == 2] = 255  # New : 2
    #line_2 = line[40:,:]#.copy()
    # if bamlan == 1:
    #    line_2 = line[40:90,0:96]#.copy()
    # if bamlan == 0:
    #    line_2 = line[40:90,48:]
    
    # NEW
    
    
    # line = cv2.resize(line, (128, 128))
    # line = line[64:, :]
    
    #ret,thresh = cv2.threshold(line_2,127,255,0)
    # line_2 = line_2.astype(np.uint8)
    # _,contours,_ = cv2.findContours(line_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(line_2,contours,-1,255,2)
    # cv2.imshow("CONTOURS",contours)

    #cv2.imshow("Line",line_2)

    # kernel = np.ones((3,3),np.uint8)
    # line = cv2.morphologyEx(line, cv2.MORPH_OPEN, kernel)
    
    kernel = np.ones((5,5),np.uint8)

    line = cv2.dilate(line, kernel,iterations = 1)
    
    #print("POLY:",fit.shape,fit)
    
            
        
    # cv2.imshow("Line2",line)
    
    # line =cv2.erode(line, kernel, iterations =1)
    
    # cv2.imshow("line erode",line)

    # cv2.imshow("Line before processing",line)
    # line_opencv = cv2.resize(line_opencv,(128,128))
    # line_opencv = line_opencv[64:,:]

    # for i in range(0,64):
    #     for j in range(0,128):
    #         if line_opencv[i][j] == 255:
    #             line[i][j] = 255

    # cv2.imshow("Line after processing: ",line)

    # cv2.imshow("line deeeplearning",line)

    # line_1= np.zeros(onehot.shape[:2])
    # line_1[single_layer==7] = 255
    # cv2.imshow("right",line_1)
    ############################

    # Dat Vu add the code on 28/04
    ################################
    road = np.zeros((144, 144))
    road[single_layer == 1] = 255  # New : 1

    # Opening algorithm to reduce noise in two lane lines.
    # kernel = np.ones((3,3),np.uint8)
    # line = cv2.morphologyEx(line_opencv, cv2.MORPH_OPEN, kernel)
    '''
    In this line, we need to remove noise road which is not inside line of road
    We can use line or line_opencv
    line : after processing line_opencv by opening algorithm (not recommended) but this can be useful in few cases
    line_opencv : raw line which we get from handling line road by HSV Color (Opencv)
    
    
    |___________|           |       |
    |   Noise   |           |       |
    |___________|   Car     |       |
                |           |       |

    '''
    # Comment this line if model don't have noise
    
    '''
    tradinh
    After remove oppsite lane we will bird-view road
    Firstly, we need resize image to 144x144
    Secondly, we use open-cv algorithm to warp image
    Finally, we will resize image again to 100x144 (Tra Dinh recommended here)
    '''
    ''' CUT IMAGE '''
    # print("THAM SO: ",thamsocatduong)
    
    # if thamsocatduong < 72: # Remove noise by get biggest components
    #     road = split_components(line, road)  # using line or line_opencv
    #     road =  road.astype('uint8')
    #     nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(road, connectivity=4)  # We can modify 4 or 8
    #     #print("NUMBER OF COMPONENTS: ", nb_components)
    #     max_label = find_biggest_components(nb_components, output, stats)
    #     road = np.zeros((144,144))
    #     road[output == max_label] = 255     
    
    # line = remove_noise_matrix_again(line)  # using line or line_opencv
    # try:
    #     coordinates = np.where(line == 255)
    #     y_range = coordinates[0]
    #     x_range = coordinates[1]
    #     fit = np.polyfit(y_range, x_range, 2)
    #     line = draw_line(line,fit[0],fit[1],fit[2])
    # except:
    #     pass        
    
    # road = remove_noise_matrix(line,bamlan,road)
    
    # cv2.imshow("LINE __", line)
    # if cut_road == False:    
    #     road = road[40:,:] # Cat duong
    # if cut_road == True:
    #     print("DANG CAT DUONG!!!!")
    #     road = road[36:,:]
    
    #line = cv2.line(line, (0,0), (120,40), 255, 2) 


    # print("SO LUONG LINE: ",line_counting)    
    # if 1600 < line_counting < 1800:
    #     print("DOAN CAN RE")
    # if line_counting < 800:
    #     print("DOAN DI THANG")
        #line = cv2.line(line, (120,0), (120,144), 255, 3) 


    road = remove_noise_matrix(line,bamlan,road) # using line or line_opencv
    
    line_counting = count_line(road) # Dem mat do line tren mot duong thang

    cv2.imshow("road after remove noise",road)
    road = road[40:,:] # Default : 40
    
    # cv2.imshow("TEST",line_2)
    rospy.logerr("MAT DO LINE DUONG NGANG: " + str(line_counting) + "    " + str(sign_id))
    

    
    cv2.imshow("ROAD CUT", road)  
    
    road = cv2.resize(road,(144,144))


    # kernel = np.ones((5,5),np.uint8)
    # road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel)
    
    
    
    
    # Remove opposite lane
    '''
    This line we remove opposite lane which we not be in there.
    |         |        |
    |         |        |
    | Remove  |  Car   |
    |  Lane   |        |
    |         |        |
    '''

    #road = remove_lane_lines(road,bamlan) # Shape (64,144)
    # cv2.imwrite('/home/goodgame/Desktop/image/road.png',road)
    # angle = compute_centroid(road,bamlan)
    # print("ANGLE: ",angle)
    # cv2.imshow("Remove lane",road)    
    
    transform_matrix = pre_processing_in.perspective_transform()
    road_on_birdview = pre_processing_in.birdView(
        road, transform_matrix['M'])
    # line_on_birdview = pre_processing_in.birdView(
    #     line, transform_matrix['M'])
    # # print("ROAD_ON_BIRDVIEW SHAPE: ", road_on_birdview.shape)
    # road_on_birdview = road_on_birdview[96:,:]
    # print("HELLO ROAD ON BIRDVIEW  : ", road_on_birdview.shape)
    # road_on_birdview = cv2.resize(road_on_birdview, (100,144))
    # line_on_birdview = cv2.resize(line_on_birdview, (100,144))
    cv2.imshow("Bird View 1",road_on_birdview)
    # cv2.imshow("LINE BIRD VIEW", line_on_birdview)

    if bamlan == 1 and sign_id == 6 and cut_road == True: # Cut duong
        rospy.logwarn("BAM LAN PHAI THUC HIEN HANH DONG CAT DUONG")
        road_on_birdview = cut_road_speedup(road_on_birdview,width=88)
    if bamlan == 0 and sign_id == 5 and cut_road == True: # Cut duong doan chu D cho mode mid
        rospy.logwarn("BAM LAN MID THUC HIEN HANH DONG CAT DUONG")
        road_on_birdview = cut_road_speedup_for_mid(road_on_birdview,width=70)
    
    cv2.imshow("Bird View 2",road_on_birdview)

    road_on_birdview = remove_lane_lines(road_on_birdview,bamlan) # Shape (64,144)

    cv2.imshow("Bird View 3",road_on_birdview)

    
    # road_on_birdview = remove_noise_matrix_again(line_on_birdview, bamlan, road_on_birdview)  # using line or line_opencv
    # print("BIRD VIEW SHAPE: ", road_on_birdview.shape,line_on_birdview.shape)
    # road_on_birdview = remove_noise_matrix(line_on_birdview,bamlan,road_on_birdview)
   
    # if bamlan == 1 or bamlan == -1: # Remove noise by get biggest components
    #     # road_on_birdview = remove_noise_matrix_again(line_on_birdview, bamlan, road_on_birdview)  # using line or line_opencv
    #     road_on_birdview =  road_on_birdview.astype('uint8')
    #     nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(road_on_birdview, connectivity=4)  # We can modify 4 or 8
    #     #print("NUMBER OF COMPONENTS: ", nb_components)
    #     max_label = find_biggest_components(nb_components, output, stats)
    #     road_on_birdview = np.zeros((144,100))
    #     road_on_birdview[output == max_label] = 255   
   
    # kernel = np.ones((5,5),np.uint8)
    # road_on_birdview = cv2.morphologyEx(road_on_birdview, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("Biggest",road_on_birdview)
  
    # cv2.imshow("Line 3",line)
    #status = getCentroidRoadLine(road,line,bamlan)
    #print("STATUS: ",status)
    # Check centroid of road with centroid of line whether not smaller

    

    
    # Remove opposite lane
    '''
    This line we remove opposite lane which we not be in there.
    |         |        |
    |         |        |
    | Remove  |  Car   |
    |  Lane   |        |
    |         |        |
    '''
    
    # print("ROAD NHU NAO: ",road_on_birdview.shape)
    # road_on_birdview = remove_lane_lines(road_on_birdview, bamlan)  # Shape (144,100)
    # #road = cv2.resize(road,(144,144))
    # # print("ROAD_SHAPE", road_on_birdview.shape)
    
    # road_on_birdview = cv2.resize(road_on_birdview,(144,144))
    # cv2.imshow("Final Road",road_on_birdview)
    
    # road_on_birdview = road
    ##############################
    
    ################################# COMPUTE ANGLE_DEGREE ###########################


    return road_on_birdview,line_counting
    
if __name__ == '__main__':

    ######################### Initialize Method ##############################
    model = segmentation()
    read = read_input()
    bamlan = 0
    sign_id = 5
    ##########################################################################

    ######################### Initialize Topic ###############################

    ##########################################################################

    ############################# START ######################################
    img = cv2.imread('/media/goodgame/3332-323312/Goodgame/Data_22022020/goodgame_raw_1967.png')
    
    cv2.imshow("Frame",img)
    if img is not None:
        pre_processing_in = pre_processing(img)
    try:
        ################ MODEL PREDICT #################################
        single_layer = model.predict(img)
        road_on_birdview,line_counting = process(single_layer,cut_road=True)
        ################################################################

        angle_degree = compute_centroid(road_on_birdview)
        
        
        ################ PUBLISH ANGLE TO CAR #############################
        ###################################################################
        ''' Goc duong re trai, goc am re phai'''
        
        # if bamlan == 1 and angle_degree > 10: 
        #     print("DANG O LAN PHAI, KHUC CUA CAN RE TRAI, RE HET TOC LUC ")
        #     angle_degree = angle_degree * 1.8 #60
        # if bamlan == 1 and angle_degree < -10:
        #     print("DANG O LAN PHAI, KHUC CUA CAN RE PHAI, RE HET TOC LUC")
        #     angle_degree = angle_degree * 1.8 #-60
        # if bamlan == -1 and angle_degree > 10:
        #     print("DANG O LAN TRAI, KHUC CUA CAN RE TRAI, RE HET TOC LUC")
        #     angle_degree = angle_degree * 1.8 #60
        # if bamlan == -1 and angle_degree < -10:
        #     print("DANG O LAN TRAI, KHUC CUA CAN RE PHAI, RE HET TOC LUC")
        #     angle_degree = angle_degree * 1.8 #-60
        # if bamlan == 0 and angle_degree > 10:
        #     angle_degree = angle_degree * 1.8
        # if bamlan == 0 and angle_degree < -10:
        #     angle_degree = angle_degree * 1.8
                                
        ###################################################################
        
        print("ANGLE: ",angle_degree)
        
        if cv2.waitKey(0) == ord('q'):
            print("pressed q")
    except Exception as e:
        print(str(e))

    # cv2.destroyAllWindows()
    

    
