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
from fptu.Handle.speed_up import angle_calculator,sign_detection,remove_noise_matrix,remove_lane_lines,compute_centroid,noright,noleft,compute_sign,find_biggest_components,speed_up_no
from fptu.Reader.get_frames import read_input
from fptu.Reader.lcd_publish import lcd_print
from fptu.Preprocessing.preprocessing import pre_processing
from fptu.Controller.error import PID,Fuzzy
from cv_bridge import CvBridge, CvBridgeError

# Add new
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

# Add new 05/06
import roslaunch

class segment:
    def __init__(self):

        K.clear_session() # Clear previous models from memory.
        
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
        
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.graph = tf.get_default_graph()
        
        f = gfile.FastGFile("/home/goodgame/catkin_ws/src/fptu_architecture/scripts/fptu/Handle/Utils/Tensor_RT Models/GG-Haianh-efficientnetb3_CT.pb", 'rb')
        
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

        self.label_codes, self.label_names = zip(*[self.parse_code(l) for l in open("/home/goodgame/catkin_ws/src/fptu_architecture/scripts/fptu/Model/PSPNET_4classes/label_colors_4classes.txt")])
        
        self.label_codes, self.label_names = list(self.label_codes), list(self.label_names)
        
        print(self.label_codes, self.label_names)

        self.code2id = {v:k for k,v in enumerate(self.label_codes)}
        self.id2code = {k:v for k,v in enumerate(self.label_codes)}

        self.name2id = {v:k for k,v in enumerate(self.label_names)}
        self.id2name = {k:v for k,v in enumerate(self.label_names)}

        # Publish centroid of traffic sign to classifiers.py
        self.sign_pub = rospy.Publisher('/localization', numpy_msg(Floats),queue_size=1)
        
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

        global count_for_sign

        global count_RGB

        global sign_id
        # global count_cnn
        '''Function to decode encoded mask labels
            Inputs: 
                onehot - one hot encoded image matrix (height x width x num_classes)
                colormap - dictionary of color to label id
            Output: Decoded RGB image (height x width x 3) 
        '''
        single_layer = np.argmax(onehot, axis=-1)

        if count_for_sign == 0:
            road_fake = np.zeros((100,128))
            road_fake_1 = speed_up_no(road_fake)
            print("Complie the algorithm on numba the first time!!!")
            sign = np.zeros((144,144))
            sign[single_layer == 0] = 255
            # Find biggest components
            sign = sign.astype('uint8')
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(sign, connectivity=4) # We can modify 4 or 8
            max_label = find_biggest_components(nb_components,output,stats)
            sign = np.zeros(output.shape)
            sign[output == max_label] = 255
            centroid_x_sign,centroid_y_sign = compute_sign(sign)
            centroids = np.array([111,72,3000,0], dtype=np.float32)
            self.sign_pub.publish(centroids)
            count_for_sign += 1

        # noleft = len(np.where(single_layer == [8])[0]) # Get density of No left sign
        # noright = len(np.where(single_layer==[7])[0]) 
        # stop = len(np.where(single_layer == [4])[0])
        # straight = len(np.where(single_layer == [10])[0])
        # turnleft = len(np.where(single_layer == [6])[0])
        # turnright = len(np.where(single_layer == [9])[0])
        '''
        We will call sign_detection on speed_up.py to make decision which object is detected is 
        But we need improve new algorithm in here
        At present, we don't have new idea for this
        '''
        # New algorithm to detect sign by Dat Vu add on 16/05/2020
        ##########################################################
        sign_density = len(np.where(single_layer == [3])[0])
        #print("HEJRLEJLRJWELRKWERWER: ",sign_density)
        if sign_density > 200:
            #print("Sign Density:",sign_density)
            sign = np.zeros((144,144))
            sign[single_layer == 3] = 255
            # Find biggest components
            sign = sign.astype('uint8')
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(sign, connectivity=4) # We can modify 4 or 8
            max_label = find_biggest_components(nb_components,output,stats)
            sign = np.zeros(output.shape)
            sign[output == max_label] = 255
            # cv2.imshow("Test",sign)
            #
            centroid_x_sign,centroid_y_sign = compute_sign(sign)
            centroids = np.array([centroid_x_sign,centroid_y_sign,sign_density,1], dtype=np.float32)
            #print("CENTROIDS FROM; ",centroids)
            self.sign_pub.publish(centroids)
            # img_cnn = img_cnn[int(centroid_y_sign) - 15 : int(centroid_y_sign) + 15, int(centroid_x_sign) - 15 : int(centroid_x_sign) +15]
            # cv2.imwrite('/media/goodgame/3332-32336/CNN/image/Anh_pre_Cnn'+str(count_cnn)+'.png',img_cnn)
            
        
        '''
        If we want to visualize all classes on frame 
        Uncomment these line
        Of course, when we visualize frame which 'll make our algorithm slower (decrease fps)
        '''"""  """
        ############################
        output = np.zeros( onehot.shape[:2]+(3,) )
        for k in colormap.keys():
          output[single_layer==k] = colormap[k]
        #cv2.imshow("Output",output)
        #out.write(output)
        ############################

        ''' 
        When we want to visualize only one class
        Uncomment these line
        '''
        ############################
        line = np.zeros(onehot.shape[:2])
        line[single_layer==1] = 255
        line = cv2.resize(line,(128,128))
        line = line[64:,:]

        #cv2.imshow("line deeeplearning",line)

        # line_1= np.zeros(onehot.shape[:2])
        # line_1[single_layer==7] = 255
        # cv2.imshow("right",line_1)
        ############################

        # Dat Vu add the code on 28/04
        ################################
        road = np.zeros((144,144))
        road[single_layer==0] = 255 # Get road class equiv class 0
        road = cv2.resize(road, (128,128))
        # cv2.imwrite('/media/goodgame/3332-32338/Data_Full/Img_'+str(count_RGB)+'.png',road)
        # count_RGB+=1
        road = road[64:,:] # Split half of image -> 144x144 convert to 64x144
        # cv2.imshow("road_before", road)
        #cv2.imshow("line_opencv", line)
        # centroid_x,centroid_y = compute_centroid(road_on_birdview)
        # angle_degree = angle_calculator(centroid_x,centroid_y)
        # if(abs(angle_degree) > 18):
        #      road = road[85:,:]
        # else:
        #     road = road[64:,:]
        #Hai Anh add for check sign. You can cmt it
        #####################################################
        # Stop:3 ; Left:4 ; Noright:5 ; Noleft:6 ; Right:7 ; Straight:8
        
        # noleft = np.zeros((144,144))
        # noleft[single_layer==6] = 255
        # cv2.imshow("no_left",noleft)

        # noright = np.zeros((144,144))
        # noright[single_layer==5] = 255
        # cv2.imshow("no_right",noright)

        # stop = np.zeros((144,144))
        # stop[single_layer==3] = 255
        # cv2.imshow("stop",stop)

        # left = np.zeros((144,144))
        # left[single_layer==4] = 255
        # cv2.imshow("left",left)

        # right = np.zeros((144,144))
        # right[single_layer==7] = 255
        # cv2.imshow("right",right)

        # straight = np.zeros((144,144))
        # straight[single_layer==8] = 255
        # cv2.imshow("straight",straight)
        #############################################################

        # Opening algorithm to reduce noise in two lane lines.
        #kernel = np.ones((3,3),np.uint8)
        #line = cv2.morphologyEx(line_opencv, cv2.MORPH_OPEN, kernel)
        ''' 
        In this line, we need to remove noise road which is not inside line of road
        We can use line or line_opencv 
        line : after processing line_opencv by opening algorithm (not recommended) but this can be useful in few cases
        line_opencv : raw line which we get from handling line road by HSV Color (Opencv)
         ___________|           |       |
        |   Noise   |           |       |
        |___________|   Car     |       |
                    |           |       |
        
        '''
        # Comment this line if model don't have noise  
        road,total_left_line,total_right_line = remove_noise_matrix(line,bamlan,road) # using line or line_opencv

        # cv2.imshow("road after remove noise",road)
        
        # Remove opposite lane
        '''
        This line we remove opposite lane which we not be in there.
        |         |        |
        |         |        |
        | Remove  |  Car   |
        |  Lane   |        |
        |         |        |
        '''

        road = remove_lane_lines(road,bamlan,sign_id) # Shape (64,144)
        #cv2.imshow("Road",road)
        ###############################
        '''
        tradinh
        After remove oppsite lane we will bird-view road
        Firstly, we need resize image to 144x144
        Secondly, we use open-cv algorithm to warp image
        Finally, we will resize image again to 100x144 (Tra Dinh recommended here)
        '''
        transform_matrix = pre_processing_in.perspective_transform()
        #cv2.imshow("Binary",binary)
        road_resized = cv2.resize(road,(128,128))
        road_on_birdview = pre_processing_in.birdView(road_resized,transform_matrix['M'])
        #print(warped_image.shape)
        road_on_birdview = cv2.resize(road_on_birdview,(100,128)) 
        # cv2.imshow("img",road_on_birdview)
        #cv2.imshow("IMAGe",warped_image)

        # Dat Vu add this line on 28/04 to compute centroid of road
        #centroid_x,centroid_y = compute_centroid(road_on_birdview)
        # print("Centroid X = ", centroid_x)
        # print("Centroid Y = ", centroid_y)
        return sign_density,np.uint8(output),road_on_birdview,total_left_line,total_right_line
        
        
    def predict(self,frame):
        with self.sess.as_default():
            with self.graph.as_default():

                img = cv2.resize(frame,(144,144))
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
               
                # line_opencv = mask[64:,:] # We want to get half of image below 
                ############################################
                image = (img[...,::-1].astype(np.float32)) / 255.0
                
                image = np.reshape(image, (1, 144, 144, 3))

                softmax_tensor = self.sess.graph.get_tensor_by_name('import/softmax/truediv:0')
                
                predict_one = self.sess.run(softmax_tensor, {'import/input_1:0': np.array(image)})
                
                # In here, I've just add a parameter into onehot_to_rgb method.
                # Before the method only have two parameters (predict_one[0], self.id2code)
                image = self.onehot_to_rgb(predict_one[0],self.id2code)
                
                #Test for spped up

                #image = onehot_to_rgb_speedup(predict_one[0])

                #cv2.imshow("Prediction",image) # If you use local machine cv2.imshow instead of cv2_imshow  
                
                return image      


if __name__ == '__main__':

    rospy.init_node('log_deep_rt', anonymous=True)

    segmentation = segment()
    read = read_input() 
    lcd = lcd_print("Goodgame",1,1) # Init LCD
    speed = rospy.Publisher("/set_speed",Float32,queue_size = 1)       
    angle_car = rospy.Publisher("/set_angle",Float32,queue_size = 1) 
    angle = Float32()
    
    count_for_sign = 0
    bamlan = 0
    sign_id = 0
    count_segment_data = 0
    ###########################################
    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('/media/goodgame/3332-32338/Log/output.mp4',fourcc, 30.0, (144,144))

    while not rospy.is_shutdown():
        start_time = time.time()
        if read.frame is not None:
            pre_processing_in = pre_processing(read.frame)
    
            try:
                sign_density,frame_segment,road_on_birdview,total_left_line,total_right_line = segmentation.predict(read.frame)

                if read.btn.bt1_bool:
                    speed.publish(20)
                    #flag_angle = 1
                    first_btn = True
                if read.btn.bt2_bool:
                    sign_id = -1  
                    speed.publish(0)
                    #flag_angle = -1
                    first_btn = False
                   #Reset sign to -1
                    speed.publish(0)
                if read.btn.bt3_bool:
                    if bamlan != 0:
                        bamlan = -bamlan
                    else:
                        bamlan += 1
                        
                if read.btn.bt4_bool:
                    read.btn.led_send_message(False)
                    bamlan = 0
                    count_for_sign = 0
                    sign_id = -1
                    speed.publish(0)
                    lcd.update_message("Restarting...",0,0)
                    lcd.clear()
                    #launch.start()

                #Log data segment + raw frame
                if(count_segment_data%5 == 0):
                    cv2.imwrite('/media/goodgame/3332-32338/Log/Visual_data/mask/Data_seg_mask_1607_'+str(count_segment_data)+'_L.png',frame_segment)
                    # cv2.imwrite('/media/goodgame/3332-32338/Log/Visual_data/frame/Data_seg_frame_1607_'+str(count_segment_data)+'.png',read.frame)
                count_segment_data +=1
                #cv2.imshow("RGB",read.frame)
                # if sign_id == -1: # Khong co bien bao thi di voi toc do 20
                #     speed.publish(20)

                ''' We will to compute angle to steer car in here

                '''
                angle_degree = compute_centroid(road_on_birdview)

                angle.data = angle_degree #Fuzzy(centroid_x - 64) #angle_degree

                angle_car.publish(angle)       
                fps = int(1 // (time.time() - start_time))
                #rospy.loginfo("FPS IN RGB FRAME: " + str(fps))
                read.btn.led_send_message(True)
                if read.btn.ss2_status:
                    first_time_display = True
                if first_time_display:
                    lcd.update_message("Running...",0,0)
                    first_time_display = False
                if bamlan == 1:
                    lcd.update_message("Mode Right" + "  FPS " +str(fps) +"    ",0,1)
                elif bamlan == -1:
                    lcd.update_message("Mode Left "+ "  FPS " + str(fps) +"    ",0,1)
                elif bamlan == 0:
                    lcd.update_message("Mode Mid"+ "  FPS " +str(fps) +"    ",0,1)
                if sign_id == -1:
                    lcd.update_message("Angle " + str(round(angle_degree,1)) + " .Speed 20",0,3)
                elif sign_id != -1:
                    lcd.update_message("Angle " + str(round(angle_degree,1)) + " .Speed 15",0,3)
                cv2.waitKey(1)
            except Exception as e:
                print(str(e))

    #out.release()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    # cv2.destroyAllWindows()
