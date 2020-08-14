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

class segment:
    def __init__(self):

        K.clear_session() # Clear previous models from memory.
        
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
        
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.graph = tf.get_default_graph()
        
        f = gfile.FastGFile("/home/goodgame/catkin_ws/src/fptu_architecture/scripts/fptu/Handle/Utils/Tensor_RT Models/DL_FL_4_Class.pb", 'rb')
        
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
        # global count_cnn
        '''Function to decode encoded mask labels
            Inputs: 
                onehot - one hot encoded image matrix (height x width x num_classes)
                colormap - dictionary of color to label id
            Output: Decoded RGB image (height x width x 3) 
        '''
        single_layer = np.argmax(onehot, axis=-1)

        if count_for_sign == 0:
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
            print("Sign Density:",sign_density)
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
            print("CENTROIDS FROM; ",centroids)
            self.sign_pub.publish(centroids)
            # img_cnn = img_cnn[int(centroid_y_sign) - 15 : int(centroid_y_sign) + 15, int(centroid_x_sign) - 15 : int(centroid_x_sign) +15]
            # cv2.imwrite('/media/goodgame/3332-32336/CNN/image/Anh_pre_Cnn'+str(count_cnn)+'.png',img_cnn)
            
        
        '''
        If we want to visualize all classes on frame 
        Uncomment these line
        Of course, when we visualize frame which 'll make our algorithm slower (decrease fps)
        '''"""  """
        ############################
        # output = np.zeros( onehot.shape[:2]+(3,) )
        # for k in colormap.keys():
        #   output[single_layer==k] = colormap[k]
        # cv2.imshow("Output",output)
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
        road = remove_noise_matrix(line,bamlan,road) # using line or line_opencv

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

        road = remove_lane_lines(road,bamlan) # Shape (64,144)
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
        return sign_density,np.uint8(road),road_on_birdview
        
        
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

''' 
Method for this?
'''
def lidar_callback(lidar_data):
    global lidar_bool
    #print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEe")
    #print(lidar_data.data)
    lidar_bool = lidar_data.data
    rospy.logerr("TRANH XE")
''' 
Method for this?
'''
def stop_avoid_callback(stop_avoid_data):
    global stop_avoid_bool
    stop_avoid_bool= stop_avoid_data.data
    rospy.logwarn("DA NHAN TIN NHAN")
''' 
Method for this?
'''
def sign_time_callback(sign_time_data):
    # print("CALLBACK HEREEEEEE")
    # global sign_time_bool
    # sign_time_bool = sign_time_data.data  
    rospy.logwarn("DA NHAN TIN NHAN")
    global sign_datvu
    sign_datvu = sign_time_data.data
def sign_time_callback_1(sign_time_data_1):
    # print("CALLBACK HEREEEEEE")
    # global sign_time_bool
    # sign_time_bool = sign_time_data.data  
    rospy.logwarn("DA NHAN TIN NHAN")
    global sign_datvu_1
    sign_datvu_1 = sign_time_data_1.data
def sign_time_callback_2(sign_time_data_2):
    #print("CALLBACK HEREEEEEE")
    # global sign_time_bool
    # sign_time_bool = sign_time_data.data  
    rospy.logwarn("Received message from avoid traffic sign!!!")
    global sign_datvu_2
    sign_datvu_2 = sign_time_data_2.data
def sign_noleftright_callback(no_left_right_data):
    rospy.logwarn("DA NHAN TIN NHAN BIEN BAO CAM")
    global no_left_right
    no_left_right=no_left_right_data.data

def steer_bool_callback(steer_bool_data):
    global steer_bool
    steer_bool = steer_bool_data.data 
    #print("Steer Boollllllllllllllllllllllllllllllll"+ str(steer_bool_data.data) )
    rospy.loginfo("Received message sign from Classifier:     " + str(steer_bool_data.data ))

if __name__ == '__main__':

    rospy.init_node('deep_rt', anonymous=True)

    segmentation = segment()
    read = read_input() 
    #lcd = lcd_print("Goodgame",1,1) # Init LCD
    speed = rospy.Publisher("/set_speed",Float32,queue_size = 1)       
    angle_car = rospy.Publisher("/set_angle",Float32,queue_size = 1) 

    lidar_detection = rospy.Subscriber("/lidar_detection",
                            Bool,
                            lidar_callback,
                            queue_size=1) 
    stop_avoid_obstacles = rospy.Subscriber("/stop_avoid_obstacles",
                            Bool,
                            stop_avoid_callback,
                            queue_size=1) 
    
    # sign_time:  bien stop
    # sign_time_1:  bien re
    # sign_time_2:  bien cam
    sign_time =rospy.Subscriber("/sign_time",
                            Bool,
                            sign_time_callback,
                            queue_size=1)
    noleft_noright= rospy.Subscriber("/sign_noleft_noright",
                                    Bool,
                                    sign_noleftright_callback,
                                    queue_size=1)

    steer_bool_subscribe = rospy.Subscriber("/traffic_sign_id",
                            Int8,
                            steer_bool_callback,
                            queue_size=1)

    sign_time_1 =rospy.Subscriber("/sign_time_1",
                        Bool,
                        sign_time_callback_1,
                        queue_size=1)
    sign_time_2 =rospy.Subscriber("/sign_time_2",
                    Bool,
                    sign_time_callback_2,
                    queue_size=1)

    

     

    # Localize
    # bridge = CvBridge()
    # image_localization = rospy.Publisher("/localizaton",Image,queue_size=1)

    ##################
    bamlan = 0      
    count = 0
    lidar_bool = False
    stop_avoid_bool = False
    sign_time_bool = False
    no_left_right = False
    
    sign_datvu = False
    sign_datvu_1=False
    sign_datvu_2= False

    angle = Float32()

    count_frames = 0
    
    sign_bool = 0
    steer_bool = 94


    count_noleft = 0
    count_noright = 0 
    count_sign=0
    break_sign = 0
    count_noleft_noright = 0 


    count_for_sign = 0
    count_raw = 0
    # count_cnn = 0
    
    ''' We must define all sign_bool sign mean
    sign_bool : 0 --> 
    sign_bol : 55 -->  DinhTra put here
    '''
    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('/media/goodgame/3332-32336/CNN/video/output_cnn.mp4',fourcc, 30.0, (144,144))
    if count_for_sign == 0:
        print("JIFJKALJDKLAWJDLAWJDKLAWD ")
        road_fake = np.zeros((100,128))
        road_fake_1 = speed_up_no(road_fake)

    while not rospy.is_shutdown():
        start_time = time.time()
        if read.frame is not None:
            #cv2.imshow("Deep handle",read.frame)
            pre_processing_in = pre_processing(read.frame)
            #read.btn.led_send_message(True)

            # image_localization.publish(bridge.cv2_to_imgmsg(read.frame, "bgr8"))
   
            # cv2.imshow("Frame",read.frame)
            #out.write(read.frame)
    

            try:
                sign_density,frame_segment,road_on_birdview = segmentation.predict(read.frame)
                # out.write(frame_segment)
                # print(frame_segment.shape)
                # cv2.imshow("Segment",frame_segment)
                # if(count_raw%5==0):
                #     cv2.imwrite('/media/goodgame/3332-32336/CNN/image_raw_2805/ImageSegment'+str(count_raw)+'.png',frame_segment)
                # count_raw += 1
                # print("DAYYYYYYYYYYYYYYYYYYYYY NEEEEEEEEE",no_left_right)
                
                # centroid_x,centroid_y = compute_centroid(road_on_birdview)
                
                ''' 
                This comment to visualize frame to see
                '''
                ###############################
                # frame_segment = cv2.resize(frame_segment,(144,144))
                # cv2.circle(frame_segment, (centroid_x, centroid_y), 5, (144, 0, 0), -1)
                # cv2.circle(frame_segment, (centroid_left, centroid_y), 5, (144, 0, 0), -1)
                # cv2.circle(frame_segment, (centroid_right, centroid_y), 5, (144, 0, 0), -1)
                # cv2.imshow("Segment",frame_segment)
                # img = cv2.resize(read.frame,(320,240))
                # output = cv2.resize(output,(320,240))
                # img_v = cv2.hconcat:([img,output])
                # cv2.imshow("Frame",img_v)
                # url = '/media/goodgame/3332-32332/Goodgame/Data_0203_sign/goodgame_data_sign' + str(count) + '.png'
                # cv2.imwrite(url,read.frame)
                ###############################
                
                if lidar_bool:
                    bamlan = -1
                if stop_avoid_bool:
                    bamlan = 0
                    stop_avoid_bool = False
                    lidar_bool = False
                """
                buoc nay de fix cung gia tri sign_bool lai vi steer_bool laf bieenr baos taij 1 thoi diem co dinh , can lam nhu the nay de bao hieu cho object tiep theo 
                """
                if steer_bool == 0: #day la stop
                    sign_bool = -1
                if steer_bool == 1: #day la right
                    sign_bool = 1
                    speed.publish(15)
                if steer_bool == 2:# day la left
                    sign_bool = 2
                '''TRA DINH +HUY PHAN LAM BIEN CAM '''
                if steer_bool == 4: # noleft
                    sign_bool = 4
                    speed.publish(15)
                if steer_bool == 5:#no right
                    sign_bool = 5
                    speed.publish(15)
                if steer_bool == -1: #Reset all sign bool
                    sign_bool = 11
                
                '''
                sign_bool= -1: stop
                sign_bool = 1: right
                sign_bool = 2: left
                sign_bool = 4: noleft
                sign_bool = 5: noright
                '''
                '''
                kich hoat bien bao dua vao lidar
                stop: 99
                turn right: 107
                turn left: 701
                noleft:69
                noright:96
                '''
                #if steer_bool !=-1:
                #    sign_bool=3

                '''TRA DINH LAM BIEN BAO CAM O DAY'''
                # if steer_bool ==4:#no right
                #     road_on_birdview=noright(road_on_birdview)
                    
                    
                # if steer_bool == 3 :
                #     road_on_birdview=noleft(road_on_birdview)
                    
                
                '''
                Edit sign_datvu to other name
                '''
                #rospy.logwarn("BOOOLENANN        " +str(sign_datvu)) # Edit content
                #rospy.logwarn("SIGN:\t" +str(sign_bool) + "STEER:\t" + str(steer_bool))
                #print("HELOOOOOOOOOOOOOOOOOOO",sign_datvu,sign_datvu_1,sign_datvu_2)# Edit content
                '''
                sign_datvu: stop
                sign_datvu_1: bien re
                sign_datvu_2: bien cam 
                '''

                #if sign_datvu_1 == True:
                #    if sign_bool == 1:
                #        rospy.logerr("TURN RIGHT!!!")
                #        sign_bool = 107
                #        #angle_car.publish(60)
                #    if sign_bool == 2:
                #        rospy.logerr("TURN LEFT!!!!")
                #        sign_bool = 701
                #    sign_datvu_1=False


                
                if sign_datvu_2 == True:              
                    if sign_bool == 1:# turn right
                        time_tra = time.time()
                        while time.time() - time_tra <= 0.8 :
                            angle_car.publish(-60)
                        bamlan = 1
                        speed.publish(20)
                        steer_bool = -1
                    # if sign_bool == 3:
                    #     bamlan = 1
                    if sign_bool ==4:#noleft
                        sign_bool = 69
                        if speed_up_no(road_on_birdview) > 100:# di thang
                            time_nl_t=time.time()
                            while time.time()- time_nl_t <=0.6:
                                #print(time.time() - time_t)
                                angle_car.publish(0)
                                #print("fuckkkkkkkk")
                                rospy.info("TURN STRAIGHT!!!")
                            #sign_bool= 66
                            steer_bool = -1
                        else:#re phai
                            # print("HERERJEIRJRJKWQNJKASBDJKASDBAHJDHJWAHJD213123213")
                            print("TURN RIGHT on NOLEFT!!!")
                            time_nl_r=time.time()
                            while time.time() - time_nl_r <=1:
                                #print("TURN RIGHT \t",time.time() - time_nl_r)
                                angle_car.publish(-60)
                                #print("fuck")
                            #sign_bool= 66
                            speed.publish(20)
                            steer_bool = -1
                    # if sign_bool == 5:#no right
                    #     sign_bool = 1196
                    #     if speed_up_no(road_on_birdview) > 110:# di thang
                    #         time_nl_t=time.time()
                    #         while time.time()- time_nl_t <=0.6:
                    #             #print(time.time() - time_t)
                    #             angle_car.publish(0)
                    #             #print("fuckkkkkkkk")
                    #             rospy.info("TURN STRAIGHT!!!")
                    #         #sign_bool= 66
                    #         steer_bool = -1
                    #     else:#re phai
                    #         # print("HERERJEIRJRJKWQNJKASBDJKASDBAHJDHJWAHJD213123213")
                    #         bamlan = 0
                    #         print("TURN RIGHT on NORIGHT!!!")
                    #         time_nl_r=time.time()
                    #         while time.time() - time_nl_r <=1:
                    #             #print("TURN RIGHT \t",time.time() - time_nl_r)
                    #             angle_car.publish(60)
                    #             #print("fuck")
                    #         #sign_bool= 66
                    #         speed.publish(20)
                    #         steer_bool = -1                       
                    sign_datvu_2=False
                
                #if sign_datvu == True:
                #    #rospy.logerr("hellobabeie")
                #    if sign_bool == -1:
                #        rospy.logerr("STOP HERE!!!")
                #        #sign_datvu = False
                #        sign_bool = 99   
                #        #speed.publish(0)

                #    sign_datvu = False


                # Huy Phan 01/06 TEST
                
                #if sign_bool == 69:
                #    if 0 <= sign_density < 50:
                #        time_nl_turn=time.time()
                #        while time.time()- time_nl_turn <= 0.8:
                #             #print(time.time() - time_t)
                #            angle_car.publish(-60)
                #        sign_bool = 66
                #        steer_bool = -1

                


                #timeh=time.time()
                #while time.time()-timeh < 3:
                #    cv2.imshow("hello ", read.frame)


                # if sign_bool == 3:
                #     count_sign = 5
                # else:
                #     count_sign = count_sign - 1
                # if (no_left_right == True or count_noleft > 0) and sign_bool == 3:
                #     road_on_birdview = noleft(road_on_birdview)
                #     print("TAO DANG CAT DUONG NE CHUNG MAY ")
                #     if no_left_right == True:
                #         count_noleft = 5
                #         break_sign=1 
                        
                # else:
                #     if count_sign <= 0:
                #         sign_bool = 66
                #     count_noleft = count_noleft-1 
                # if (no_left_right == True or count_noright> 0) and sign_bool == 4:
                #     road_on_birdview = noright(road_on_birdview)
                #     if no_left_right == True:
                #         count_noright = 5 
                # else:
                #     print("heREEEEEEEFUCKKKKKKKKKKKKKKKK", count_sign)
                #     if count_sign <=0:
                #         sign_bool= 66
                #     count_noright= count_noright - 1
                # print("hereEEEEEEEEEEEEEEEEEEEEEEEEE", "noleft", count_noleft, "noright", count_noright)

                """
                HERE
                """
                # if sign_bool == 3:
                #     count_sign = 5
                # else:
                #     count_sign = count_sign - 1
                # if (no_left_right == True or count_noleft_noright > 0):
                #     if sign_bool == 3:
                #         road_on_birdview = noleft(road_on_birdview)
                #         print("TAO DANG CAT DUONG NE CHUNG MAY ")
                #         if no_left_right == True:
                #             count_noleft_noright = 5 
                #     if sign_bool == 4:
                #         road_on_birdview = noright(road_on_birdview)
                #         if no_left_right == True:
                #             count_noleft_noright = 5 
                # else:
                #     print("heREEEEEEEFUCKKKKKKKKKKKKKKKK", count_sign)
                #     if count_sign <=0:
                #         sign_bool= 66
                #     count_noleft_noright= count_noleft_noright - 1
                # print("hereEEEEEEEEEEEEEEEEEEEEEEEEE", "noleft", count_noleft, "noright", count_noright)
                '''
                End
                '''

                # cv2.imshow("hello",road_on_birdview)

                # cv2.imshow("hello",road_on_birdview_T)
                # print("day neeeeeeeeeeeeeeeeeee",sum(road_on_birdview_T[50]/255))
                # start_time_t = time.time()
                # road_on_birdview_T=road_on_birdview.T
                # print(sum(road_on_birdview_T[50]/255))
                # print("THOI GIAN TEST: ",time.time()-start_time_t) 
                #print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",sign_bool)
                '''
                kich hoat bien bao dua vao lidar
                stop: 99
                turn right: 107
                turn left: 701
                noleft:69
                noright:96
                
                '''
                #if sign_bool ==4:
                #    road_on_birdview=noleft(road_on_birdview)
                #if sign_bool ==5:
                #    road_on_birdview=noright(road_on_birdview)
                #print("DENSITY OF ROAD: ",speed_up_no(road_on_birdview))
                if sign_bool == 169:#noleft
                    if speed_up_no(road_on_birdview) > 100:# di thang
                        time_nl_t=time.time()
                        while time.time()- time_nl_t <=0.6:
                            #print(time.time() - time_t)
                            angle_car.publish(0)
                            #print("fuckkkkkkkk")
                            rospy.info("TURN STRAIGHT!!!")
                        #sign_bool= 66
                        steer_bool = -1
                    else:#re phai
                        # print("HERERJEIRJRJKWQNJKASBDJKASDBAHJDHJWAHJD213123213")
                        print("TURN RIGHT on NOLEFT!!!")
                        time_nl_r=time.time()
                        while time.time() - time_nl_r <=1:
                            print("TURN RIGHT \t",time.time() - time_nl_r)
                            angle_car.publish(-60)
                            #print("fuck")
                        #sign_bool= 66
                        steer_bool = -1
                if sign_bool == 96:# no right
                    if speed_up_no(road_on_birdview) > 80:# di thang
                        time_nr_t= time.time()
                        while time.time()-time_nr_t <= 0.6:
                            angle_car.publish(0)
                        steer_bool = -1
                    else:# re trai
                        time_nr_l = time.time()
                        while time.time()-time_nr_l <= 1:
                            angle_car.publish(60)
                        steer_bool = -1
                if sign_bool == 99:#stop
                    time_t = time.time()
                    while time.time() - time_t <= 3:
                        print(time.time() - time_t)
                        speed.publish(0)
                    sign_bool = 66
                    bamlan = 0
                    steer_bool = -1
                if sign_bool == 107:# turn right
                    time_tra = time.time()
                    while time.time() - time_tra <= 0.8 :
                        angle_car.publish(-60)
                    #sign_bool=66
                    bamlan = 1
                    steer_bool = -1
                if sign_bool == 701:#turn left
                    time_tl = time.time()
                    while time.time()-time_tl <= 0.8:
                        angle_car.publish(60)
                    bamlan=1
                    steer_bool=-1

                if read.btn.bt1_bool:
                    sign_bool = 66
                    #speed.publish(15)
                if read.btn.bt2_bool:
                    #speed.publish(0)
                    sign_bool = 55
                if read.btn.bt3_bool:
                    bamlan = -1
                if read.btn.bt4_bool:
                    bamlan = 1  

                # if sign_bool == 0:
                #     speed.publish(15)
                if sign_bool == 66:
                    speed.publish(20)
                if sign_bool == 55:
                    speed.publish(0)
                #if sign_bool == 314:
                #    angle_car.publish(0)
                ''' We will to compute angle to steer car in here

                '''
                # cv2.imshow("hello", road_on_birdview)
                centroid_x,centroid_y = compute_centroid(road_on_birdview)

                angle_degree = angle_calculator(centroid_x,centroid_y) # Call angle_calculator method in speed_up.py to use numba function
                # if(abs(angle_degree) > 18):
                #    road = road[85:,:]
                # else:
                #    road = road[64:,:]

                #print("Goc lai la ", angle_degree)
                rospy.logwarn("Steer Angle:  " + str(angle_degree))

                angle.data = angle_degree #Fuzzy(centroid_x - 64) #angle_degree

                angle_car.publish(angle)       

                
                #lcd.update_message("Hello",1,1)

            except Exception as e:
                print(str(e))
        fps = 1 // (time.time() - start_time)
        rospy.loginfo("FPS IN RGB FRAME: " + str(fps))
        read.btn.led_send_message(True)
        cv2.waitKey(1)
        # count_cnn += 1

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    # cv2.destroyAllWindows()
