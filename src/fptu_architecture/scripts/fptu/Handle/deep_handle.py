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
        
        f = gfile.FastGFile("/Users/datvu/Documents/GitHub/Digital-Race/src/fptu_architecture/scripts/fptu/Model/pspnet_rb.pb", 'rb')
        
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
        
        self.count_objects = rospy.Publisher('/count', numpy_msg(Floats),queue_size=1)

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

        global centroid_x_sign
        global centroid_y_sign
        global centroid_x_before
        global centroid_y_before
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
            centroid_x_signs,centroid_y_signs = compute_sign(sign)
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
        # Assign old centroids = new centroids 
        centroid_x_before = centroid_x_sign
        centroid_y_before = centroid_y_sign
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
            centroid_x_sign,centroid_y_sign = compute_sign(sign)
            centroids = np.array([centroid_x_sign,centroid_y_sign,sign_density,1], dtype=np.float32)
            #print("CENTROIDS FROM; ",centroids)
            self.sign_pub.publish(centroids)
            # count_centroids = np.array([centroid_x_before,centroid_y_before,centroid_x_sign,centroid_y_sign],dtype=np.float32)
            # self.count_objects.publish(count_centroids)
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
        
        # cv2.imshow("Line before processing",line)
        # line_opencv = cv2.resize(line_opencv,(128,128))
        # line_opencv = line_opencv[64:,:]

        # for i in range(0,64):
        #     for j in range(0,128):
        #         if line_opencv[i][j] == 255:
        #             line[i][j] = 255
        
        # cv2.imshow("Line after processing: ",line)

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
        return sign_density,np.uint8(road),road_on_birdview,total_left_line,total_right_line
        
        
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
def sign_turn_callback(sign_turn_data):
    #print("nhan tin nhan")
    global sign_turn_hello
    sign_turn_hello = sign_turn_data.data
def sign_time_passaway_callback(sign_data):
    #rospy.logwarn("Received message from avoid traffic sign!!!")
    global sign_passaway
    sign_passaway = sign_data.data

def traffic_sign_callback(traffic_sign_id):
    global sign_id
    sign_id = traffic_sign_id.data 
    rospy.loginfo("Received message sign from Classifier:     " + str(traffic_sign_id.data ))
def pass_sign_callback(pass_sign_data):
    global pass_sign_id
    pass_sign_id = pass_sign_data.data

#Day la ham giai thich dem bien cam trong do bien cam trai(signid = 4) va cam phai (signid = 5)
def count_bien_cam(sign_id, count_sign_bien_bao_cam):
    global count_bien_bao_cam
    if(sign_id == 4 or sign_id == 5):
        if(count_sign_bien_bao_cam!= 0):
            count_bien_bao_cam += 1
    return count_bien_bao_cam    

#Day la ham xac dinh flag (tuc la xac dinh trang thai cua bien dieu huong) sign_straigh = 3, sight_right = 1, sight_left = 2
def check_flag(sign_id, count_sign_bien_bao_straight, count_sign_bien_left_right, count_sign_bien_bao_stop):
    global flag_sign_dieu_huong
    if(sign_id == 3):
        if(count_sign_bien_bao_straight != 0):
            flag_sign_dieu_huong = 1 ## dieu huong di thang
    elif(sign_id == 1 or sign_id == 2):
        if(count_sign_bien_left_right != 0):
            flag_sign_dieu_huong = 0 ## dieu huong re trai hoac re phai
    elif(sign_id == 0):
        if(count_sign_bien_bao_stop):
            flag_sign_dieu_huong = 2 ###day la bien stop de reset        
    else:
       # print("HEREEEEEEE!!!!")
        flag_sign_dieu_huong = -1

    return flag_sign_dieu_huong

if __name__ == '__main__':

    flag_sign_dieu_huong = -1

    count_bien_bao_cam = 0

    count_RGB = 0

    rospy.init_node('deep_rt', anonymous=True)

    segmentation = segment()
    read = read_input() 
        
    lcd = lcd_print("Goodgame",1,1) # Init LCD
    speed = rospy.Publisher("/set_speed",Float32,queue_size = 1)       
    angle_car = rospy.Publisher("/set_angle",Float32,queue_size = 1) 
    angle = Float32()

    # lidar_detection = rospy.Subscriber("/lidar_detection",
    #                         Bool,
    #                         lidar_callback,
    #                         queue_size=1) 
    # stop_avoid_obstacles = rospy.Subscriber("/stop_avoid_obstacles",
    #                         Bool,
    #                         stop_avoid_callback,
    #                         queue_size=1) 
    pass_sign = rospy.Subscriber("/pass_sign",
                                Int8,
                                pass_sign_callback,
                                queue_size=1)

    traffic_sign_subscribe = rospy.Subscriber("/traffic_sign_id",
                            Int8,
                            traffic_sign_callback,
                            queue_size=1)

    sign_time_passaway_sub =rospy.Subscriber("/sign_time_passaway",
                    Int8,
                    sign_time_passaway_callback,
                    queue_size=1)
    
    sign_turn =rospy.Subscriber("/turn_sign",
                Int8,
                sign_turn_callback,
                queue_size=1)
    

    
    ##################
    bamlan = 0      
    lidar_bool = False
    stop_avoid_bool = False
    sign_passaway = 0
    third_btn = 0
    sign_id = -1 # Default -1
    first_time_display = False
    count_for_sign = 0
    flag_angle = -1 #day la flag cua goc lai ; 2 mode: 1-25 and 0-15
    first_btn = False
    sign_turn_hello =0
    pass_sign_id = 0
    count_sign = 0
    check = True
    count_sign_bien_bao_cam = 0
    count_sign_bien_bao_straight = 0
    count_sign_bien_left_right = 0
    count_sign_bien_bao_stop = 0
    count_f = 0
    # Dat Vu add 17/07
    centroid_x_sign = 0
    centroid_y_sign = 0
    centroid_x_before = 0
    centroid_y_before = 0
    #

    #count_raw = 0
    #######################################
    
    ###########################################
    ###### Button Four Reset roslaunch  #######
    ###########################################
    #
    #
    # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    # roslaunch.configure_logging(uuid)
    # launch = roslaunch.parent.ROSLaunchParent(uuid, ['/home/goodgame/catkin_ws/src/fptu_architecture/launch/server.launch'])
    #
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
                count_bien_bao_cam = count_bien_cam(sign_id, count_sign_bien_bao_cam)
                flag_sign_dieu_huong = check_flag(sign_id, count_sign_bien_bao_straight, count_sign_bien_left_right, count_sign_bien_bao_stop )
                #matdo = speed_up_no(road_on_birdview)

                #### Dat Vu add on 17/07

                #lcd.update_message(str(total_right_line),0,0)
                #out.write(frame_segment)

                # if sign_passaway ==1 or sign_passaway ==2:
                #     bamlan = -1
                if stop_avoid_bool:
                    bamlan = 0
                    stop_avoid_bool = False
                    lidar_bool = False

                """
                buoc nay de fix cung gia tri sign_bosign_id == 3ol lai vi steer_bool laf bieenr baos taij 1 thoi diem co dinh , can lam nhu the nay de bao hieu cho object tiep theo 
                """

                    
                # if sign_id == 4:
                #     #flag_angle = 0 # Neu gap bien bao giam toc do xuong
                #     #speed.publish(15)

                # if(sign_id == 4):
                #     if(count_bien_bao_cam == 2):
                #         print("CHo nay cam re trai nen ta re phai")

                # if(sign_id == 5):
                #     if(count_bien_bao_cam == 2):
                #         print("CHo nay cam re phai nen ta re trai")   
                # #### bien cam trai(signid = 4) va cam phai (signid = 5)
                # #### sign_straight = 3, sight_right = 1, sight_left = 2   
                # if(flag_sign_dieu_huong == 1):
                #     if(sign_id == 4):
                #         if(count_bien_bao_cam == 3 or count_bien_bao_cam == 4):
                #             print("Gap bien bao straight va bien cam Cho nay di thang")
                #             #xu li thuat toan lai xe theo huong nay
                #         elif(count_bien_bao_cam == 5):
                #             print("Gap bien bao straight va bien cam Cho nay phai re phai") 
                #             #xu li thuat toan lai xe theo huong nay   
                #     elif(sign_id == 5):
                #         if(count_bien_bao_cam == 3 or count_bien_bao_cam == 4):
                #             print("Gap bien bao straight va bien cam Cho nay di thang")
                #             #xu li thuat toan lai xe theo huong nay
                #         elif(count_bien_bao_cam == 5):
                #             print("Gap bien bao straight va bien cam Cho nay re trai")    
                #             #xu li thuat toan lai xe theo huong nay

                # if(flag_sign_dieu_huong == 0):
                #     if(sign_id == 4):
                #         if(count_bien_bao_cam >= 3 and count_bien_bao_cam <= 5):
                #             print("Gap bien bao left or right va bien cam Cho nay di thang")
                #             #xu li thuat toan lai xe theo huong nay
                #     elif(sign_id == 5):
                #         if(count_bien_bao_cam >= 3 and count_bien_bao_cam <= 5):
                #             print("Gap bien bao left or right va bien cam Cho nay di thang")
                #             #xu li thuat toan lai xe theo huong nay

                # if(flag_sign_dieu_huong == 2): ######day la flag stop de reset
                #     count_bien_bao_cam = 0
                #     flag_sign_dieu_huong = -1
                
                
                # if check == True and sign_id == 4:
                #     count_sign_bien_bao_cam += 1
                #     check = False
                # if check == True and sign_id == 3:
                #     count_sign_bien_bao_straight += 1
                #     check = False
                # if check == True and (sign_id == 1 or sign_id == 2) :
                #     count_sign_bien_left_right += 1
                #     check = False
                # if check == True and (sign_id == 0) :
                #     count_sign_bien_bao_stop += 1
                #     check = False
                # if (pass_sign_id == 1):
                #     check = True
                #     sign_id = -1
                #print("DEM BIEN     ", count_sign_bien_bao_cam, count_sign_bien_bao_stop, count_sign_bien_bao_straight, count_sign_bien_left_right,flag_sign_dieu_huong)   
                                    
                # if sign_turn_hello == 1:
                #     lcd.update_message("Traffic sign!!!",0,0)

                #     if sign_id ==0:# stop 
                #         time_stop = time.time()
                #         while time.time()-time_stop < 3:
                #             speed.publish(0)
                #         speed.publish(20)
                #         sign_id = -1
                #     if sign_id == 1:# turn right
                #         print("heloo")
                #         time_tra = time.time()
                #         while time.time() -time_tra < 0.05:
                #             angle_car.publish(0)
                #         time_tra_1 = time.time()
                #         while time.time() - time_tra_1 <= 0.8 :
                #             angle_car.publish(-60)
                #         bamlan = 1
                #         speed.publish(20) #flag_angle = 1
                #         sign_id = -1 # Reset bienbao
                #     # if sign_bool == 3:
                #     #     bamlan = 1
                #     sign_turn_hello =0

                # if sign_id == 3:


                # if sign_passaway == 3:       # Vuot qua bien bao bat dau lam gi.    
                #     lcd.update_message("Traffic sign!!!",10,2)
                #     if sign_id == 4:#noleft
                #         if total_right_line < 200:# di thang
                #             print("TURN RIGHT on NOLEFT!!!")
                #             time_nl_r=time.time()
                #             while time.time() - time_nl_r <=1:
                #                 #print("TURN RIGHT \t",time.time() - time_nl_r)
                #                 angle_car.publish(-60)
                #                 #print("fuck")
                #             #sign_bool= 66
                #             speed.publish(20) #flag_angle = 1    
                #             #sign_bool= 66
                #         else:#re phai
                #             #print("HERERJEIRJRJKWQNJKASBDJKASDBAHJDHJWAHJD213123213")
                #             time_nl_t=time.time()
                #             while time.time()- time_nl_t <= 0.4:
                #                 #print(time.time() - time_t)
                #                 angle_car.publish(0)
                #                 #print("fuckkkkkkkk")
                #                 #rospy.info("TURN STRAIGHT!!!")
                #             speed.publish(20)
                 
                #         sign_id = -1

                # #### Algorithm 2 #####
                # #sign_density_before = sign_density
                # # if sign_id == 4:
                    
                # #     if sign_density > 220:
                # #             print("RE PHAI")
                # #             time_nl_turn=time.time()
                # #             while time.time()- time_nl_turn <= 1.2:
                # #                 #print(time.time() - time_t)
                # #                 angle_car.publish(-60)
                # #     else:
                # #         print("DI THANG")
                # #         time_nl_t=time.time()
                # #         while time.time()- time_nl_t <= 0.4:
                # #             angle_car.publish(0)
                # #         speed.publish(20)                        
                    
                # #     sign_id = -1
                # ######################################################


                #     # if sign_bool == 5:#no right
                #     #     sign_bool = 1196
                #     #     if speed_up_no(road_on_birdview) > 110:# di thang
                #     #         time_nl_t=time.time()
                #     #         while time.time()- time_nl_t <=0.6:
                #     #             #print(time.time() - time_t)
                #     #             angle_car.publish(0)
                #     #             #print("fuckkkkkkkk")
                #     #             rospy.info("TURN STRAIGHT!!!")
                #     #         #sign_bool= 66
                #     #         steer_bool = -1
                #     #     else:#re phai
                #     #         # print("HERERJEIRJRJKWQNJKASBDJKASDBAHJDHJWAHJD213123213")
                #     #         bamlan = 0
                #     #         print("TURN RIGHT on NORIGHT!!!")
                #     #         time_nl_r=time.time()
                #     #         while time.time() - time_nl_r <=1:
                #     #             #print("TURN RIGHT \t",time.time() - time_nl_r)
                #     #             angle_car.publish(60)
                #     #             #print("fuck")
                #     #         #sign_bool= 66
                #     #         speed.publish(20)
                #     #         steer_bool = -1                       
                #     sign_passaway=0
                
                #print("Mat do bien",sign_density)


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


                # if sign_id == -1: # Khong co bien bao thi di voi toc do 20
                #     speed.publish(20)

                ''' We will to compute angle to steer car in here

                '''
                angle_degree = compute_centroid(road_on_birdview)
                # if (abs(angle_degree) > 18):
                #     flag_angle = 0
                # if (abs(angle_degree) < 18 and first_btn == True):
                #     flag_angle = 1

                
                
                
                # if (flag_angle == 1):
                #     speed.publish(20)
                # elif (flag_angle ==0):
                #     speed.publish(18)            

                #angle_degree = angle_calculator(centroid_x,centroid_y) # Call angle_calculator method in speed_up.py to use numba function

                #rospy.logwarn("Steer Angle:  " + str(angle_degree))
                #lcd.update_message("Angle " + str(round(angle_degree,1)),0,3)

                angle.data = angle_degree #Fuzzy(centroid_x - 64) #angle_degree
                if sign_id == 4:
                    if count_f < 30:
                        angle.data -= 12.5
                    elif count_f > 30:
                        count_f = 0
                        sign_id = -1
                    count_f += 1
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
