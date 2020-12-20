
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
from datetime import datetime
import threading


def stop:
    time_stop = time.time()
    while time.time()-time_stop < 3:
        speed.publish(0)
    speed.publish(20)
def turn_right:
    time_turn_right = time.time()
    while time.time() - time_turn_right <= 0.8 :
        angle_car.publish(-60)
def turn_left:
    time_turn_left = time.time()
    while time.time() - time_turn_left <= 0.8 :
        angle_car.publish(60)
def straight:
    time_run_away= time.time()
    while time.time() - time_run_away <= 0.8 :
        angle_car.publish(0)