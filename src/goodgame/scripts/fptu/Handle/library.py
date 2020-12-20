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
from fptu.Handle.speed_up import angle_calculator,sign_detection,remove_noise_matrix,remove_lane_lines,compute_centroid,noright,noleft,compute_sign,find_biggest_components,speed_up_no,count_line,cut_road_speedup,cut_road_speedup_for_mid,stop_line
from fptu.Reader.get_frames import read_input
from fptu.Reader.lcd_publish import lcd_print
from fptu.Preprocessing.preprocessing import pre_processing
# from fptu.Controller.error import PID,Fuzzy
from cv_bridge import CvBridge, CvBridgeError
# from fptu.Controller.control_pass_sign import pass_sign
# Add new
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

# Add new 05/06
import roslaunch
from datetime import datetime
import threading