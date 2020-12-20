import os
import numpy as np
from std_msgs.msg import Header,String,Float32,Int8
from sensor_msgs.msg import CompressedImage,Image,Imu
import rospy
import cv2

class mpu9250:
    
    def __init__(self):
        self.mpu_angle = rospy.Subscriber("/imu",
                                    Imu,
                                    self.mpu_angle_callback,
                                    queue_size=1)
                                                             
    def mpu_angle_callback(self,ros_data):
        x = 0
        #print("HEREEEEEEEEEEEEEEEEEEEEEEEEE",ros_data)

