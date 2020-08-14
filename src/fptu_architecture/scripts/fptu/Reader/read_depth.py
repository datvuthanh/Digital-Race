#!/usr/bin/env python

'''
MIT License

Copyright (c) 2019 Stephen Vu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import numpy as np
from std_msgs.msg import Header,String,Float32,Int8
from sensor_msgs.msg import CompressedImage,Image,Imu
import rospy
import cv2
import time

from fptu.Preprocessing.preprocessing import pre_processing
from fptu.Reader.lcd_publish import lcd_print
from fptu.Reader.btn_status import btn_status
from fptu.Reader.mpu9250 import mpu9250
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError

#from fptu.Handle.deep_handle import segment
import math
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
#from fptu.Handle.tensorrt_inference import segment_rt

class read_depth:

    def __init__(self):


        self.get_depth = rospy.Subscriber("/camera/depth/image_raw",
                                    Image,
                                    self.convert_depth_image,
                                    queue_size=1,buff_size=2**24) 

        self.frame_depth = None
    def callback_depth(self,ros_data):

        self.frame_depth = self.convert_to_np(ros_data)

        #cv2.imshow("Depth",self.frame_depth)
        
        #cv2.waitKey(1)

        return self.frame_depth
    

    def convert_depth_image(self, ros_image):
        bridge = CvBridge()
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
        #Convert the depth image using the default passthrough encoding
            depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
        except CvBridgeError as e:
                pass
                #print(str(e))
        #Convert the depth image to a Numpy array
        self.frame_depth = np.array(depth_image, dtype=np.float32)
        if centroid_x_sign != 0:
            print("Centroids: ",centroid_x_sign,centroid_y_sign)
            print("Gia tri here",self.frame_depth[centroid_y_sign][centroid_x_sign])
        
        return self.frame_depth
        #rospy.loginfo(depth_array)

def count_objects(centroids):
    global centroid_x_before
    global centroid_y_before
    global centroid_x_sign
    global centroid_y_sign
    
    centroid_x_sign = int(centroids.data[0] * 40 / 9)
    centroid_y_sign = int(centroids.data[1] * 10 /3)
    centroid_x_before = int(centroids.data[2] * 40 / 9)
    centroid_y_before = int(centroids.data[3] * 10 /3)


if __name__ == '__main__':

    centroid_x_before = 0
    centroid_y_before = 0
    centroid_x_sign = 0
    centroid_y_sign = 0

    rospy.init_node('read_depth', anonymous=True)
    
    read = read_depth()
    
    rospy.Subscriber("/count", numpy_msg(Floats), count_objects)

    #while not rospy.is_shutdown():
        # define frame object and detect
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    


    #cv2.destroyAllWindows()
