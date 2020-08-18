import os
import numpy as np
from std_msgs.msg import Header,String,Float32,Int8,Bool
from sensor_msgs.msg import CompressedImage,Image
import rospy
import cv2

'''
MIT License

Copyright (c) 2019 Dat Vu

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

class btn_status:
    def __init__(self):
        self.bt1_status = rospy.Subscriber("/bt1_status",
                                    Bool,
                                    self.bt1_status_callback,
                                    queue_size=1)
        self.bt2_status = rospy.Subscriber("/bt2_status",
                                    Bool,
                                    self.bt2_status_callback,
                                    queue_size=1)
        self.bt3_status = rospy.Subscriber("/bt3_status",
                                    Bool,
                                    self.bt3_status_callback,
                                    queue_size=1)
        self.bt4_status = rospy.Subscriber("/bt4_status",
                                    Bool,
                                    self.bt4_status_callback,
                                    queue_size=1)          
        self.ss1_status = rospy.Subscriber("/ss1_status",
                                    Bool,
                                    self.ss1_status_callback,
                                    queue_size=1) 

        self.ss2_status = rospy.Subscriber("/ss2_status",
                                    Bool,
                                    self.ss2_status_callback,
                                    queue_size=1) 

        self.led_publish = rospy.Publisher("/led_status",Bool,queue_size = 1)

        self.bt1_bool = False
        self.bt2_bool = False
        self.bt3_bool = False
        self.bt4_bool = False
        self.ss1_status = False
        self.ss2_status = False
        self.mess = Bool()
                                                             
    def bt1_status_callback(self,ros_data):
        self.bt1_bool = ros_data.data
        
    def bt2_status_callback(self,ros_data):
        # print("Buton 2: ",ros_data.data)
        self.bt2_bool = ros_data.data

    def bt3_status_callback(self,ros_data):
        # print("Buton 3: ",ros_data.data)
        self.bt3_bool = ros_data.data

    def bt4_status_callback(self,ros_data):
        # print("Buton 4: ",ros_data.data)
        self.bt4_bool = ros_data.data
    def ss1_status_callback(self,ros_data):
        # print("Buton 4: ",ros_data.data)
        self.ss1_status = ros_data.data
    def ss2_status_callback(self,ros_data):
        # print("Buton 4: ",ros_data.data)
        self.ss2_status = ros_data.data

    def led_send_message(self,message):

        self.mess.data = message
        self.led_publish.publish(message)


