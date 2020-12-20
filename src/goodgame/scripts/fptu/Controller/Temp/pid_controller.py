#!/usr/bin/python

import PID
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline #  Switched to BSpline
import os
import numpy as np
from std_msgs.msg import Header,String,Float32,Int8
from sensor_msgs.msg import CompressedImage,Image,LaserScan
import rospy
import cv2
import time
import math

class pid_controller:
    def __init__(self):
        self.message = rospy.Subscriber("/final_compute",
                                    Float32,
                                    self.pid_callback,
                                    queue_size=1)

        self.speed_publish = rospy.Publisher("/set_speed",Float32,queue_size = 1)
        
        self.angle_publish = rospy.Publisher("/set_angle",Float32,queue_size = 1)

    def pid_callback(self):
        # Receive message speed and angle.
        self.test_pid()
        
        # Publish speed and angle after compute PID

        # self.speed_publish.publish(message)
        # self.angle_publish.publish(message)

    def test_pid(P = 0.2,  I = 0.0, D= 0.0, L=100):

        """Self-test PID class

        .. note::
            ...
            for i in range(1, END):
                pid.update(feedback)
                output = pid.output
                if pid.SetPoint > 0:
                    feedback += (output - (1/i))
                if i>9:
                    pid.SetPoint = 1
                time.sleep(0.02)
            ---
        """

        pid = PID.PID(P, I, D)

        pid.SetPoint=0.0
        pid.setSampleTime(0.01)

        END = L
        feedback = 0

        feedback_list = []
        time_list = []
        setpoint_list = []

        for i in range(1, END):
            pid.update(feedback)
            output = pid.output
            if pid.SetPoint > 0:
                feedback += (output - (1/i))
            if i>9:
                pid.SetPoint = 1
            time.sleep(0.02)

            feedback_list.append(feedback)
            setpoint_list.append(pid.SetPoint)
            time_list.append(i)

        time_sm = np.array(time_list)
        time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)

        # feedback_smooth = spline(time_list, feedback_list, time_smooth)
        # Using make_interp_spline to create BSpline
        helper_x3 = make_interp_spline(time_list, feedback_list)
        feedback_smooth = helper_x3(time_smooth)

        plt.plot(time_smooth, feedback_smooth)
        plt.plot(time_list, setpoint_list)
        plt.xlim((0, L))
        plt.ylim((min(feedback_list)-0.5, max(feedback_list)+0.5))
        plt.xlabel('time (s)')
        plt.ylabel('PID (PV)')
        plt.title('TEST PID')

        plt.ylim((1-0.5, 1+0.5))

        plt.grid(True)
        plt.show()

if __name__ == "__main__":

    rospy.init_node('pid_controller', anonymous=True)
    
    #test_pid(1.2, 1, 0.001, L=50)

    #rate = rospy.Rate(30) 

    pid = pid_controller()
    
    #while not rospy.is_shutdown():
        # define frame object and detect
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    
