import os
import numpy as np
from std_msgs.msg import Header,String,Float32,Int8
from sensor_msgs.msg import CompressedImage,Image
import rospy
import cv2

class lcd_print:

    def __init__(self,mess,row,column):
        
        self.lcd_publish = rospy.Publisher("/lcd_print",String,queue_size = 1)

        self.row = str(row)

        self.column = str(column)

        self.mess = mess

        self.message = (self.row) + ':' + (self.column) + ':' + self.mess

        self.message_lcd = String()

        self.send_message()

    def send_message(self):

        if 0 <= int(self.row) <= 3 and 0 <= int(self.column) <= 19:
            
            self.message_lcd.data = self.message

            self.lcd_publish.publish(self.message_lcd)
    
    def update_message(self,mess,row,column):

        self.row = str(row)

        self.column = str(column)

        self.mess = mess
        
        self.message = (self.row) + ':' + (self.column) + ':' + self.mess

        self.send_message()

    def clear(self):
        
        self.message = "0" + ":" + "0" + ":" + "                    "

        self.send_message()

        self.message = "0" + ":" + "1" + ":" + "                    "

        self.send_message()

        self.message = "0" + ":" + "2" + ":" + "                    "

        self.send_message()

        self.message = "0" + ":" + "3" + ":" + "                    "

        self.send_message()




