import os
import numpy as np
from std_msgs.msg import Header,String,Float32,Int8
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




