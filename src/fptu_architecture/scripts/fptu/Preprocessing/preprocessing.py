#!/usr/bin/env python3

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

# from fptu.Reader.get_frames import read_input
import rospy
import cv2
import numpy as np 

class pre_processing():

    def __init__(self,frame):
        self.rgb_frame = frame
        # self.src_pts = np.float32([[0,85],[320,85],[320,240],[0,240]])
        # self.dst_pts = np.float32([[0,0],[320,0],[200,240],[120,240]])
        self.src_pts = np.float32([[35,0],[93,0],[128,64],[0,64]])
        self.dst_pts = np.float32([[30,0],[98,0],[98,128],[30,128]])
        # self.depth_frame = read_input.frame_depth
        # rospy.loginfo("IMAGE DATA")
        # print(self.rgb_frame)
        # cv2.imshow("RGB",self.rgb_frame)
        # cv2.waitKey(1)

    def gray_frame(self):
        self.gray_frame = cv2.cvtColor(self.rgb_frame,cv2.COLOR_BGR2GRAY)
        return self.gray_frame

    def processImage_houghLine(self):

        self.gray_image = cv2.cvtColor(self.rgb_frame,cv2.COLOR_BGR2GRAY)
        self.kernel_size = 5
        self.blur_gray = cv2.GaussianBlur(self.gray_image,(self.kernel_size,self.kernel_size),0)
        self.low_threshold = 50
        self.high_threshold = 150
        self.edges = cv2.Canny(self.blur_gray,self.low_threshold,self.high_threshold)
        #cv2.imshow("Edges",self.edges)
        self.mask = np.zeros_like(self.edges)
        self.ignore_mask_color = 255

        self.imshape = self.rgb_frame.shape
        #vertices = np.array([[(0,imshape[0] *1 / 2),(imshape[1], imshape[0] * 1 / 2), (imshape[1], imshape[0]), (0,imshape[0])]], dtype=np.int32)
        self.vertices = np.array([[(0,self.imshape[0] * 1 / 2),(self.imshape[1], self.imshape[0] * 1 / 2), (self.imshape[1], self.imshape[0]), (0,self.imshape[0])]], dtype=np.int32)
        cv2.fillPoly(self.mask, self.vertices, self.ignore_mask_color)
        self.masked_edges = cv2.bitwise_and(self.edges, self.mask)
        #cv2.imshow("mask",self.masked_edges)

        #find all your connected components (white blobs in your image)
        self.nb_components, self.output, self.stats, self.centroids = cv2.connectedComponentsWithStats(self.masked_edges, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        self.sizes = self.stats[1:, -1]; self.nb_components = self.nb_components - 1
        
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 80    # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 60 #minimum number of pixels making up a line
        max_line_gap = 30    # maximum gap in pixels between connectable line segments
        self.line_image = np.copy(self.rgb_frame)*0 # creating a blank to draw lines on
        
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        self.lines = cv2.HoughLinesP(self.masked_edges, rho, theta, threshold, np.array([]), #modify masked_edges = img2.astype(np.uint8)
                                    min_line_length, max_line_gap)
        
        # Iterate over the output "lines" and draw lines on a blank image

        self.line = None
        try:
            for self.line in self.lines:
                for x1,y1,x2,y2 in self.line:
                    cv2.line(self.line_image,(x1,y1),(x2,y2),(255,0,0),2)
        except:
            pass
            #print("No line to draw")
        # Create a "color" binary image to combine with line image
        #color_edges = np.dstack((edges, edges, edges))wdlog_lineRight
        
        # Draw the lines on the original image
        self.lines_edges = cv2.addWeighted(self.rgb_frame, 0.8, self.line_image, 1, 0)
        
        return self.lines_edges,self.lines,self.nb_components

    def birdView(self,img,M):
        '''
        Transform image to birdeye view
        img:binary image
        M:transformation matrix
        return a wraped image
        '''
        #print("wewewewewewewwewewew",img.shape)
        img_sz = (img.shape[1],img.shape[0])
        img_warped = cv2.warpPerspective(img,M,img_sz,flags = cv2.INTER_LINEAR)
        return img_warped

    def perspective_transform(self):
        '''
        perspective transform
        args:source and destiantion points
        return M and Minv
        '''

        M = cv2.getPerspectiveTransform(self.src_pts,self.dst_pts)
        Minv = cv2.getPerspectiveTransform(self.dst_pts,self.src_pts)

        return {'M':M,'Minv':Minv}   
         
# if __name__ == '__main__':

#     rospy.init_node('preprocessing', anonymous=True)

#     #rate = rospy.Rate(30) 
#     #rgb_frame = read_input.callback_rgb()

#     #cv2.imshow("rgb",rgb_frame)
#     #cv2.waitKey(1)

#     #while not rospy.is_shutdown():
#         # define frame object and detect
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down ROS Image feature detector module")

#     #cv2.destroyAllWindows()