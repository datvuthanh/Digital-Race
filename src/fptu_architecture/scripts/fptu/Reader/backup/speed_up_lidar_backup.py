#!/usr/bin/env python3
from numba import jit,cuda,prange
import numpy as np
import time
import math 

@jit(nopython=True,fastmath=True)
def lidar_speed(count,ranges,distance_x_min,distance_y_min,distance_y_min_raw, distance_x_min_raw):

    mat_do_diem = 0
    mat_do_diem_goc_rong = 0

    matran_X=[]
    matran_Y=[]

    for i in range(0,count):

        distance_x = ranges[i]*math.sin(degtorad(i))
        distance_y = ranges[i]*math.cos(degtorad(i))
        matran_X.append(int(32*(distance_x+12)))
        matran_Y.append(int(32*(distance_y+12)))
        
        '''
        goc quet cua lidar

                               y
                               
                               ^  
                               -
        0.4                    -                      -0.4
        ---------------------------------------------------
        -                      -                      -
         -                     -                     -
          -                    -                    -
           -                   -                   -
            -                  -                  -
             -                 -   2m            -
              -                -                -
               -               -               -
                -              -              -
                 -             -             -
                  -            -            -
                   -           -           -
        x<-------------------------------------------------
                0.2                      -0.2


        '''  
        if 0.5 < distance_y < 2:
            if abs(distance_x) < (distance_y*1/10+0.2) :
                mat_do_diem_goc_rong = mat_do_diem_goc_rong + 1

        if ranges[i] < 3:
            #print("hello")
            if 0.5 < distance_y < 2 and abs(distance_x) < 0.20 :
                #print("fuck")
                mat_do_diem = mat_do_diem + 1
                if abs(distance_y_min) > abs(float(distance_y)) and abs(distance_x_min) > abs(float(distance_x)):
                    #rospy.logerr("MINIMUM: \t" + str(distance_x) + "\t" + str(distance_y))
                    distance_x_min = distance_x
                    distance_y_min = distance_y
        
        # if ranges[i] < 0.5:
        #     if abs(distance_y_min_raw) > abs(float(distance_y)) and abs(distance_y) >0.2:
        #         distance_y_min_raw = distance_y  


        # if  distance_x > -0.7 and distance_x < 0 and   distance_y>-0.5 and distance_y < 1.5:
        #     distance_y_min_raw = distance_y_min
        #print(distance_x_min,distance_y_min)
        if -0.7 < distance_x < 0:
            #print("fuck")
            if -0.6 < distance_y < 2.5:
                #print("hello")
                #print("hello", distance_x_min_raw, distance_x)
                if abs(distance_x_min_raw) > abs(distance_x):
                    distance_x_min_raw = distance_x
                    distance_y_min_raw = distance_y
                    
    #print("DISTANCE Y RAW",distance_y_min_raw)
    # print("DEMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM", mat_do_diem )
    # print("fuckyouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu", mat_do_diem_goc_rong)
    # print("chieu oxxxxxxxxxxxxxxxxxxxxxxxx",matran_X)
    # print("chieu oyyyyyyyyyyyyyyyyyyyyyyyy",matran_Y)
    # matran[22][22]=1
    # for i in range (0,len(matran_X)):
    #     if 0<matran_X[i]<384 and 0< matran_Y[i]<384:
    #         matran[matran_X[i]][384-matran_Y[i]]=255
    
    # print("matran diem la:", matran[22][22], matran)
    print(mat_do_diem,mat_do_diem_goc_rong)
    if  mat_do_diem > 7 and mat_do_diem_goc_rong < 12 :
        print ("111111111111111111111111111111")
        return -1,matran_X,matran_Y #co vat can (th ne xe tren duong thang bth )
    elif 0 < mat_do_diem and 4 < mat_do_diem_goc_rong < 9:
        return -1,matran_X, matran_Y  # co vat can ( th ne xe o goc cua )
        # th gap bien bao (mat_do_diem <4 va mat_do_diem_goc_rong < 4)
        # th gap rao chan (mat_do_diem_goc_rong > 12)
    elif 0 < distance_y_min_raw < 0.7:
        #print("hellloooooooooooooooooooooo",distance_y_min_raw)
        return 1,matran_X, matran_Y # bien bao  
    
    
    return 0,matran_X, matran_Y #khong co vat can , bien bao 

# @jit(nopython=True,parallel=True)
# def lidar_speed_raw(count,ranges,distance_x_min,distance_y_min):
#     for i in range(0,count):
#         distance_x = ranges[i]*math.sin(degtorad(i))
#         distance_y = ranges[i]*math.cos(degtorad(i))

#         if ros_data.ranges[i] < 3:
#             if abs(self.distance_y_min) > abs(float(distance_y)) and abs(self.distance_x_min) > abs(float(distance_x)) and abs(distance_x) < 0.25:
#                 if abs(distance_y) > 0.5:
#                     #rospy.logerr("MINIMUM: \t" + str(distance_x) + "\t" + str(distance_y))
#                     self.distance_x_min = distance_x
#                     self.distance_y_min = distance_y

#         if ros_data.ranges[i] < 0.5:
#             if abs(self.distance_y_min_raw) > abs(float(distance_y)) and abs(distance_y) >0.2:
#                 self.distance_y_min_raw = distance_y

#         #rospy.logerr("toa_do  " + str(self.distance_x_min) +"-------" +str(self.distance_y_min))
#         if  0.3 < self.distance_y_min < 2 :
#             boolean = Bool()
#             boolean.data = True
#             self.detection.publish(boolean)
#             rospy.logerr("OBSTACLE DETECTION     " + str(self.distance_y_min)


@jit(nopython=True,fastmath=True)
def lidar_speed_backup(count,ranges,distance_x_min,distance_y_min,distance_y_min_raw, distance_x_min_raw):
    bao_hieu_1=0
    bao_hieu_2=0
    bao_hieu_3=0
    bao_hieu_4=0
    mat_do_diem = 0
    mat_do_diem_goc_rong = 0

    matran_X=[]
    matran_Y=[]

    for i in range(0,count):
        distance_x = ranges[i]*math.sin(degtorad(i))
        distance_y = ranges[i]*math.cos(degtorad(i))
        matran_X.append(int(32*(distance_x+12)))
        matran_Y.append(int(32*(distance_y+12)))
        
        '''
        goc quet cua lidar

                               y
                               
                               ^  
                               -
        0.4                    -                      -0.4
        ---------------------------------------------------
        -                      -                      -
         -                     -                     -
          -                    -                    -
           -                   -                   -
            -                  -                  -
             -                 -   2m            -
              -                -                -
               -               -               -
                -              -              -
                 -             -             -
                  -            -            -
                   -           -           -
        x<-------------------------------------------------
                0.2                      -0.2


        '''
        '''  
        return lai nhieu gia tri khac nhau
        bao hieu 1, bao hieu 2 , bao hieu 3, bao hieu 4, matranX, matranY
        bao hieu 1: vat can 
        bao hieu 2: bien bao stop
        bao hieu 3: bien bao re trai+ re phai
        bao hieu 4: bien bao cam re trai+ cam re phai
        matranX + matranY: mapping
        '''
        if 0.5 < distance_y < 2:
            if abs(distance_x) < (distance_y*1/10+0.2) :
                mat_do_diem_goc_rong = mat_do_diem_goc_rong + 1
        if ranges[i] < 3:
            #print("hello")
            if 0.5 < distance_y < 2 and abs(distance_x) < 0.20:
                #print("fuck")
                mat_do_diem = mat_do_diem + 1
                if abs(distance_y_min) > abs(float(distance_y)) and abs(distance_x_min) > abs(float(distance_x)):
                    #rospy.logerr("MINIMUM: \t" + str(distance_x) + "\t" + str(distance_y))
                    distance_x_min = distance_x
                    distance_y_min = distance_y
        
        # if ranges[i] < 0.5:
        #     if abs(distance_y_min_raw) > abs(float(distance_y)) and abs(distance_y) >0.2:
        #         distance_y_min_raw = distance_y  


        # if  distance_x > -0.7 and distance_x < 0 and   distance_y>-0.5 and distance_y < 1.5:
        #     distance_y_min_raw = distance_y_min
        #print(distance_x_min,distance_y_min)
        if -0.7 < distance_x < 0:
            #print("fuck")
            if -0.6 < distance_y < 2.5:
                #print("hello")
                #print("hello", distance_x_min_raw, distance_x)
                if abs(distance_x_min_raw) > abs(distance_x):
                    distance_x_min_raw = distance_x
                    distance_y_min_raw = distance_y
                    
    #print("DISTANCE Y RAW",distance_y_min_raw)
    # print("DEMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM", mat_do_diem )
    # print("fuckyouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu", mat_do_diem_goc_rong)
    # print("chieu oxxxxxxxxxxxxxxxxxxxxxxxx",matran_X)
    # print("chieu oyyyyyyyyyyyyyyyyyyyyyyyy",matran_Y)
    # matran[22][22]=1
    # for i in range (0,len(matran_X)):
    #     if 0<matran_X[i]<384 and 0< matran_Y[i]<384:
    #         matran[matran_X[i]][384-matran_Y[i]]=255
    
    # print("matran diem la:", matran[22][22], matran)

    if  mat_do_diem > 7 and mat_do_diem_goc_rong < 12 :
        bao_hieu_1 = 1 #co vat can (th ne xe tren duong thang bth )
    elif 0 < mat_do_diem and 4 < mat_do_diem_goc_rong < 9:
        bao_hieu_1 = 1  # co vat can ( th ne xe o goc cua )
        # th gap bien bao (mat_do_diem <4 va mat_do_diem_goc_rong < 4)
        # th gap rao chan (mat_do_diem_goc_rong > 12)
    '''bien bao stop'''
    if 0 < distance_y_min_raw < 1.3:
        #print("hellloooooooooooooooooooooo",distance_y_min_raw)
        # bien bao  
        bao_hieu_2 = 1
    '''bien bao re trai+ re phai'''
    if 0 < distance_y_min_raw < 0.7:
        bao_hieu_3 = 1
    '''bien cam re'''
    if 0 < distance_y_min_raw < 1.3:
        bao_hieu_4 = 1
    return bao_hieu_1,bao_hieu_2,bao_hieu_3,bao_hieu_4,matran_X,matran_Y
    #khong co vat can , bien bao


@jit(nopython=True,fasthmath=True)
def radtodeg(x):
    return x*180./math.pi

@jit(nopython=True,fastmath=True)
def degtorad(x):
    return float(x*math.pi/180)
