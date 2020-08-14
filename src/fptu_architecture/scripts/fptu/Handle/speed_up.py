#!/usr/bin/env python3
from numba import njit,jit,cuda,prange
import numpy as np
import time
import math 
import cv2

@njit(fastmath=True)
def angle_calculator(x,y):
    
    slope = (x - 50) / float(y - 128) # (50,128) is centroid of image size 100x128

    angle_radian= float(math.atan(slope))

    angle_degree= float(angle_radian * 180.0 / math.pi)

    return angle_degree

@njit(fastmath=True)
def sign_detection(straight,left,right,stop,noleft,noright):
    if straight > 100:
        return 0
    elif left > 100:
        return -1
    elif right > 100:
        return 1
        
    elif stop  > 100:
        return 2
    elif noleft >100:
        return -3
    elif noright > 100:
        return 3
######################################################
# Dat Vu add new method on 28/04 
@njit(fastmath=True)
def remove_noise_matrix(line,bamlan,road):
  min_x = 255
  max_x = -1
  count_noise_lan_trai = 0
  count_noise_lan_phai = 0
  
  total_right_line = 0
  total_left_line = 0
  for i in range(0,64):
    for j in range(0,128):
      #print(i,j)
      if line[i][j] == 255:
        if j > max_x:
          max_x = j
        if j < min_x:
          min_x = j
        if j > 64:
          total_right_line += 1
        if j < 64: 
          total_left_line += 1
    if min_x != 255 or max_x != -1:
      if bamlan == -1 and min_x < 64:
        for k in range(0,min_x):
          if road[i][k] == 255:
            #count_noise_lan_trai += 1
            road[i][k] = 0
      if bamlan == 1 and max_x > 64:
        #print("GIA TRI MAX (X,Y): ",max_x,j,i)
        for k in range(max_x,128):
          if road[i][k] == 255:
            #count_noise_lan_phai += 1
            road[i][k] = 0
      # if bamlan == 0 and min_x < 64 and max_x > 64:
      #     for k in range(0,min_x):
      #         if road[i][k] == 255:
      #             road[i][k] = 0
      #     for k in range(max_x,128):
      #         if road[i][k] == 255:
      #             road[i][k] = 0
    # In the case, we can't find line of road but model still detect road
    # We need remove road in row of line
    #elif min_x == 255 or max_x != -1:
    #  if bamlan == -1 or bamlan == 1:
    #    for k in prange(0,128):
    #      road[i][k] = 0

    max_x = -1
    min_x = 255
  # for i in range(0,64):
  #   for j in range(0,128):
  #     if line[i][j]==255:
  return road,total_left_line,total_right_line

@njit(fastmath=True)
def remove_lane_lines(road,bamlan,sign_id):
  min_x = 255
  max_x = -1
  for i in range(0,64):
    for j in range(0,128): 
      if road[i][j] == 255:
        if j < min_x:
          min_x = j 
        if j > max_x:
          max_x = j
    #print("Max_x",max_x,min_x)
    centroid_x_new = math.ceil((max_x + min_x ) / 2)
    if bamlan == 1:
      for k in range(0,centroid_x_new):  
        if road[i][k] == 255:
          road[i][k] = 0
    if bamlan == -1:
      for k in range(centroid_x_new,128):  
        if road[i][k] == 255:
          road[i][k] = 0
      
    #Reset max_x min_x
    min_x = 255
    max_x = -1 
  
  return road

# Compute centroid of road 
@njit(fastmath=True)
def compute_centroid(road):
  count = 0
  centroid_x = 0
  centroid_y = 0 

  ####
  count_2 = 0
  centroid_x_2 = 0 
  centroid_y_2 = 0
  for i in range(0,128): # We will compute based on matrix 100x128 (rows * columns) # Default 0,128
    for j in range(0,100):
      if road[i][j] == 255:
        count += 1
        centroid_y += i
        centroid_x += j
      if road[i][j] == 255 and i > 85:
        count_2 += 1
        centroid_x_2 += j
        centroid_y_2 += i

  if centroid_x != 0 or centroid_y != 0 or count != 0:
    centroid_x = centroid_x / count
    centroid_y = centroid_y / count
    angle_degree = angle_calculator(centroid_x,centroid_y) # Call angle_calculator method in speed_up.py to use numba function


  if centroid_x_2 != 0 or centroid_y_2 != 0 or count_2 != 0:
    centroid_x_2 = centroid_x_2 / count_2
    centroid_y_2 = centroid_y_2 / count_2
    angle_degree_2 = angle_calculator(centroid_x_2,centroid_y_2) # Call angle_calculator method in speed_up.py to use numba function
  
  if abs(angle_degree) < 18:
    return angle_degree
  else:
    return angle_degree

# @njit(fastmath=True)
# def compute_centroid_abc(road):
#   count = 0
#   centroid_x = 0
#   centroid_y = 
#   count_asdfjkjasdk = 0
#   for i in range(0,128): # We will compute based on matrix 100x128 (rows * columns) # Default 0,128
#     for j in range(0,100):
#       if road[i][j] == 255:
#         count += 1
#         centroid_y += i
#         centroid_x += 
#       if road[i][j] == 255 and i > 85:
        
#   if centroid_x != 0 or centroid_y != 0 or count != 0:
#     centroid_x = centroid_x / count
#     centroid_y = centroid_y / 
  
#   ## Goc lai
#   # if goclai > 18:
#   #   road(100,85)

#   return centroid_x,centroid_y


###########################################
# Dat Vu add on 16/05/2020 

@njit(fastmath=True)
def getclassIndex(prob_array):
  #print(prob_array)
  length_array = len(prob_array)
  max_value = -1
  index = -1
  for i in range(0,length_array):
    if prob_array[i] > max_value:
      max_value = prob_array[i]
      index = i
  return index,max_value

@njit(fastmath=True)
def noright(road_on_birdview):
  max1=0
  max2=0
  for i in range(0,100):
      if road_on_birdview[0][i]!=0:
          max1=i
      if road_on_birdview[127][i]!=0:
          max2=i
  for i in range(0,128):
      for j in range(min(max1,max2),100):
          road_on_birdview[i][j]=0
  return road_on_birdview

@njit(fastmath=True)
def noleft(road_on_birdview):
  min1=0
  min2=0
  for i in range(0,100):
      if road_on_birdview[0][i]!=0:
          min1=i
          break
  for i in range(0,100):
      if road_on_birdview[127][i]!=0:
          min2=i
          break
  for i in range(0,128):
      for j in range(0,max(min1,min2)):
          road_on_birdview[i][j]=0
  return road_on_birdview



### Huy Phan do this function to drive car on no left---no right sign
@njit(fastmath = True)
def notTurnLeft(road,line,sign_bool):
  if sign_bool == left: #we get no left detection
    for i in range(0,64):
      for j in range(60,68):
        #if detect any line in this range we confirm that we in 90 degree
        #any  point is not white so we keep go straight that mean angle_degree = 0

        if line[i][j] != 255:
          angle_degree =0
        else:
          #we got only left plane of road
          road = road[:,64:]
  return road 

@njit(fastmath = True)
def notTurnRight(road,line,sign_bool):
  if sign_bool == right: #we get no left detection
    for i in range(0,64):
      for j in range(60,68):
        #if detect any line in this range we confirm that we in 90 degree
        #any  point is not white so we keep go straight that mean angle_degree = 0
        if line[i][j] != 255:
          angle_degree = 0
        else:
          #we got only right plane of road
          road = road[:,:64]
  return road          

# Dat Vu add on 23/05/2020
# Compute centroid of road 
@njit(fastmath=True)
def compute_sign(sign):
  count = 0
  centroid_x = 0
  centroid_y = 0
  for i in range(0,144): # We will compute based on matrix 100x128 (rows * columns) # Default 0,128
    for j in range(0,144):
      if sign[i][j] == 255:
        count += 1
        centroid_y += i
        centroid_x += j
  if centroid_x != 0 or centroid_y != 0 or count != 0:
    centroid_x = centroid_x / count
    centroid_y = centroid_y / count

  return centroid_x,centroid_y

@jit(nopython=True,fastmath=True)
def find_biggest_components(nb_components,output,stats):
  sizes = stats[:, -1]
  max_label = 1
  max_size = sizes[1]
  for i in range(2, nb_components):
    if sizes[i] > max_size:
      max_label = i
      max_size = sizes[i]
  return max_label


# Dat Vu add for classifiers.py
@jit(nopython=True,fastmath=True)
def sign_density_check(image_localization,sign_density,centroid_x,centroid_y):
  check = False
  if 4000>sign_density>2000:
    if(centroid_x >110):
      #print("Case: 2000 < sign_density < 4000")
      image_localization = image_localization[centroid_y - 30 : centroid_y + 30, centroid_x - 30 : centroid_x + (144-centroid_x)]
      check = True
    else:
      image_localization = image_localization[centroid_y - 30 : centroid_y + 30, centroid_x - 30 : centroid_x + 30]
      check = True
  if 2000>=sign_density>1000:
    #print("Case: 1000 < sign_density < 2000")
    if(centroid_x > 115):
      image_localization = image_localization[centroid_y - 25 : centroid_y + 30, centroid_x - 25 : centroid_x + (144-centroid_x)]
      check = True
    else:
      image_localization = image_localization[centroid_y - 25 : centroid_y + 30, centroid_x - 25 : centroid_x + 25]
      check = True
    
  if 1000>=sign_density>500:
    #print("Case: 500 < sign_density < 1000")
    if(centroid_x>123):
      image_localization = image_localization[centroid_y - 20 : centroid_y + 20, centroid_x - 20 : centroid_x + (144-centroid_x)]
      check = True
    else:
      image_localization = image_localization[centroid_y - 20 : centroid_y + 20, centroid_x - 20 : centroid_x + 20]
      check = True
    
  if 500>=sign_density>250:
    #print("Case: 250 < sign_density < 500")
    if(centroid_x>128):
      image_localization = image_localization[centroid_y - 15 : centroid_y + 15, centroid_x - 15 : centroid_x + (144-centroid_x)]
      check = True
    else:
      image_localization = image_localization[centroid_y - 15 : centroid_y + 15, centroid_x - 15 : centroid_x + 15]
      check = True
    
  if 250>=sign_density>0:
    #print("Case: 0 < sign_density < 250")
    if(centroid_x>131):
      image_localization = image_localization[centroid_y - 12 : centroid_y + 12, centroid_x - 12 : centroid_x + (144-centroid_x)]
    else:
      image_localization = image_localization[centroid_y - 12 : centroid_y + 12, centroid_x - 12 : centroid_x + 12]
    check = True
    
  #print('Centroid_y: ',centroid_y, 'Centroid x: ', centroid_x, 'sign density: ', sign_density)
  return check, image_localization
 
 
@jit(nopython=True,fastmath=True)
def getClassName(classNo):
    if classNo == 0: return "Stop"
    if classNo == 1: return "Right"
    if classNo == 2: return "Left"
    if classNo == 3: return "Straight"
    if classNo == 4: return "No Left"
    if classNo == 5: return "No Right"


@jit(nopython=True,fastmath=True)
def speed_up_no(road_on_birdview):
  #road_on_birdview_T=road_on_birdview.T
  sum_road = 0
  for i in range(0,128):
    if road_on_birdview[i][50] == 255:
      sum_road += 1
  ####
  return sum_road