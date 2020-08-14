#!/usr/bin/python
import cython

cdef float speed_real = 0
cdef float angle_real = 0
cpdef float velocity(float x):
    global speed_real
    speed_real += x * -1
    return speed_real
cpdef float steering(float x):
    global angle_real
    angle_real += x * -1
    return angle_real
