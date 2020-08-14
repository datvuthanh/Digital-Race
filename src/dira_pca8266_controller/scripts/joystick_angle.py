#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32
# Author: Andrew Dai
# This ROS Node converts Joystick inputs from the joy node
# into commands for turtlesim

# Receives joystick messages (subscribed to Joy topic)
# then converts the joysick inputs into Twist commands
# axis 1 aka left stick vertical controls linear speed
# axis 0 aka left stick horizonal controls angular speed
velocity = 0
steering = 0
right_y = 0
left_x = 0

axis_h = 0
axis_v = 0
def callback(data):
    global velocity
    global steering
    global right_y
    global left_x
    global axis_h
    global axis_v 
   #print(data)
    #twist = Twist()
    #twist.linear.x = 4*data.axes[1]
    #twist.angular.z = 4*data.axes[0]
    #pub.publish(twist)
    right_y_before = right_y
    left_x_before = left_x

    left_x = data.axes[0]
    right_y  = data.axes[4]
    velocity += right_y * 1
    steering += left_x * 10

    print("VAN TOC: ",velocity)
    
    print("GOC LAI: ",steering)

    #if right_y_before * right_y <= 0:
    #    velocity = 0
    #if left_x_before * left_x <= 0:
    #    steering = 0  
    
    if 0 > velocity:
        velocity = 0
        axis_h = 0
    if 0 < velocity < 15:
        axis_h = velocity
    elif velocity >= 15:
        axis_h = 15
        velocity = 15
    elif velocity <= -15:
        axis_h = -15
        velocity = -15
    else:
        axis_h = 0

    if -60 < steering < 60:
        axis_v = steering
    elif steering >= 60: 
        steering = 60
        axis_v = 60
    elif steering <= -60:
        steering = -60
        axis_v = -60
    else:
        axis_v = 0
    #print("FINAL: ",axis_h,axis_v)
    try:
        if data.buttons[10] == 1: #Dung lai ve 0 
            velocity = 0
            axis_h = 0
        elif data.buttons[9] == 1:
            steering = 0
            axis_v = 0
    except:
        pass

    speed_data = Float32()
    speed_data.data = axis_h
    #speed_pub.publish(speed_data)
    angle_data = Float32()
    angle_data.data = axis_v
    steerAngle_pub.publish(angle_data)
    print("FINAL: ",axis_h,axis_v)
# Intializes everything
def start():
    # publishing to "turtle1/cmd_vel" to control turtle1
    global speed_pub
    global steerAngle_pub 
    #pub = rospy.Publisher('turtle1/cmd_vel', Twist)
    rospy.init_node('gg_never_die', anonymous=True)
    speed_pub = rospy.Publisher("/set_speed", Float32, queue_size=1)
    steerAngle_pub = rospy.Publisher("/set_angle", Float32, queue_size=1)
    # subscribed to joystick inputs on topic "joy"
    rospy.Subscriber("joy", Joy, callback)
    # starts the node
    #rospy.init_node('Joy2Turtle')
    rospy.spin()

if __name__ == '__main__':
    start()
