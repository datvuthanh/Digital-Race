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
last_axis_h = 0
last_axis_v = 0
def callback(data):
    global velocity
    global steering
    global right_y
    global left_x
    global axis_h
    global axis_v
    global last_axis_h
    global last_axis_v

    #print(data)
    #twist = Twist()
    #twist.linear.x = 4*data.axes[1]
    #twist.angular.z = 4*data.axes[0]
    #pub.publish(twist)
    
    #right_y_before = right_y
    #left_x_before = left_x

    left = data.buttons[4]
    right = data.buttons[5]
    down = data.buttons[0]
    up = data.buttons[3]

    up_runner = data.buttons[1]
    up_runner2 = data.buttons[2]

    last_axis_h = axis_h
    last_axis_v = axis_v
    #right_y  = data.axes[4]
    
    if down == 1:
        velocity = 0
    if up == 1:
        velocity = 15
    if left == 1:
        steering += left * 10
    if right == 1:
        steering += right * -10
    if up_runner == 1:
        velocity = 25
    if up_runner2 == 1:
        velocity = 20
    #velocity += right_y * 1
    #steering += left_x * 10

    #print("VAN TOC: ",velocity)
    
    #print("GOC LAI: ",steering)

    #if right_y_before * right_y <= 0:
    #    velocity = 0
    #if left_x_before * left_x <= 0:
    #    steering = 0  
    if 0 >= velocity:
        axis_h = 0
        velocity = 0
    if 0 < velocity < 25:
        axis_h = velocity
    elif velocity >= 25:
        axis_h = 25
        velocity = 25
    elif velocity <= -25:
        axis_h = -25
        velocity = -25
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

    #if data.buttons[0] == 1:
    #    axis_h = 0
    #if data.buttons[1] == 1:
    #    axis_v = 60
    #if data.buttons[2] == 1:
    #    axis_v = -60
    #if data.buttons[3] == 1:
    #    axis_h = 15
    if last_axis_h != axis_h or last_axis_v != axis_v:
        speed_data = Float32()
        speed_data.data = axis_h
        speed_pub.publish(speed_data)
        angle_data = Float32()
        angle_data.data = axis_v
        steerAngle_pub.publish(angle_data)
        print("FINAL: ",axis_h,axis_v)
    #global rate
    #rate.sleep()
# Intializes everything
def start():
    # publishing to "turtle1/cmd_vel" to control turtle1
    global speed_pub
    global steerAngle_pub
    global rate
    #pub = rospy.Publisher('turtle1/cmd_vel', Twist)
    rospy.init_node('goodgame_never_die', anonymous=True)
    speed_pub = rospy.Publisher("/set_speed", Float32, queue_size=1)
    steerAngle_pub = rospy.Publisher("/set_angle", Float32, queue_size=1)
    # subscribed to joystick inputs on topic "joy"
    rospy.Subscriber("joy", Joy, callback)
    #rate =rospy.Rate(10)
    # starts the node
    #rospy.init_node('Joy2Turtle')
    rospy.spin()

if __name__ == '__main__':
    start()
