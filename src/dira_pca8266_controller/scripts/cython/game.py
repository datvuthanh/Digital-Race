#!/usr/bin/python

import pygame
import time
import rospy
from std_msgs.msg import Float32
#import sys
import xbox360_controller
import game
rospy.init_node('goodgame_never_die', anonymous=True)
speed_pub = rospy.Publisher("/set_speed", Float32, queue_size=1)
steerAngle_pub = rospy.Publisher("/set_angle", Float32, queue_size=1)

rate = rospy.Rate(10)

pygame.init()

# # define some colors
# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# RED = (255, 0, 0)

# # window settings
# size = [600, 600]
# screen = pygame.display.set_mode(size)
# pygame.display.set_caption("Simple Game")
FPS = 60
clock = pygame.time.Clock()

# # make a controller
controller = xbox360_controller.Controller()

# # make a ball
# ball_pos = [290, 290]
# ball_radius = 10
# ball_color = WHITE

# game loop
playing = False
done = False

#velocity = 0
#steering = 0
#left_x = 0
#right_y = 0

while not rospy.is_shutdown():
    while not done:
        # event handling
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                done=True
    #           sys.exit(0)

            if event.type == pygame.JOYBUTTONDOWN:
                # handle events for all controllers
                if not playing:
                    if event.button == xbox360_controller.START:
                        playing = True
                else:
                    if event.button == xbox360_controller.BACK:
                        playing = False

                # handle events for specific controllers
                if event.joy == controller.get_id():
                    if event.button == xbox360_controller.A:
                        test = Float32()
                        test.data = 0.0
                        speed_pub.publish(test)
                    if event.button == xbox360_controller.B:
                        test = Float32()
                        test.data = 60.0
                        steerAngle_pub.publish(test)                        
                    if event.button == xbox360_controller.X:
                        test = Float32()
                        test.data = -60.0
                        steerAngle_pub.publish(test)        
                    if event.button == xbox360_controller.Y:
                        test = Float32()
                        test.data = 15.0
                        speed_pub.publish(test)        

        # handle joysticks

        left_x, left_y = controller.get_left_stick()
        right_x, right_y  = controller.get_right_stick()
        # print("HERE: ",right_x,right_y)
        velocity = game.velocity(right_y)
        steering = game.steering(left_x)

        #print("SIGN: ",right_y,left_x)

        #print("VAN TOC VA GOC: ",velocity,steering)

        if -15 < velocity < 15:
            axis_h = velocity
        elif velocity > 15:
            axis_h = 15
        elif velocity < -15:
            axis_h = -15
        else:
            axis_h = 0

        if -60 < steering < 60:
            axis_v = steering
        elif steering > 60: 
            axis_v = 60
        elif steering < -60:
            axis_v = -60
        else:
            axis_v = 0

        #axis_h = max(-15, right_y * (-15))
        # print("GIC TRI TOC DO: ",axis_h)
        #axis_v = left_x * -90
        # print("GIA TRI GOC:", axis_v)
        # game logic
        if playing:
            print("GIA TRI VAN TOC VA GOC LAI: ",axis_h,axis_v)
            speed_data = Float32()
            speed_data.data = axis_h
            speed_pub.publish(speed_data)
            angle_data = Float32()
            angle_data.data = axis_v
            steerAngle_pub.publish(angle_data)

        # drawing
        # screen.fill(BLACK)
        # pygame.draw.circle(screen, ball_color, ball_pos, ball_radius)

        # # update screen
        # pygame.display.flip()
        clock.tick(FPS)
        #rate.sleep()
    # close window on quit
    #rate.sleep()

    pygame.quit ()

