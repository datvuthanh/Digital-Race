#!/usr/bin/env python3

from fptu.Handle.library import *

from fptu.Handle.dl_load import load_model, parse_code, class_return

from fptu.Handle.segment import segmentation

def resetAll():
    global bamlan
    global sign_id
    global first_btn
    global sign_turn_hello
    global object_id
    global avoid_id
    global count_sign
    global check_site 
    global is_avoid_object 
    global straight_time 
    global turnleft_time
    global turnright_time 
    global normal_speed
    global curve_speed 
    global reset_all
    global cuting 
    global cut_road 
    global turn_off
    global car_seg 
    global stop_bool 
    ######################### Initialize Variables #########################
    bamlan = rospy.get_param('~bamlan')  # 0
    sign_id = rospy.get_param('~sign_id')  # -1 # Default -1
    first_btn = rospy.get_param('~first_btn')  # False
    sign_turn_hello = rospy.get_param('~sign_turn_hello')  # 0
    object_id = rospy.get_param('~object_id')  # 0
    avoid_id = rospy.get_param('~avoid_id')  # 0
    count_sign = rospy.get_param('~count_sign')  # 0
    check_site = rospy.get_param('~check_site')  # 0
    is_avoid_object = rospy.get_param('~is_avoid_object')  # False
    straight_time = rospy.get_param('~straight_time')
    turnleft_time = rospy.get_param('~turnleft_time')
    turnright_time = rospy.get_param('~turnright_time')
    normal_speed = rospy.get_param('~normal_speed')
    curve_speed = rospy.get_param('~curve_speed')
    reset_all = rospy.get_param('~reset_all') # False
    cuting = rospy.get_param('~cuting')  # False
    cut_road = rospy.get_param('~cut_road') # False
    turn_off = rospy.get_param('~turn_off') # False
    car_seg = rospy.get_param('~car_seg') # False
    stop_bool = rospy.get_param('~stop_bool') # False
    ##########################################################################
   
def process(single_layer,cut_road=False):
    global bamlan

    global sign_id

    '''
    If we want to visualize all classes on frame
    Uncomment these line
    Of course, when we visualize frame which 'll make our algorithm slower (decrease fps)
    '''"""  """
    ############################
    # output = np.zeros( single_layer.shape[:2]+(3,) )
    # for k in model.id2code.keys():
    #   output[single_layer==k] = model.id2code[k]
    # cv2.imshow("Output",output)
    # out.write(output)
    ############################

    '''
    When we want to visualize only one class
    Uncomment these line
    '''
    ############################
    line = np.zeros(single_layer.shape[:2])
    line[single_layer == 2] = 255  # New : 2
    
    line_2 = line[70:,:]#.copy() # 90 for 20, 70 for 22
    line_stop = stop_line(line_2)
    #rospy.loginfo("MAT DO STOP: " + str(line_stop))
    
    kernel = np.ones((5,5),np.uint8)

    line = cv2.dilate(line, kernel,iterations = 1)
    

    road = np.zeros((144, 144))
    road[single_layer == 1] = 255  # New : 1


    #road = remove_noise_matrix(line,bamlan,road) # using line or line_opencv # Comment on 11/12
    
    line_counting = count_line(road) # Dem mat do line tren mot duong thang

    road = road[40:,:] # Default : 40
    
    # cv2.imshow("TEST",line_2)
    #rospy.logerr("MAT DO LINE DUONG NGANG: " + str(line_counting) + "    " + str(sign_id))
    
    
    road = cv2.resize(road,(144,144))
    
    # Remove opposite lane
    '''
    This line we remove opposite lane which we not be in there.
    |         |        |
    |         |        |
    | Remove  |  Car   |
    |  Lane   |        |
    |         |        |
    '''  
    
    transform_matrix = pre_processing_in.perspective_transform()
    road_on_birdview = pre_processing_in.birdView(
        road, transform_matrix['M'])

    #rospy.logwarn("DAY NE!!! " + str(cut_road) + "SIGN ID: " + str(sign_id) + " " + str(bamlan))
    if bamlan == 1 and sign_id == 6 and cut_road == True: # Cut duong
        #rospy.logwarn("BAM LAN PHAI THUC HIEN HANH DONG CAT DUONG")
        road_on_birdview = cut_road_speedup(road_on_birdview,width=88)
    if bamlan == 0 and sign_id == 5 and cut_road == True: # Cut duong doan chu D cho mode mid
        #rospy.logwarn("BAM LAN MID THUC HIEN HANH DONG CAT DUONG")
        road_on_birdview = cut_road_speedup_for_mid(road_on_birdview,width=70)
    if bamlan == 0 and sign_id == 4 and cut_road == True:
        #rospy.logwarn("BAM LAN MID THUC HIEN HANH DONG CAT DUONG")
        road_on_birdview = cut_road_speedup_for_mid(road_on_birdview,width=50)

    road_on_birdview = remove_lane_lines(road_on_birdview,bamlan) # Shape (64,144)



    return road_on_birdview,line_counting,line_stop

def sign_turn_callback(sign_turn_data):
    #rospy.loginfo("RECEIVED CLUSTER FROM LIDAR DATA")
    global sign_turn_hello
    sign_turn_hello = sign_turn_data.data
    #rospy.logwarn("SIGN HELLO:    " + str(sign_turn_data.data))
    lcd.update_message("SIGN FROM LIDAR", 0, 0)


def traffic_sign_callback(traffic_sign_id):
    global sign_id
    sign_id = traffic_sign_id.data
    #rospy.loginfo("Received message sign from Classifier:     " +str(traffic_sign_id.data))


def object_detect_callback(object_detect_id):
    global object_id
    object_id = object_detect_id.data
    #rospy.logwarn("Received message sign from OBJ_DETECTION:     " + str(object_detect_id.data))
    lcd.update_message("OBJECT DETECTION", 0, 0)


def avoid_object_callback(avoid_object_id):
    global avoid_id
    avoid_id = avoid_object_id.data
    #rospy.logerr("Received message sign from AVOID OBJECT:     " +str(avoid_object_id.data))
    lcd.update_message("OBJECT AVOIDANCE", 0, 0)

def count_traffic_signs(number_of_signs):
    global count_sign
    count_sign = number_of_signs.data
    #rospy.logerr("Received message sign from SSD TO COUNT SIGNS:     " +str(number_of_signs.data))

def keep_cuting_road(cuting):
    global cut_road
    cut_road = cuting.data
    
def turn_off_ssd_mode(data):
    global turn_off
    turn_off = data.data 
    
def stop_bool_now(data):
    global stop_bool
    stop_bool = data.data        
if __name__ == '__main__':

    rospy.init_node('rightmap_final', anonymous=True)

    ######################### Initialize Variables #########################
    bamlan = rospy.get_param('~bamlan')  # 0
    sign_id = rospy.get_param('~sign_id')  # -1 # Default -1
    first_btn = rospy.get_param('~first_btn')  # False
    sign_turn_hello = rospy.get_param('~sign_turn_hello')  # 0
    object_id = rospy.get_param('~object_id')  # 0
    avoid_id = rospy.get_param('~avoid_id')  # 0
    centroid_x_sign = rospy.get_param('~centroid_x_sign')  # 0
    centroid_y_sign = rospy.get_param('~centroid_y_sign')  # 0
    count_sign = rospy.get_param('~count_sign')  # 0
    check_site = rospy.get_param('~check_site')  # 0
    is_avoid_object = rospy.get_param('~is_avoid_object')  # False
    straight_time = rospy.get_param('~straight_time')
    turnleft_time = rospy.get_param('~turnleft_time')
    turnright_time = rospy.get_param('~turnright_time')
    normal_speed = rospy.get_param('~normal_speed')
    curve_speed = rospy.get_param('~curve_speed')
    reset_all = rospy.get_param('~reset_all') # False
    cuting = rospy.get_param('~cuting')  # False
    cut_road = rospy.get_param('~cut_road') # False
    turn_off = rospy.get_param('~turn_off') # False
    car_seg = rospy.get_param('~car_seg') # False
    stop_bool = rospy.get_param('~stop_bool') # False
    ######################### Initialize Method ##############################
    model = segmentation()
    read = read_input()
    ######################### Initialize Topic ###############################
    lcd = lcd_print("Goodgame", 1, 1)  # Init LCD
    speed = rospy.Publisher("/set_speed", Float32, queue_size=1)
    angle_car = rospy.Publisher("/set_angle", Float32, queue_size=1)
    is_traffic_count = rospy.Publisher("/is_traffic_count", Bool, queue_size=1) # Allow to count traffic signs
    avoid_obj = rospy.Publisher("/is_avoid_obj", Bool, queue_size=1)
    angle = Float32()

    object_detect = rospy.Subscriber("/objects_detect_cluster",
                                     Int8,
                                     object_detect_callback,
                                     queue_size=1)

    traffic_sign_subscribe = rospy.Subscriber("/traffic_sign_id",
                                              Int8,
                                              traffic_sign_callback,
                                              queue_size=1)

    sign_turn = rospy.Subscriber("/sign_detect_cluster",
                                Int8,
                                sign_turn_callback,
                                queue_size=1)

    avoid_object = rospy.Subscriber("/objects_detect_avoid_cluster",
                                    Int8,
                                    avoid_object_callback,
                                    queue_size=1)
    
    count_traffic_signs = rospy.Subscriber("/call_count_sign",
                                    Int8,
                                    count_traffic_signs,
                                    queue_size=1)
    
    keep_cuting = rospy.Subscriber("/keep_cuting",
                                    Bool,
                                    keep_cuting_road,
                                    queue_size=1)
    turnoff_ssd = rospy.Subscriber("/turn_off",
                                    Bool,
                                    turn_off_ssd_mode,
                                    queue_size=1)
        
    stop_bool_ssd = rospy.Subscriber("/stop_bool",
                                    Bool,
                                    stop_bool_now,
                                    queue_size=1)
    ##########################################################################

    ###########################################
    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('/media/goodgame/3332-32338/Log/output.mp4',fourcc, 30.0, (144,144))
    # r = rospy.Rate(10) # 10hz 

    ############################# START ######################################
    while not rospy.is_shutdown():
        # Start time
        count_time = time.time()
        if read.frame is not None:
            pre_processing_in = pre_processing(read.frame)

            ############################## SENSOR IS FALSE ########################
            if read.btn.ss2_status == False:
                speed.publish(0)
                # We publish speed = 0 here
            ################ MODEL PREDICT #################################
            single_layer = model.predict(read.frame)
            road_on_birdview,line_counting,line_stop = process(single_layer,cut_road)
            ''' 
            We will to compute angle to steer car in here
            '''
            angle_degree = compute_centroid(road_on_birdview)
            
            ################################################################
            if read.btn.ss2_status == True and reset_all == False:
                ###############################       RUN       ########################
                #rospy.logwarn("RIGHT-MAP SEMI STARTING !!!")
                #rospy.logwarn("NHAN TIN NHAN CAN CAT DUONG: " + str(cut_road))
                try:
                              
                    #rospy.logerr("MAT DO LINE DUONG TRONG HAM MAIN" + str(line_counting))                                
                    
                    #rospy.loginfo("IS AVOID OBJECT : " + str(is_avoid_object))
                    
                    if object_id == 1 and is_avoid_object == False and check_site == 1 :
                        # if check_site ==1:
                        #rospy.logwarn("MOVE TO OTHER LANE RIGHT NOW!!!")
                        time_obj = time.time()
                        #speed.publish(15)
                        #while time.time()-time_obj <= 0.1:
                            #print(time.time()-time_obj)
                        #    angle_car.publish(60)
                        speed.publish(0) # Test
                        bamlan = -1
                        object_id = 0
                        is_avoid_object = True
                        car_seg = True
                        # them is avoid object thanh false khi count_sign chuyen thanhf 5
                    if avoid_id == 1:
                        #rospy.logwarn("COMEBACK TO LANE!!!")
                        if bamlan == -1:
                            time_avoid = time.time()
                            while time.time()-time_avoid <= 0.3:
                                    angle_car.publish(-60)
                            #rospy.loginfo("MOVE TO RIGHT LANE!!!")
                            bamlan = 1
                        avoid_id = 0
                        car_seg = False
                        # avoid_obj.publish(False)
                        

                        
                    # if count_sign == 6: 
                    #     #speed.publish(17)
                    #     #rospy.loginfo("STOP!!!! MOVE SLOWER IN 17%")
                        
                        
                        
                    ################################# LIDAR #####################################   
                    if sign_turn_hello == 1:
                        #rospy.loginfo("TRAFFIC SIGN CLUSTER IS RECEIVED FROM LIDAR")
                    # if sign_id == 1:  # stop 
                    #     #rospy.loginfo("STOP TRAFFIC SIGN HERE")
                    #     #rospy.loginfo("STOP COUNTING TRAFFIC SIGN RIGHT NOW!!!")
                    #     is_traffic_count.publish(False) # Publish False is denied to count traffic signs
                    #     time_stop = time.time()
                    #     speed.publish(0)
                    #     while time.time()-time_stop < 3:
                    #         # speed.publish(0)
                    #         pass
                    #     # Reset 
                    #     is_traffic_count.publish(True)    
                    #     # Them vao de test stop
                    #     resetAll()
                    #     # Edit on 18/11 in the experiment
                    #     # speed.publish(normal_speed)

                        if sign_id == 3 or sign_id == 2:  # turn right --> Do SSD nhan sai bien re phai thanh re trai nen tam thoi the nay
                            time_tra_1 = time.time()
                            speed.publish(curve_speed)
                            while time.time() - time_tra_1 <= turnleft_time: 
                                angle_car.publish(-60)
                            bamlan = 1
                            sign_id = -1  # Reset bienbao
                        #     bamlan = 1
                            check_site = 2 # 07/12 la 2
                            speed.publish(normal_speed) #flag_angle = 1

                    # if sign_id == 4:  # straight
                    #     time_straight = time.time()
                        
                    #     while time.time()-time_straight <= straight_time: 
                    #         angle_car.publish(-5) 
                            
                    #     # speed.publish(normal_speed)
                    #     sign_id = -1
                    #     bamlan = 1
                    #     check_site = 1

                        sign_turn_hello = 0
                        
                    # if count_sign == 6:
                    #     is_avoid_object = False;                    
                    
                    ################ PUBLISH ANGLE TO CAR #############################
                    ###################################################################
                    ''' Goc duong re trai, goc am re phai'''
                    
                    # if sign_id == 3 and cut_road == True:
                    #     time_tra_1 = time.time()
                    #     print("HERERERERERERERE")
                    #     # speed.publish(curve_speed)
                    #     while (time.time() - time_tra_1) <= turnleft_time: 
                    #         print("HERERE")
                    #         angle_car.publish(-60)                        
                    #     bamlan = 1
                    #     sign_id = -1  # Reset bienbao
                    #     check_site = 2 # 07/12 la 2
                    
                    
                    #if cut_road == True:
                    #    if sign_id == 6 or sign_id == 5:
                    #        speed.publish(17) # Khi gap bien no-left, no-right thi di toc do 17
                    # Doan chu B di bam lan phai gap bien no-right
                    
                    if bamlan == 1 and line_counting < 120 and cut_road == True and sign_id == 6:
                        #rospy.logerr("CAN PHAI RE KHI GAP BIEN NO RIGHT")
                        start_time = time.time()
                        speed.publish(curve_speed)
                        while(time.time() - start_time < 0.8):
                            angle_degree = 60
                            angle.data = angle_degree
                            angle_car.publish(angle)   
                        speed.publish(normal_speed)
                        # Reset SignID
                        sign_id = -1
                        
                    # Doan chu D gap bien no left
                    if bamlan == 0 and line_counting < 120 and cut_road == True and sign_id == 5:
                        # Gap No-left nhung can phai re phai
                        lcd.update_message("CAN RE",0,0)
                        #rospy.logerr("Gap No-left nhung can phai re phai")
                        start_time = time.time()
                        speed.publish(curve_speed)
                        while(time.time() - start_time < 0.8):
                            angle_degree = -60
                            angle.data = angle_degree
                            angle_car.publish(angle)   
                        speed.publish(normal_speed)
                        #bamlan = 1
                        # Reset SignID
                        sign_id = -1                        
                    
                    if bamlan == 0 and sign_id == 4 and turn_off == True:
                        # print("RETURNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
                        sign_id = -1
                        check_site = 1
                        bamlan = 1
                        turn_off = False

                    if line_stop > 90 and stop_bool == True:  # stop 
                        #rospy.loginfo("STOP TRAFFIC SIGN HERE")
                        #rospy.loginfo("STOP COUNTING TRAFFIC SIGN RIGHT NOW!!!")
                        is_traffic_count.publish(False) # Publish False is denied to count traffic signs
                        time_stop = time.time()
                        speed.publish(0)
                        while time.time()-time_stop < 3:
                            # speed.publish(0)
                            # print("HEHEE")
                            pass
                        # Reset 
                        stop_bool = False
                        is_traffic_count.publish(True)   
                        avoid_obj.publish(False)
                        # Them vao de test stop
                        resetAll()
                        # Edit on 18/11 in the experiment
                        # speed.publish(normal_speed)
                                                    
                    # print("CAR SEG",car_seg)    
                    if bamlan == 1 and angle_degree > 10 and car_seg == False : 
                        # print("DANG O LAN PHAI, KHUC CUA CAN RE TRAI, RE HET TOC LUC ")
                        angle_degree = angle_degree * 3.5 #
                        speed.publish(curve_speed)
                    if bamlan == 1 and angle_degree < -10 and car_seg == False :
                        # print("DANG O LAN PHAI, KHUC CUA CAN RE PHAI, RE HET TOC LUC")
                        angle_degree =  angle_degree * 3.25 
                        speed.publish(curve_speed)
                    if bamlan == -1 and angle_degree > 10 and car_seg == False :
                        # print("DANG O LAN TRAI, KHUC CUA CAN RE TRAI, RE HET TOC LUC")
                        angle_degree = angle_degree * 3.25 #60
                        speed.publish(curve_speed)
                    if bamlan == -1 and angle_degree < -10 and car_seg == False :
                        # print("DANG O LAN TRAI, KHUC CUA CAN RE PHAI, RE HET TOC LUC")
                        angle_degree = angle_degree * 3.25 #-60
                        speed.publish(curve_speed)
                    if bamlan == 1 or bamlan == -1 and car_seg == False:
                       if angle_degree > -12 and angle_degree < 12:
                           speed.publish(normal_speed)
                    if bamlan == 0 and angle_degree > 20 and count_sign != 6:
                        angle_degree = angle_degree * 2.5
                        speed.publish(curve_speed)
                    if bamlan == 0 and angle_degree < -20 and count_sign != 6:
                        angle_degree = angle_degree * 2.5
                        speed.publish(curve_speed)
                    if bamlan == 0 and count_sign != 6:
                       if angle_degree > -20 and angle_degree < 20:
                           speed.publish(normal_speed)          
                    
                    ##################################################################
                    ############################ ANGLE FOR CAR #######################
                    if bamlan == -1 and angle_degree > 12 and car_seg == True :
                        # print("DANG O LAN TRAI, KHUC CUA CAN RE TRAI, RE HET TOC LUC")
                        angle_degree = angle_degree * 2.5 #60
                        speed.publish(curve_speed)
                    if bamlan == -1 and angle_degree < -12 and car_seg == True :
                        # print("DANG O LAN TRAI, KHUC CUA CAN RE PHAI, RE HET TOC LUC")
                        angle_degree = angle_degree * 2.5 #-60
                        speed.publish(curve_speed)
                    if bamlan == 1 or bamlan == -1 and car_seg == True:
                        if angle_degree > -12 and angle_degree < 12:
                           speed.publish(normal_speed)                                                
                    ###################################################################
                    
                    # rospy.logwarn("ANGLE OF CAR: " + str(angle_degree))

                    angle.data = angle_degree
                    
                    angle_car.publish(angle)      
                    
                    ################################ LCD PRINT #########################
                    fps = int(1 // (time.time() - count_time))
                    #rospy.loginfo("FPS " + str(fps) )#+ "L " + str(line_counting))
                    lcd.update_message("RIGHT-MAP FINAL",0,0)
                    if bamlan == 1:
                        lcd.update_message(
                            "Mode Right" + "  FPS " + str(fps) + "    ", 0, 1)
                    elif bamlan == -1:
                        lcd.update_message(
                            "Mode Left " + "  FPS " + str(fps) + "    ", 0, 1)
                    elif bamlan == 0:
                        lcd.update_message(
                            "Mode Mid" + "  FPS " + str(fps) + "    ", 0, 1)
                    lcd.update_message("Angle " + str(round(angle_degree, 1)) + " Count " + str(count_sign) + "   ", 0, 3)
                    cv2.waitKey(1)
                except Exception as e:
                    print(str(e))
            
            read.btn.led_send_message(True)

            if read.btn.bt1_bool:
                print("BAM LAN PHAI")
                first_btn = True
                reset_all = False
                bamlan = 1 
                speed.publish(normal_speed)
                
            if read.btn.bt2_bool:
                print("BAM LAN TRAI")
                bamlan = -1
                reset_all = False

            if read.btn.bt3_bool:
                print("RETURN TO NORMAL MODE")
                reset_all = False
                bamlan = 0
                speed.publish(normal_speed)
                
            if read.btn.bt4_bool:
                print("RESET ALL VARIABLES!!!")
                resetAll()
                # read.btn.ss2_status = False
                read.btn.led_send_message(False)
                reset_all = True
                avoid_obj.publish(False)
                speed.publish(0)
                lcd.update_message("RESET ALL...", 0, 0)
                lcd.clear()
        # r.sleep() # Very important
        
    # out.release()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    # cv2.destroyAllWindows()
    

    
