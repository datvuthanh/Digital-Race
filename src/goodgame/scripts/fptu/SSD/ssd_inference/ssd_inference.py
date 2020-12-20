#!/usr/bin/env python3

from fptu.SSD.ssd_inference.library import *

from fptu.SSD.ssd_inference.dl_load import load_model

from fptu.SSD.ssd_inference.model import detection

import math

from numba import njit

@njit
def find_largest_box(boxes):    
    largest_box = 0
    index = 0
    if len(boxes) != 0:
        for i in range(len(boxes)):
            box = (boxes[i])
            # xmin = box[2]
            # ymin = box[3]
            # xmax= box[4]
            # ymax= box[5]
            # triagonal = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            # if triagonal > largest_box:
            #     index = i
    return index

'''
STOP
LEFT
RIGHT
STRAIGHT
NOLEFT
NORIGHT
'''

# def sign_turn_callback(sign_turn_data):
#     global sign_region_detection
#     sign_region_detection = sign_turn_data.data
#     rospy.logwarn("SIGN REGION DETECTION FROM LIDAR NODE:    " + str(sign_turn_data.data))
#     lcd.update_message("SIGN FROM LIDAR", 0, 0)

def is_traffic_count(allow_data):
    global is_counting
    is_counting = allow_data.data
    rospy.logwarn("STOP COUNTING FROM SEGMENT NODE    " + str(is_counting))
    
if __name__ == '__main__':

    rospy.init_node('ssd_rt', anonymous=True)
    read = read_input()
    ssd_model = detection()
    lcd = lcd_print("Goodgame",1,1) # Init LCD
    traffic_sign_publish = rospy.Publisher("/traffic_sign_id",Int8,queue_size = 1)    
    count_sign_pl = rospy.Publisher("/call_count_sign", Int8, queue_size=1)
    
    keep_cuting = rospy.Publisher("/keep_cuting", Bool, queue_size=1)
    
    turn_off = rospy.Publisher("/turn_off", Bool, queue_size=1)
    
    stop_bool = rospy.Publisher("/stop_bool", Bool, queue_size=1)

    # sign_turn = rospy.Subscriber("/sign_detect_cluster",
    #                         Int8,
    #                         sign_turn_callback,
    #                         queue_size=1)
    
    allow_counting = rospy.Subscriber("/is_traffic_count",
                            Bool,
                            is_traffic_count,
                            queue_size=1)    
    
    
    # sign_region_detection = 0
    count_sign_mess = Int8()

    last_index = 0
    sign_id = 0
    last_sign_id = 0
    number_of_object = 0
    is_counting = True
    last_count = 0
    miss_frame = 0
    sign_count = 0
    keep_tracking = 0
    miss_frame_tracking = 0
    start_counting = False
    count_ssd = 0
    #stop_bool  = False
    ############################# START ######################################
    while not rospy.is_shutdown():
        # Start time

        if read.frame is not None:
            if read.btn.ss2_status == False: 
                pass
            try:
                # if number_of_object > 20:
                #     number_of_object = 0
                start_time = time.time()
                img = read.frame#[0:360,0:640]
                
                # cv2.imwrite('/media/goodgame/3332-323312/SSD/SSD_1412/SSD_' + str(count_ssd) +'.png',read.frame)
                img = cv2.resize(img,(240,180))
                # vis_img = img#.copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.reshape(img,(1,180, 240, 3))
                
                pred = ssd_model.predict(img)
                # print("CLASSS::: ", len(pred[0]))
                classes = ['Background','Stop','Left','Right','Straight','NoLeft','NoRight']

                largest_box = 0
                index = 0
                boxes = pred[0]
                if len(boxes) != 0:
                    for i in range(len(boxes)):
                        box = (boxes[i])
                        label = int(box[0])
                        conf = (box[1])
                        xmin = int(box[2])
                        ymin = int(box[3])
                        xmax= int(box[4])
                        ymax= int(box[5])
                        triagonal = (xmax - xmin) * (ymax -ymin)
                        # vis_img = cv2.rectangle(vis_img, (xmin,ymin), (xmax,ymax), (0,0,0), 1) 
                        # vis_img = cv2.putText(vis_img, str(classes[label]) + " " + str(conf) + " " + str(largest_box), (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.2, (255,255,255), 1, cv2.LINE_AA)
                        if triagonal > largest_box:
                            index = i
                            largest_box = triagonal
                            
                last_index = index # Gan index lan cuoi cung = index
                
                # print("LAST INDEX: ", last_sign_id)
                # print("INDEX: ", sign_id)
                
                # rospy.loginfo("DIEN TICH: " + str(largest_box))
                
                if len(pred[0]) != 0 and read.btn.ss2_status == True:  
                    last_sign_id = sign_id
                    last_count = number_of_object              
                    box = pred[0][index]
                    label = int(box[0])
                    conf= box[1]
                    xmin = int(box[2])
                    ymin = int(box[3])
                    xmax= int(box[4])
                    ymax=int(box[5])
                    
                    # Gan id
                    sign_id = label

                    if conf > 0.9: ########## STOP
                        if label == 1:
                            # print("STOPPPPPPPPPPPPPPPPPPPPPPPPPPP!!!!")
                            traffic_sign_publish.publish(label) # Publish ID to Deep Handle
                            stop_bool.publish(True)
                    if conf > 0.9 and largest_box > 800 and is_counting == True: # DIEN TICH : 600 ~ 1m #  and  800 ~ 75cm
                        if last_sign_id == sign_id: 
                            #rospy.logwarn("CO BIEN BAO")
                            lcd.update_message(str(classes[label]),2,2)        
                            traffic_sign_publish.publish(label) # Publish ID to Deep Handle
                            #Update Traffic sign + 1
                            if sign_id == 4 or sign_id == 3 or sign_id ==2:
                                start_counting = True
                            if start_counting == True:
                                number_of_object += 1  # Update number of signs
                            #Publish number of signs to other packages
                            count_sign_mess.data = number_of_object
                            count_sign_pl.publish(count_sign_mess)
                            ######################## SLEEP #################################
                            sleep_time = time.time()
                            #rospy.logerr("LABELLLLLLL: " + str(classes[label]))
                            if label == 6: # Neu sign la noleft or noright --> publish tin nhan cho phep cat duong ben deep handle
                                #print("CAT DUONG DI!!!")
                                keep_cuting.publish(True)
                                while (time.time() - sleep_time < 1.2):
                                    pass
                            if label == 5:
                                keep_cuting.publish(True)
                                while (time.time() - sleep_time < 1):
                                    pass
                            if label == 4: #Straight
                                keep_cuting.publish(True)
                                while(time.time() - sleep_time < 1.2):
                                    pass
                                turn_off.publish(True)
                            if label == 2 or label == 3: #Right or left
                                keep_cuting.publish(True)
                                while(time.time() - sleep_time < 1.2):
                                    pass    
                                    #print(time.time() - sleep_time)
                                #turn_off.publish(True)
                            keep_cuting.publish(False)    
                            # sign_region_detection = 0 #  Reset LiDAR
                            
                            # if sign_id == 4 or sign_id == 3 or sign_id == 2:
                            #    start_counting = True
                                

                # print("COUNT: ", last_count,number_of_object)
                
                ####################### METHOD 2 ####################################
                # if number_of_object != last_count:
                #     print("MISS FRAME")
                #     miss_frame += 1
                # if number_of_object == last_count and number_of_object != 0:
                #     keep_tracking = number_of_object
                #     miss_frame_tracking += 1
                #     print("HERE")
                # if miss_frame > 10 or miss_frame_tracking > 10:
                #     rospy.logerr("DA VUOT QUA!!!")
                #     print("VUOT QUA BIEN")
                #     print("CAN THUC HIEN HANH DONG")
                #     number_of_object = 0
                #     last_count = 0
                #     miss_frame = 0
                #     miss_frame_tracking = 0
                #     traffic_sign_publish.publish(sign_id) # Publish ID to Deep Handle
                #     keep_cuting.publish(True)
                #     sleep_time = time.time()
                #     # while (time.time() - sleep_time < 0.5):
                #     #     print(time.time() - sleep_time)
                #     keep_cuting.publish(False)    

                ####################### METHOD 3 ####################################
                
                #Reset counting when car meet stop
                if is_counting == False:
                    start_counting = False
                    number_of_object = 0   
                if read.btn.bt4_bool:
                    last_index = 0
                    number_of_object = 0
                    start_counting = False
                if read.btn.bt3_bool == True or read.btn.bt2_bool == True or read.btn.bt1_bool == True:
                    is_counting = True               
                # rospy.loginfo("NUMBER OF TRAFFIC SIGNS : \t" + str(number_of_object))
                
                ############################ LOG DATA ###################################
                # vis_img = cv2.resize(vis_img,(640,480))
                # cv2.imshow("Frame",vis_img)
                
                # cv2.imwrite('/media/goodgame/3332-323312/SSD/SSD_1412_pred/SSD_' + str(count_ssd) +'.png',vis_img)
                
                # count_ssd += 1
    
                fps = int(1 // (time.time() - start_time))
                #rospy.loginfo("FPS IN SSD FRAME: " + str(fps))            
                cv2.waitKey(1)
            except Exception as e: 
                print(str(e))
            
    # out.release()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

    # cv2.destroyAllWindows()
