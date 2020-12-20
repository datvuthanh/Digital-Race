#!/usr/bin/env python3

from fptu.Handle.library import *

from fptu.Handle.dl_load import load_model,parse_code,class_return

class segmentation:
    
    def __init__(self):

        K.clear_session() # Clear previous models from memory.
        
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
        
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.graph = tf.get_default_graph()
        
        # url = "/home/goodgame/catkin_ws/src/semi_final/scripts/fptu/Handle/Utils/Tensor_RT Models/Model_2109_144_rb.pb"
        url = rospy.get_param("~pspnet_model") #"./Model/PSPNET_3classes/seg_datvu_0812_8846_9342.pb"

        graph_def = load_model(url)
        
        self.sess = tf.InteractiveSession(config=config)

        self.sess.graph.as_default()
        
        K.set_session(self.sess) 

        with self.sess.as_default():       
            with self.graph.as_default(): 
            # Import a serialized TensorFlow `GraphDef` protocol buffer
            # and place into the current default `Graph`.
                tf.import_graph_def(graph_def)

        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)

        self.id2code = class_return()

        # Publish centroid of traffic sign to classifiers.py
        self.sign_pub = rospy.Publisher('/localization', numpy_msg(Floats),queue_size=1)
        

    def onehot_to_rgb(self,onehot, colormap):

        '''Function to decode encoded mask labels
            Inputs: 
                onehot - one hot encoded image matrix (height x width x num_classes)
                colormap - dictionary of color to label id
            Output: Decoded RGB image (height x width x 3) 
        '''
        
        single_layer = np.argmax(onehot, axis=-1) # Dat Vu convert numpy to tensorflow

        return single_layer

    def predict(self,frame):
        with self.sess.as_default():
            with self.graph.as_default():
                
                img = frame[220:,:]
                img = cv2.resize(img,(144,144))
                
                # test = img.copy()
                
                # test = test[20:,:]
                # cv2.imshow("Frame",img)
                
                # cv2.imwrite('/home/goodgame/Desktop/image/road.png',img)
                #img_cnn = img.copy()
                # The code is implemented by Dat Vu on 28/04
                '''
                We can get line of road without using deep learning by opencv method (useful)
                We want to find line of road by convert image to HSV color
                Get line by HSV range 
                '''
                # ## Pre-processing
                # lower = np.array([0, 0, 215]) #### ---> Modify here
                # upper = np.array([179, 255, 255])
                # # Create HSV Image and threshold into a range.
                # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # mask = cv2.inRange(hsv, lower, upper)
                # cv2.imshow("LINE OPENCV", mask)
                # line_opencv = mask[64:,:] # We want to get half of image below 
                ############################################
                image = (img[...,::-1].astype(np.float32)) / 255.0
                
                image = np.reshape(image, (1, 144, 144, 3))

                softmax_tensor = self.sess.graph.get_tensor_by_name('import/softmax/truediv:0')
                
                predict_one = self.sess.run(softmax_tensor, {'import/input_1:0': np.array(image)})
                
                # print(predict_one[0])
                # In here, I've just add a parameter into onehot_to_rgb method.
                # Before the method only have two parameters (predict_one[0], self.id2code)
                image = self.onehot_to_rgb(predict_one[0],self.id2code)
                
                #Test for spped up

                #image = onehot_to_rgb_speedup(predict_one[0])

                #cv2.imshow("Prediction",image) # If you use local machine cv2.imshow instead of cv2_imshow  
                
                return image 