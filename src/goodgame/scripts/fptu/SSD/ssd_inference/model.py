#!/usr/bin/env python3

from fptu.SSD.ssd_inference.library import *

from fptu.SSD.ssd_inference.dl_load import load_model

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast


class detection:
    def __init__(self):

        K.clear_session() # Clear previous models from memory.
        
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
        
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.graph = tf.get_default_graph()
        
        url = rospy.get_param("~ssd_model") #"./ssd_models/ssd_1212.pb"

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

    def predict(self,image):
        with self.sess.as_default():
            with self.graph.as_default():
                
                softmax_tensor = self.sess.graph.get_tensor_by_name('import/predictions/concat:0')
                
                pred = self.sess.run(softmax_tensor, {'import/input_1:0': np.array(image)})        
                
                y_pred_decoded = decode_detections(pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.2,
                                   top_k=200,
                                   normalize_coords=True,
                                   img_height=180,
                                   img_width=240)
                
                return y_pred_decoded        
