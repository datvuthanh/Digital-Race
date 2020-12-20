#!/usr/bin/env python3

from fptu.SSD.ssd_inference.library import *


def load_model(url):
    f = gfile.FastGFile(url, 'rb')
    
    graph_def = tf.GraphDef()
    
    # Parses a serialized binary message into the current message.
    
    graph_def.ParseFromString(f.read())
    
    f.close()

    return graph_def
