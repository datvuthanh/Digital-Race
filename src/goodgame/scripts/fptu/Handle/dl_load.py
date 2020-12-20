#!/usr/bin/env python3

from fptu.Handle.library import *


def load_model(url):
    f = gfile.FastGFile(url, 'rb')
    
    graph_def = tf.GraphDef()
    
    # Parses a serialized binary message into the current message.
    
    graph_def.ParseFromString(f.read())
    
    f.close()

    return graph_def


def parse_code(l):
    '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
    '''
    if len(l.strip().split("\t")) == 2:
        a, b = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c


def class_return():
    
    url = rospy.get_param("~label_colors")
    label_codes, label_names = zip(*[parse_code(l) for l in open(url)])

    label_codes, label_names = list(label_codes), list(label_names)

    id2code = {k:v for k,v in enumerate(label_codes)}

    return id2code
