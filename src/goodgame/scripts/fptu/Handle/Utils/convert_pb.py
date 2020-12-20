
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)


import segmentation_models as sm
from segmentation_models import PSPNet

# sm.set_framework('keras')

# Create Model
model = PSPNet(backbone_name='efficientnetb3',
                input_shape=(144,144,3),
                classes=4,
                activation='softmax',
                weights= None,#'/home/memi/Desktop/Goodgame/models/goodgame_epoch-031_loss-0.5104_val_loss-0.9101_iouscore-0.8130_f1score-0.8763.h5',
                encoder_weights='imagenet',
                encoder_freeze=True,
                downsample_factor=8,
                psp_conv_filters=512,
                psp_pooling_type='avg',
                psp_use_batchnorm=True,
                psp_dropout=0.25)

model.load_weights('/home/goodgame/Downloads/GG_model_1_12_epoch-001_loss-0.7875_val_loss-0.7807.h5')
    
    
print(model.outputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
print(model.inputs)
# [<tf.Tensor 'conv2d_1_input:0' shape=(?, 28, 28, 1) dtype=float32>]

from keras import backend as K
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

# Save to ./model/tf_model.pb
tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)
