# The pre-trained model class

import os

import numpy as np
import tensorflow as tf

data_dir = "models/mobilenet_v1/"

# TF Graph
path_graph_def = "graph_freeze.pb"


class MobileNet:
    # Name of the tensor for feeding the input image.
    tensor_name_input_image = 'image:0'


    # Names of layers to be used, use get_all_layer_names to find them
    layer_names = ['MobilenetV1/Conv2d_0/Conv2D',
                   'MobilenetV1/Conv2d_1_depthwise/depthwise',
                   'MobilenetV1/Conv2d_2_depthwise/depthwise',
                   'MobilenetV1/Conv2d_3_depthwise/depthwise',
                   'MobilenetV1/Conv2d_4_depthwise/depthwise',
                   'MobilenetV1/Conv2d_5_depthwise/depthwise',
                   'MobilenetV1/Conv2d_6_depthwise/depthwise',
                   'MobilenetV1/Conv2d_7_depthwise/depthwise',
                   'MobilenetV1/Conv2d_8_depthwise/depthwise',
                   'MobilenetV1/Conv2d_9_depthwise/depthwise',
                   'MobilenetV1/Conv2d_10_depthwise/depthwise',
                   'MobilenetV1/Conv2d_11_depthwise/depthwise',
                   'Conv2d_3_pool']

    def __init__(self):
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')

            # Get a reference to the tensor for inputting images to the graph.
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # Get references to the tensors for the commonly used layers.
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def get_layer_tensors(self, layer_ids):
        return [self.layer_tensors[idx] for idx in layer_ids]

    def get_layer_names(self, layer_ids):
        return [self.layer_names[idx] for idx in layer_ids]

    def get_all_layer_names(self, startswith=None):
        names = [op.name for op in self.graph.get_operations()]
        if startswith is not None:
            names = [name for name in names if name.startswith(startswith)]

        return names

    def create_feed_dict(self, image):
        """
        Create and return a feed-dict with an image.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Dict for feeding to the graph in TensorFlow.
        """
        image = np.expand_dims(image, axis=0)

        feed_dict = {self.tensor_name_input_image: image}

        return feed_dict


'''
    layer_names = ['MobilenetV1/Conv2d_0/Conv2D',
                   'MobilenetV1/Conv2d_1_depthwise/depthwise',

                   'MobilenetV1/Conv2d_1_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_2_depthwise/depthwise',
                   'MobilenetV1/Conv2d_2_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_3_depthwise/depthwise',
                   'MobilenetV1/Conv2d_3_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_4_depthwise/depthwise',
                   'MobilenetV1/Conv2d_4_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_5_depthwise/depthwise',
                   'MobilenetV1/Conv2d_5_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_6_depthwise/depthwise',
                   'MobilenetV1/Conv2d_6_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_7_depthwise/depthwise',
                   'MobilenetV1/Conv2d_7_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_8_depthwise/depthwise',
                   'MobilenetV1/Conv2d_8_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_9_depthwise/depthwise',
                   'MobilenetV1/Conv2d_9_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_10_depthwise/depthwise',
                   'MobilenetV1/Conv2d_10_pointwise/Conv2D',

                   'MobilenetV1/Conv2d_11_depthwise/depthwise',
                   'MobilenetV1/Conv2d_11_pointwise/Conv2D',

                   'Conv2d_3_pool']
'''
