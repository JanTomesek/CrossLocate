#!/usr/bin/python3
"""
	Neural Network

	Created by Jan Tomesek on 18.2.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "tomesek.j@gmail.com"
__copyright__ = "Copyright 2019, Jan Tomesek"


import tensorflow as tf
import numpy as np


from enum import Enum

class Architecture(Enum):
    ALEXNET_SIMPLE = 1
    ALEXNET_SPLIT = 2
    VGG16 = 3

class Pooling(Enum):
    NONE = 0
    MAC = 1
    SPOC = 2
    NETVLAD = 3

class FeatureNormalization(Enum):
    NONE = 0
    L2 = 1


class NeuralNetwork():
    """
        Superclass providing methods for assembly of neural networks.
        Provides methods for creating variables.
        Provides methods for creating standard layers.
        Provides method for creating aggregation layers.
    """

    def __init__(self):
        return

    def getConvolutionVars(self, kernelShape, numberOfKernels, trainable=True):
        kernels = tf.Variable(tf.truncated_normal((*kernelShape, numberOfKernels), stddev=1e-1, dtype=tf.float32), trainable=trainable, name='kernels')
        biases = tf.Variable(tf.constant(0.0, shape=[numberOfKernels], dtype=tf.float32), trainable=trainable, name='biases')

        self.parameters += [kernels, biases]
        return kernels, biases

    def getFullyConnectedVars(self, inputDims, outputDims):
        weights = tf.Variable(tf.truncated_normal([inputDims, outputDims], stddev=1e-1, dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.constant(1.0, shape=[outputDims], dtype=tf.float32), name='biases')

        self.parameters += [weights, biases]
        return weights, biases

    def convolution2D(self, input, kernels, biases, stride=1, padding='SAME', withRelu=True, ID=''):
        conv = tf.nn.conv2d(input, kernels, strides=[1, stride, stride, 1], padding=padding)
        conv = tf.nn.bias_add(conv, biases)
        if (withRelu):
            conv = tf.nn.relu(conv)

        print('[{}] Conv{}: \t\t{}'.format(self.tag, ID, conv.shape)
              + (' \t(not trainable)' if (not kernels.trainable or not biases.trainable) else ''))
        return conv

    def splitConvolution2D(self, input, kernels, biases, stride=1, padding='SAME', withRelu=True, ID=''):
        # Splits inputs (channels) into half and kernels (number) into half (preserving kernel depths, corresponding to split inputs)
        inputSplits = tf.split(input, 2, axis=3)
        kernelsSplits = tf.split(kernels, 2, axis=3)
        # Computes convolutions on split inputs (generating half of the normal activation maps)
        convSplits = [tf.nn.conv2d(input, kernels, strides=[1, stride, stride, 1], padding=padding) for input, kernels in zip(inputSplits, kernelsSplits)]
        # Concatenates convolution activation maps
        conv = tf.concat(convSplits, axis=3)

        conv = tf.nn.bias_add(conv, biases)
        if (withRelu):
            conv = tf.nn.relu(conv)

        print('[{}] SplitConv{}: \t{}'.format(self.tag, ID, conv.shape)
              + ('\t(not trainable)' if (not kernels.trainable or not biases.trainable) else ''))
        return conv

    def maxPooling2D(self, input, k=2, s=2, padding='SAME', ID=''):
        maxPool = tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)

        print('[{}] MaxPool{}:\t\t{}'.format(self.tag, ID, maxPool.shape))
        return maxPool

    def localResponseNormalization(self, input, depthRadius, bias, alpha, beta, ID=''):
        lrNorm = tf.nn.local_response_normalization(input, depth_radius=depthRadius, bias=bias, alpha=alpha, beta=beta)

        print('[{}] LRNorm{}: \t\t{}'.format(self.tag, ID, lrNorm.shape))
        return lrNorm

    def fullyConnected(self, input, weights, biases, withRelu=True, ID=''):
        fc = tf.matmul(input, weights)
        fc = tf.nn.bias_add(fc, biases)
        if (withRelu):
            fc = tf.nn.relu(fc)

        print('[{}] FC{}: \t\t{}'.format(self.tag, ID, fc.shape))
        return fc

    def flatten(self, input):
        flatShape = int(np.prod(input.get_shape()[1:]))
        flat = tf.reshape(input, [-1, flatShape])

        print('[{}] Flatten: \t\t{}'.format(self.tag, flat.shape))
        return flat, flatShape

    def aggregationLayers(self, input, preFeatNorm, pooling, postFeatNorm):
        normInput = input

        if (preFeatNorm is FeatureNormalization.L2):
            with tf.name_scope('pre_feat_norm'):
                normInput = tf.nn.l2_normalize(normInput, axis=-1)
            print('[{}] L2-norm: \t\t{}'.format(self.tag, normInput.shape))

        if (pooling is Pooling.MAC):
            with tf.name_scope('mac'):
                self.mac = tf.reduce_max(normInput, axis=[1, 2])

                print('[{}] MAC: \t\t{}'.format(self.tag, self.mac.shape))
                output = self.mac

        elif (pooling is Pooling.SPOC):
            with tf.name_scope('spoc'):
                self.spoc = tf.reduce_sum(normInput, axis=[1, 2])

                print('[{}] SPoC: \t\t{}'.format(self.tag, self.spoc.shape))
                output = self.spoc

        elif (pooling is Pooling.NETVLAD):
            import models.VGG.uzhrpg_layers as uzhrpg_layers

            with tf.name_scope('netvlad'):
                # NetVLAD
                x = uzhrpg_layers.netVLAD(normInput, 64, ID=self.ID)

                # PCA
                x = tf.layers.conv2d(tf.expand_dims(tf.expand_dims(x, 1), 1),
                                     4096, 1, 1, name='WPCA'+'_'+str(self.ID))
                self.netvlad = tf.layers.flatten(x)

                print('[{}] NetVLAD: \t\t{}'.format(self.tag, self.netvlad.shape))
                output = self.netvlad

        normOutput = output

        if (postFeatNorm is FeatureNormalization.L2):
            with tf.name_scope('post_feat_norm'):
                normOutput = tf.nn.l2_normalize(normOutput, axis=-1)
            print('[{}] L2-norm: \t\t{}'.format(self.tag, normOutput.shape))

        return normOutput
