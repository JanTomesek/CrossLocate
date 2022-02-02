#!/usr/bin/python3
"""
	VGG-16 architecture

	Created by Jan Tomesek on 2.5.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "tomesek.j@gmail.com"
__copyright__ = "Copyright 2019, Jan Tomesek"


import tensorflow as tf
import numpy as np

from networks.NeuralNetwork import Pooling, FeatureNormalization

from networks.NeuralNetwork import NeuralNetwork

class VGG16(NeuralNetwork):
    """
        Assembles VGG-16 architecture with optional fully connected layers.
        Loads pretrained weights as network parameters.
    """

    def __init__(self, ID, input, pooling,
                 preFeatNorm=FeatureNormalization.NONE, postFeatNorm=FeatureNormalization.NONE,
                 trainableFromLayer=None, numberOfClasses=1000, descriptorsLayer=None):
        self.tag = 'VGG16 ({})'.format(ID)

        self.ID = ID
        self.input = input
        self.preFeatNorm = preFeatNorm
        self.pooling = pooling
        self.postFeatNorm = postFeatNorm

        self.trainableFromLayer = trainableFromLayer

        self.numberOfClasses = numberOfClasses

        self.parameters = []

        self.output = self.__architecture()

        if (descriptorsLayer is not None):
            if (descriptorsLayer == 'fc2'):
                self.descriptors = self.fc2

    def __architecture(self):
        print('[{}] Input: \t\t{}'.format(self.tag, self.input.shape))
        print('')

        outputConvLayer = self.__convolutionalLayers()
        print('')

        if (self.pooling is Pooling.NONE):
            self.logits, outputLayer = self.__fullyConnectedLayers(outputConvLayer)
        else:
            outputLayer = self.aggregationLayers(outputConvLayer, self.preFeatNorm, self.pooling, self.postFeatNorm)

        return outputLayer

    def __convolutionalLayers(self):
        if (self.trainableFromLayer is None):
            trainable = True
        else:
            trainable = False

        with tf.name_scope('conv1'):
            kernels, biases = self.getConvolutionVars((3, 3, 3), 64, trainable=trainable)
            self.conv1 = self.convolution2D(self.input, kernels, biases, ID=1)

        with tf.name_scope('conv2'):
            kernels, biases = self.getConvolutionVars((3, 3, 64), 64, trainable=trainable)
            self.conv2 = self.convolution2D(self.conv1, kernels, biases, ID=2)

        with tf.name_scope('maxpool1'):
            self.maxPool1 = self.maxPooling2D(self.conv2, ID=1)

        with tf.name_scope('conv3'):
            kernels, biases = self.getConvolutionVars((3, 3, 64), 128, trainable=trainable)
            self.conv3 = self.convolution2D(self.maxPool1, kernels, biases, ID=3)

        with tf.name_scope('conv4'):
            kernels, biases = self.getConvolutionVars((3, 3, 128), 128, trainable=trainable)
            self.conv4 = self.convolution2D(self.conv3, kernels, biases, ID=4)

        with tf.name_scope('maxpool2'):
            self.maxPool2 = self.maxPooling2D(self.conv4, ID=2)

        with tf.name_scope('conv5'):
            kernels, biases = self.getConvolutionVars((3, 3, 128), 256, trainable=trainable)
            self.conv5 = self.convolution2D(self.maxPool2, kernels, biases, ID=5)

        with tf.name_scope('conv6'):
            kernels, biases = self.getConvolutionVars((3, 3, 256), 256, trainable=trainable)
            self.conv6 = self.convolution2D(self.conv5, kernels, biases, ID=6)

        with tf.name_scope('conv7'):
            kernels, biases = self.getConvolutionVars((3, 3, 256), 256, trainable=trainable)
            self.conv7 = self.convolution2D(self.conv6, kernels, biases, ID=7)

        with tf.name_scope('maxpool3'):
            self.maxPool3 = self.maxPooling2D(self.conv7, ID=3)

        with tf.name_scope('conv8'):
            kernels, biases = self.getConvolutionVars((3, 3, 256), 512, trainable=trainable)
            self.conv8 = self.convolution2D(self.maxPool3, kernels, biases, ID=8)

        with tf.name_scope('conv9'):
            kernels, biases = self.getConvolutionVars((3, 3, 512), 512, trainable=trainable)
            self.conv9 = self.convolution2D(self.conv8, kernels, biases, ID=9)

        with tf.name_scope('conv10'):
            kernels, biases = self.getConvolutionVars((3, 3, 512), 512, trainable=trainable)
            self.conv10 = self.convolution2D(self.conv9, kernels, biases, ID=10)

        with tf.name_scope('maxpool4'):
            self.maxPool4 = self.maxPooling2D(self.conv10, ID=4)

        if (self.trainableFromLayer == 'conv11'):
            trainable = True

        with tf.name_scope('conv11'):
            kernels, biases = self.getConvolutionVars((3, 3, 512), 512, trainable=trainable)
            self.conv11 = self.convolution2D(self.maxPool4, kernels, biases, ID=11)

        with tf.name_scope('conv12'):
            kernels, biases = self.getConvolutionVars((3, 3, 512), 512, trainable=trainable)
            self.conv12 = self.convolution2D(self.conv11, kernels, biases, ID=12)

        with tf.name_scope('conv13'):
            kernels, biases = self.getConvolutionVars((3, 3, 512), 512, trainable=trainable)
            self.conv13 = self.convolution2D(self.conv12, kernels, biases, withRelu=(self.pooling is Pooling.NONE), ID=13)

        if (self.pooling is not Pooling.NONE):
            return self.conv13

        with tf.name_scope('maxpool5'):
            self.maxPool5 = self.maxPooling2D(self.conv13, ID=5)

        return self.maxPool5

    def __fullyConnectedLayers(self, input):
        with tf.name_scope('flatten'):
            self.flat, flatShape = self.flatten(input)

        with tf.name_scope('fc1'):
            weights, biases = self.getFullyConnectedVars(flatShape, 4096)
            self.fc1 = self.fullyConnected(self.flat, weights, biases, ID=1)

        with tf.name_scope('fc2'):
            weights, biases = self.getFullyConnectedVars(4096, 4096)
            self.fc2 = self.fullyConnected(self.fc1, weights, biases, ID=2)

        with tf.name_scope('fc3'):
            weights, biases = self.getFullyConnectedVars(4096, self.numberOfClasses)
            self.fc3 = self.fullyConnected(self.fc2, weights, biases, withRelu=False, ID=3)

        with tf.name_scope('softmax'):
            self.softmax = tf.nn.softmax(self.fc3)

        return self.fc3, self.softmax

    def loadWeights(self, weightsPath, sess):
        weights = np.load(weightsPath)
        keys = sorted(weights.keys())

        for paramIndex, key in enumerate(keys):
            if (self.pooling is not Pooling.NONE) and (paramIndex == 26):
                print('[{}] Loaded weights for convolutional layers'.format(self.tag))
                return
            if (self.numberOfClasses != 1000) and (paramIndex == 30):
                print('[{}] Loaded weights up to layer {} (excluded)'.format(self.tag, key[:-2]))
                return

            sess.run(self.parameters[paramIndex].assign(weights[key]))
