#!/usr/bin/python3
"""
	AlexNet architecture

	Created by Jan Tomesek on 18.2.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "tomesek.j@gmail.com"
__copyright__ = "Copyright 2019, Jan Tomesek"


import tensorflow as tf
import numpy as np

from networks.NeuralNetwork import Architecture, Pooling, FeatureNormalization

from networks.NeuralNetwork import NeuralNetwork

class AlexNet(NeuralNetwork):
    """
        Assembles two variants of AlexNet architecture (simple and split) with optional fully connected layers.
        Loads pretrained weights as network parameters (only for split variant).
    """

    def __init__(self, ID, architecture, input, pooling,
                 lrNorm=True,
                 preFeatNorm=FeatureNormalization.NONE, postFeatNorm=FeatureNormalization.NONE,
                 trainableFromLayer=None, numberOfClasses=1000, descriptorsLayer=None):
        self.tag = 'AlexNet ({})'.format(ID)

        self.ID = ID
        self.architecture = architecture
        self.input = input
        self.preFeatNorm = preFeatNorm
        self.pooling = pooling
        self.postFeatNorm = postFeatNorm

        self.lrNorm = lrNorm

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
        lrnDepthRadius = 2
        lrnBias = 2.0
        lrnAlpha = 1e-04
        lrnBeta = 0.75

        if (self.trainableFromLayer is None):
            trainable = True
        else:
            trainable = False

        with tf.name_scope('conv1'):
            kernels, biases = self.getConvolutionVars((11, 11, 3), 96, trainable=trainable)
            self.conv1 = self.convolution2D(self.input, kernels, biases, stride=4, ID=1)

        if (self.lrNorm):
            with tf.name_scope('lr_norm1'):
                self.lrNorm1 = self.localResponseNormalization(self.conv1, depthRadius=lrnDepthRadius, bias=lrnBias, alpha=lrnAlpha, beta=lrnBeta, ID=1)
        else:
            self.lrNorm1 = self.conv1

        with tf.name_scope('maxpool1'):
            self.maxPool1 = self.maxPooling2D(self.lrNorm1, k=3, s=2, padding='VALID', ID=1)

        with tf.name_scope('conv2'):
            if (self.architecture is Architecture.ALEXNET_SIMPLE):
                kernels, biases = self.getConvolutionVars((5, 5, 96), 256, trainable=trainable)
                self.conv2 = self.convolution2D(self.maxPool1, kernels, biases, ID=2)
            elif (self.architecture is Architecture.ALEXNET_SPLIT):
                kernels, biases = self.getConvolutionVars((5, 5, 48), 256, trainable=trainable)
                self.conv2 = self.splitConvolution2D(self.maxPool1, kernels, biases, ID=2)

        if (self.lrNorm):
            with tf.name_scope('lr_norm2'):
                self.lrNorm2 = self.localResponseNormalization(self.conv2, depthRadius=lrnDepthRadius, bias=lrnBias, alpha=lrnAlpha, beta=lrnBeta, ID=2)
        else:
            self.lrNorm2 = self.conv2

        with tf.name_scope('maxpool2'):
            self.maxPool2 = self.maxPooling2D(self.lrNorm2, k=3, s=2, padding='VALID', ID=2)

        with tf.name_scope('conv3'):
            kernels, biases = self.getConvolutionVars((3, 3, 256), 384, trainable=trainable)
            self.conv3 = self.convolution2D(self.maxPool2, kernels, biases, ID=3)

        with tf.name_scope('conv4'):
            if (self.architecture is Architecture.ALEXNET_SIMPLE):
                kernels, biases = self.getConvolutionVars((3, 3, 384), 384, trainable=trainable)
                self.conv4 = self.convolution2D(self.conv3, kernels, biases, ID=4)
            elif (self.architecture is Architecture.ALEXNET_SPLIT):
                kernels, biases = self.getConvolutionVars((3, 3, 192), 384, trainable=trainable)
                self.conv4 = self.splitConvolution2D(self.conv3, kernels, biases, ID=4)

        if (self.trainableFromLayer == 'conv5'):
            trainable = True

        with tf.name_scope('conv5'):
            if (self.architecture is Architecture.ALEXNET_SIMPLE):
                kernels, biases = self.getConvolutionVars((3, 3, 384), 256, trainable=trainable)
                self.conv5 = self.convolution2D(self.conv4, kernels, biases, withRelu=(self.pooling is Pooling.NONE), ID=5)
            elif (self.architecture is Architecture.ALEXNET_SPLIT):
                kernels, biases = self.getConvolutionVars((3, 3, 192), 256, trainable=trainable)
                self.conv5 = self.splitConvolution2D(self.conv4, kernels, biases, withRelu=(self.pooling is Pooling.NONE), ID=5)

        if (self.pooling is not Pooling.NONE):
            return self.conv5

        with tf.name_scope('maxpool3'):
            self.maxPool3 = self.maxPooling2D(self.conv5, k=3, s=2, padding='VALID', ID=3)

        return self.maxPool3

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
        weights = np.load(weightsPath, encoding='latin1').item()
        weightLayers = sorted(weights.keys())

        paramIndex = 0
        # For every layer's weights
        for layer in weightLayers:
            # For both weights and biases
            for i in range(len(weights[layer])):                                # weights[layer] = [weights, biases]
                if (self.pooling is not Pooling.NONE) and (paramIndex == 10):
                    print('[{}] Loaded weights for convolutional layers'.format(self.tag))
                    return
                if (self.numberOfClasses != 1000) and (paramIndex == 14):
                    print('[{}] Loaded weights up to layer {} (excluded)'.format(self.tag, layer))
                    return

                sess.run(self.parameters[paramIndex].assign(weights[layer][i]))
                paramIndex += 1
