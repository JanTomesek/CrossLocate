#!/usr/bin/python3
"""
	Network Builder

	Created by Jan Tomesek on 14.11.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import tensorflow as tf
from networks.VGG16 import VGG16
from networks.AlexNet import AlexNet

from networks.NeuralNetwork import Architecture, Pooling


from enum import Enum

class Branching(Enum):
	ONE_BRANCH = 1
	TWO_BRANCH = 2

class Metric(Enum):
	EUCLIDEAN = 1
	COSINE = 2


class NetworkBuilder():
	"""
		Builds convolutional neural networks and adds loss functions and optimizers.
	"""

	def __init__(self, inputDimensions, batchSize, numberOfNegatives, metric, optimizerName):
		self.tag = 'NetworkBuilder'

		self.inputDimensions = inputDimensions
		self.batchSize = batchSize
		self.numberOfNegatives = numberOfNegatives
		self.metric = metric
		self.optimizerName = optimizerName

		self.tripletSize = 1 + 1 + self.numberOfNegatives

	def build(self, branching, architecture, preFeatNorm, pooling, postFeatNorm, margin):
		self.margin = margin

		print('[{}] Building {} variant of {} network with {} pooling'.format(self.tag, branching.name, architecture.name, pooling.name))
		print('')

		graph = tf.Graph()			# a TensorFlow computation, represented as a dataflow graph
		with graph.as_default():	# overrides the current default graph for the lifetime of the context
			model = self.__buildNetwork(1, architecture, preFeatNorm, pooling, postFeatNorm)
			print('')

			if (branching is Branching.TWO_BRANCH):
				model2 = self.__buildNetwork(2, architecture, preFeatNorm, pooling, postFeatNorm)
				print('')

			if (branching is Branching.ONE_BRANCH):
				with tf.name_scope('triplet_loss'):
					trainLoss = self.__tripletLoss(model.output)
			elif (branching is Branching.TWO_BRANCH):
				trainLoss = self.__tripletLossTwoBranch(model.output, model2.output)
				model = (model, model2)

			learningRate = tf.placeholder(tf.float32, shape=())

			if (self.optimizerName == 'Adam'):
				optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)

			optimizer = optimizer.minimize(trainLoss)

			initializer = tf.global_variables_initializer()		# an Op that initializes global variables in the graph

		return graph, model, trainLoss, learningRate, optimizer, initializer

	def __buildNetwork(self, ID, architecture, preFeatNorm, pooling, postFeatNorm):
		input = tf.placeholder(tf.float32, (None, *self.inputDimensions, 3))

		if (architecture is Architecture.VGG16):
			network = VGG16(ID, input, pooling, preFeatNorm=preFeatNorm, postFeatNorm=postFeatNorm)

		elif (architecture in [Architecture.ALEXNET_SIMPLE, Architecture.ALEXNET_SPLIT]):
			network = AlexNet(ID, architecture, input, pooling, preFeatNorm=preFeatNorm, postFeatNorm=postFeatNorm)

		return network

	def __tripletLossTwoBranch(self, firstBranchOutput, secondBranchOutput):
		print('[{}] ----- Triplet loss (two branch) -----'.format(self.tag))
		print('[{}] Inputs: {} {} '.format(self.tag, firstBranchOutput.shape, secondBranchOutput.shape))

		concat = tf.concat((firstBranchOutput, secondBranchOutput), axis=0)
		print('[{}] Concat: {}'.format(self.tag, concat.shape))
		print('')

		with tf.name_scope('triplet_loss'):
			tripletLoss = self.__tripletLoss(concat)

		return tripletLoss

	def __tripletLoss(self, output):
		print('[{}] ----- Triplet loss -----'.format(self.tag))
		print('')

		print('[{}] Input'.format(self.tag))
		print('[{}] {}'.format(self.tag, output.shape))
		print('')

		queries = output[0*self.batchSize:1*self.batchSize]
		positives = output[1*self.batchSize:2*self.batchSize]
		negatives = output[2*self.batchSize:self.tripletSize*self.batchSize]

		print('[{}] Slices'.format(self.tag))
		print('[{}] {}'.format(self.tag, queries.shape))
		print('[{}] {}'.format(self.tag, positives.shape))
		print('[{}] {}'.format(self.tag, negatives.shape))
		print('')

		queries = tf.reshape(queries, [self.batchSize, 1, -1])
		positives = tf.reshape(positives, [self.batchSize, 1, -1])
		negatives = tf.reshape(negatives, [self.batchSize, self.numberOfNegatives, -1])

		print('[{}] Individual triplets (add (second) dimension)'.format(self.tag))
		print('[{}] {}'.format(self.tag, queries.shape))
		print('[{}] {}'.format(self.tag, positives.shape))
		print('[{}] {}'.format(self.tag, negatives.shape))
		print('')

		if (self.metric is Metric.EUCLIDEAN):
			positiveDists = tf.reduce_sum(tf.square(queries - positives), axis=2)
			negativeDists = tf.reduce_sum(tf.square(queries - negatives), axis=2)
		elif (self.metric is Metric.COSINE):
			positiveDists = 1.0 - tf.reduce_sum(tf.multiply(queries, positives), axis=2)
			negativeDists = 1.0 - tf.reduce_sum(tf.multiply(queries, negatives), axis=2)

		print('[{}] Distances {} (query - positive, query - negatives)'.format(self.tag, self.metric.name))
		print('[{}] {}'.format(self.tag, positiveDists.shape))
		print('[{}] {}'.format(self.tag, negativeDists.shape))
		print('')

		loss = tf.maximum(positiveDists + self.margin - negativeDists, 0.0)

		print('[{}] Loss values (max-margin)'.format(self.tag))
		print('[{}] {}'.format(self.tag, loss.shape))
		print('')

		totalLoss = tf.reduce_sum(loss)

		print('[{}] Total loss'.format(self.tag))
		print('[{}] {}'.format(self.tag, totalLoss.shape))

		return totalLoss
