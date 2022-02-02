#!/usr/bin/python3
"""
	Plotter

	Created by Jan Tomesek on 29.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import numpy as np


from enum import Enum

class RecallType(Enum):
	POS = 1
	POS_AND_ORIENT = 2


from core.MasterPlotter import MasterPlotter

class Plotter(MasterPlotter):
	"""
		Plots charts for losses and recalls.

		Assumptions:
		Recalls are fixed on 1, 10 and 100 candidates.
	"""

	def __init__(self, recallNumbers, distanceRecallsXTicks, outputPath):
		super().__init__(outputPath, tag='Plotter')

		self.recallNumbers = recallNumbers

		self.distanceRecallsXTicks = distanceRecallsXTicks

		self.numberOfRecallNumbers = len(self.recallNumbers)

		self.trainMarker = '8'
		self.valMarker = 'v'

		self.recallSpecification = {
			1: { 'realColor': 'red',
				 'realLabel': 'recall@1',
				 'chanceColor': 'dimgray',
				 'chanceLabel': 'chance recall@1',
				 'multiColors': ['#AA0000', '#DD0000', 'red'] },
			10: { 'realColor': 'blue',
				  'realLabel': 'recall@10',
				  'chanceColor': 'darkgray',
				  'chanceLabel': 'chance recall@10',
				  'multiColors': ['#0000AA', '#0000DD', 'blue'] },
			100: { 'realColor': 'green',
				   'realLabel': 'recall@100',
				   'chanceColor': 'lightgray',
				   'chanceLabel': 'chance recall@100',
				   'multiColors': ['green', '#00AA00', '#00DD00'] },
			1000: { 'realColor': 'gold',
					'realLabel': 'recall@1000',
					'chanceColor': 'whitesmoke',
					'chanceLabel': 'chance recall@1000',
					'multiColors': ['gold', '#FFEE00', 'yellow'] }
		}


	# Plot single


	def plotTrainBatchLoss(self, history, epoch):
		self._plotSingle(None, history, 'Training Batch Loss (epoch {})'.format(epoch), 'Batch', 'Loss', 'g', 'trainBatchLoss{}'.format(epoch))

	def plotValBatchLoss(self, history, epoch):
		self._plotSingle(None, history, 'Validation Batch Loss (epoch {})'.format(epoch), 'Batch', 'Loss', 'r', 'valBatchLoss{}'.format(epoch))

	def plotTrainEpochLoss(self, history):
		self._plotSingle(None, history, 'Training Loss', 'Epoch', 'Loss', 'g', 'trainEpochLoss')

	def plotValEpochLoss(self, history):
		self._plotSingle(None, history, 'Validation Loss', 'Epoch', 'Loss', 'r', 'valEpochLoss')


	# Plot multiple


	def plotEpochLossComparison(self, multiHistory):
		colors = ['green', 'red']
		labels = ['training Loss', 'validation Loss']
		markers = [self.trainMarker, self.valMarker]
		title = 'Loss comparison'
		xLabel = 'Epoch'
		yLabel = 'Loss'
		xStart = 1
		fileName = 'epochLossComparison'

		self._plotMultiple(None, multiHistory, colors, labels, markers, title, xLabel, yLabel, xStart, fileName)


	# Recalls


	def plotRecalls(self, recallType, trainMultiThresholdHistory, valMultiThresholdHistory, trainChanceRecalls, valChanceRecalls, plotTrainValIndividual=True, plotTrainValComparison=True):
		trainChanceMultiThresholdHistory = {}
		valChanceMultiThresholdHistory = {}

		# Convert chance recalls into chance recall histories by repeating values "in time"
		for trainLocationThreshold, valLocationThreshold in zip(trainChanceRecalls, valChanceRecalls):
			assert (trainLocationThreshold == valLocationThreshold), 'Not matching location thresholds for training and validation chance recalls!'
			numberOfEpochs = len(trainMultiThresholdHistory[trainLocationThreshold][0])
			trainChanceMultiThresholdHistory[trainLocationThreshold] = self.__repeatRecallsInTime(trainChanceRecalls[trainLocationThreshold], numberOfEpochs)
			valChanceMultiThresholdHistory[valLocationThreshold] = self.__repeatRecallsInTime(valChanceRecalls[valLocationThreshold], numberOfEpochs)

		# Plot recalls for all (selected) thresholds jointly
		self.__plotRecallsAtAllThresholds(recallType, trainMultiThresholdHistory, valMultiThresholdHistory)

		# Plot recalls for each (selected) threshold individually
		for trainLocationThreshold, valLocationThreshold in zip(trainMultiThresholdHistory.keys(), valMultiThresholdHistory.keys()):
			assert (trainLocationThreshold == valLocationThreshold), 'Not matching location thresholds for training and validation recalls!'
			self.__plotRecallsAtThreshold(recallType, trainLocationThreshold, trainMultiThresholdHistory[trainLocationThreshold], valMultiThresholdHistory[valLocationThreshold], trainChanceMultiThresholdHistory[trainLocationThreshold], valChanceMultiThresholdHistory[valLocationThreshold], plotTrainValIndividual, plotTrainValComparison)

	def __repeatRecallsInTime(self, recallsAtThreshold, repeats):
		return np.repeat(np.array(recallsAtThreshold).reshape(-1, 1), repeats, axis=1)

	def __plotRecallsAtAllThresholds(self, recallType, trainMultiThresholdHistory, valMultiThresholdHistory):
		colors, labels = self.__prepareColorsAndLabelsForRecallsAtAllThresholds(trainMultiThresholdHistory.keys())
		markers = []
		if (recallType is RecallType.POS):
			title = 'Recalls (position)'
		elif (recallType is RecallType.POS_AND_ORIENT):
			title = 'Recalls (position and orientation)'
		xLabel = 'Epoch'
		yLabel = 'Recall'
		xStart = 0
		if (recallType is RecallType.POS):
			fileName = 'PositionRecalls'
		elif (recallType is RecallType.POS_AND_ORIENT):
			fileName = 'PositionAndOrientationRecalls'

		for trainLocationThreshold, valLocationThreshold in zip(trainMultiThresholdHistory.keys(), valMultiThresholdHistory.keys()):
			assert (trainLocationThreshold == valLocationThreshold), 'Not matching location thresholds for training and validation recalls!'


		# Plot training/validation recalls individually


		# Concatenate training recall histories across thresholds
		trainConcatThresholdHistory = self.__concatenateRecallsAcrossThresholds(trainMultiThresholdHistory)

		# Plot training recall histories (for all (selected) thresholds)
		markers = [self.trainMarker] * self.numberOfRecallNumbers * len(trainMultiThresholdHistory.keys())
		self._plotMultiple(None, trainConcatThresholdHistory, colors, labels, markers, 'Training '+title, xLabel, yLabel, xStart, 'train'+fileName, yNorm=True)

		# Concatenate validation recall histories across thresholds
		valConcatThresholdHistory = self.__concatenateRecallsAcrossThresholds(valMultiThresholdHistory)

		# Plot validation recall histories (for all (selected) thresholds)
		markers = [self.valMarker] * self.numberOfRecallNumbers * len(valMultiThresholdHistory.keys())
		self._plotMultiple(None, valConcatThresholdHistory, colors, labels, markers, 'Validation '+title, xLabel, yLabel, xStart, 'val'+fileName, yNorm=True)

	def __concatenateRecallsAcrossThresholds(self, multiThresholdHistory):
		concatThresholdHistory = None
		for locationThreshold in multiThresholdHistory.keys():
			if (concatThresholdHistory is None):
				concatThresholdHistory = multiThresholdHistory[locationThreshold]
			else:
				concatThresholdHistory = np.concatenate((concatThresholdHistory, multiThresholdHistory[locationThreshold]))

		return concatThresholdHistory

	def __plotRecallsAtThreshold(self, recallType, locationThreshold, trainMultiRecallsHistory, valMultiRecallsHistory, trainChanceMultiRecallsHistory, valChanceMultiRecallsHistory, plotTrainValIndividual=True, plotTrainValComparison=True):
		realColors, realLabels, chanceColors, chanceLabels = self.__prepareColorsAndLabelsForRecalls()
		markers = []
		if (recallType is RecallType.POS):
			title = 'Recalls (position) at {} m'.format(locationThreshold)
		elif (recallType is RecallType.POS_AND_ORIENT):
			title = 'Recalls (position and orientation) at {} m'.format(locationThreshold)
		xLabel = 'Epoch'
		yLabel = 'Recall'
		xStart = 0
		if (recallType is RecallType.POS):
			fileName = 'PositionRecallsAt{}m'.format(locationThreshold)
		elif (recallType is RecallType.POS_AND_ORIENT):
			fileName = 'PositionAndOrientationRecallsAt{}m'.format(locationThreshold)

		if (plotTrainValIndividual):
			colors = realColors + chanceColors
			labels = realLabels + chanceLabels

			trainAllMultiRecallsHistory = np.concatenate((trainMultiRecallsHistory, trainChanceMultiRecallsHistory))
			valAllMultiRecallsHistory = np.concatenate((valMultiRecallsHistory, valChanceMultiRecallsHistory))

			markers = [self.trainMarker]*self.numberOfRecallNumbers + [self.trainMarker]*self.numberOfRecallNumbers
			self._plotMultiple(None, trainAllMultiRecallsHistory, colors, labels, markers, 'Training '+title, xLabel, yLabel, xStart, 'train'+fileName, yNorm=True)

			markers = [self.valMarker]*self.numberOfRecallNumbers + [self.valMarker]*self.numberOfRecallNumbers
			self._plotMultiple(None, valAllMultiRecallsHistory, colors, labels, markers, 'Validation '+title, xLabel, yLabel, xStart, 'val'+fileName, yNorm=True)

		if (plotTrainValComparison):
			colors = realColors + realColors
			labels = [label+' (train)' for label in realLabels] + [label+' (val)' for label in realLabels]
			markers = [self.trainMarker]*self.numberOfRecallNumbers + [self.valMarker]*self.numberOfRecallNumbers
			self._plotMultiple(None, np.concatenate((trainMultiRecallsHistory, valMultiRecallsHistory)), colors, labels, markers, title, xLabel, yLabel, xStart, 'comparison'+fileName, yNorm=True)


	# Distance recalls


	def plotDistanceRecalls(self, recallType, trainDistanceRecalls, valDistanceRecalls, trainChanceDistanceRecalls, valChanceDistanceRecalls, plotTrainValIndividual=True, plotTrainValComparison=True):
		realColors, realLabels, chanceColors, chanceLabels = self.__prepareColorsAndLabelsForRecalls()
		markers = []
		if (recallType is RecallType.POS):
			title = 'Distance Recalls (position)'
		elif (recallType is RecallType.POS_AND_ORIENT):
			title = 'Distance Recalls (position and orientation)'
		xLabel = 'Distance [m]'
		yLabel = 'Recall'
		xStart = None
		if (recallType is RecallType.POS):
			fileName = 'PositionDistanceRecalls'
		elif (recallType is RecallType.POS_AND_ORIENT):
			fileName = 'PositionAndOrientationDistanceRecalls'

		trainConcatDistanceRecalls = self.__concatenateDistanceRecallsAcrossRecallNumbers(trainDistanceRecalls)
		valConcatDistanceRecalls = self.__concatenateDistanceRecallsAcrossRecallNumbers(valDistanceRecalls)

		trainChanceConcatDistanceRecalls = self.__concatenateDistanceRecallsAcrossRecallNumbers(trainChanceDistanceRecalls)
		valChanceConcatDistanceRecalls = self.__concatenateDistanceRecallsAcrossRecallNumbers(valChanceDistanceRecalls)

		if (plotTrainValIndividual):
			colors = realColors + chanceColors
			labels = realLabels + chanceLabels

			trainAllConcatDistanceRecalls = np.concatenate((trainConcatDistanceRecalls, trainChanceConcatDistanceRecalls))
			valAllConcatDistanceRecalls = np.concatenate((valConcatDistanceRecalls, valChanceConcatDistanceRecalls))

			markers = [self.trainMarker]*self.numberOfRecallNumbers + [self.trainMarker]*self.numberOfRecallNumbers
			self._plotMultiple(trainDistanceRecalls['thresholds'], trainAllConcatDistanceRecalls, colors, labels, markers, 'Training '+title, xLabel, yLabel, xStart, 'train'+fileName, xTicks=self.distanceRecallsXTicks, yNorm=True)

			markers = [self.valMarker]*self.numberOfRecallNumbers + [self.valMarker]*self.numberOfRecallNumbers
			self._plotMultiple(valDistanceRecalls['thresholds'], valAllConcatDistanceRecalls, colors, labels, markers, 'Validation '+title, xLabel, yLabel, xStart, 'val'+fileName, xTicks=self.distanceRecallsXTicks, yNorm=True)

		if (plotTrainValComparison):
			colors = realColors + realColors
			labels = [label+' (train)' for label in realLabels] + [label+' (val)' for label in realLabels]
			markers = [self.trainMarker]*self.numberOfRecallNumbers + [self.valMarker]*self.numberOfRecallNumbers
			self._plotMultiple(trainDistanceRecalls['thresholds'], np.concatenate((trainConcatDistanceRecalls, valConcatDistanceRecalls)), colors, labels, markers, title, xLabel, yLabel, xStart, 'comparison'+fileName, xTicks=self.distanceRecallsXTicks, yNorm=True)

	def __concatenateDistanceRecallsAcrossRecallNumbers(self, distanceRecalls):
		concatDistanceRecalls = []
		for key in distanceRecalls.keys():
			if (key == 'thresholds'):
				continue
			concatDistanceRecalls.append(distanceRecalls[key])

		concatDistanceRecalls = np.stack(concatDistanceRecalls)

		return concatDistanceRecalls

	def __prepareColorsAndLabelsForRecalls(self):
		realColors, realLabels, chanceColors, chanceLabels = [], [], [], []
		for recallNumber in self.recallNumbers:
			realColors.append(self.recallSpecification[recallNumber]['realColor'])
			realLabels.append(self.recallSpecification[recallNumber]['realLabel'])
			chanceColors.append(self.recallSpecification[recallNumber]['chanceColor'])
			chanceLabels.append(self.recallSpecification[recallNumber]['chanceLabel'])

		return realColors, realLabels, chanceColors, chanceLabels

	def __prepareColorsAndLabelsForRecallsAtAllThresholds(self, locationThresholds):
		_, realLabels, _, _ = self.__prepareColorsAndLabelsForRecalls()

		colors, labels = [], []
		for i, locThr in enumerate(locationThresholds):
			for recallNumber in self.recallNumbers:
				colors.append(self.recallSpecification[recallNumber]['multiColors'][i])
			labels += [label + ' at {} m'.format(locThr) for label in realLabels]

		return colors, labels
