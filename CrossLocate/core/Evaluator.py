#!/usr/bin/python3
"""
	Evaluator

	Created by Jan Tomesek on 24.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import radians, pi
import random

class Evaluator():
	"""
		Computes recalls and distance recalls based on query and database descriptors or by simulating chance.

		Assumptions:
		Descriptor distances (from NN) are already sorted (small to big).
	"""

	def __init__(self, dataset, recallNumbers, locationThresholds, locationThresholdStep, yawThreshold, successfulLocalizationThreshold, outputPath):
		self.tag = 'Evaluator ({})'.format(dataset.getSet())

		self.dataset = dataset

		self.recallNumbers = recallNumbers

		self.locationThresholds = locationThresholds
		self.locationSquaredThresholds = np.square(locationThresholds)
		self.locationThresholdStep = locationThresholdStep
		self.yawThreshold = radians(yawThreshold)

		self.successfulLocalizationThreshold = successfulLocalizationThreshold

		self.outputPath = outputPath

		self.nnAlgorithm = 'ball_tree'
		self.nnMetric = 'euclidean'

	def setDescriptors(self, dbDescriptors, qDescriptors):
		self.dbDescriptors = dbDescriptors
		self.qDescriptors = qDescriptors

	def computeRecallsAtNs(self, chanceSolution=False, withPrint=False):
		bestNN = NearestNeighbors(n_neighbors=self.recallNumbers[-1], algorithm=self.nnAlgorithm, metric=self.nnMetric)
		bestNN.fit(self.dbDescriptors)

		# Create array with 100 nearest neighbor database IDs for each query
		qNearestNeighbors = []
		for qDesc in self.qDescriptors:
			if (not chanceSolution):
				qDescriptor = qDesc.reshape(1, -1)
				distances, indices = bestNN.kneighbors(qDescriptor)		# [[distances]], [[indices]]

				qNearestNeighbors.append(indices[0])
			else:
				qNearestNeighbors.append([])

		# Initialize recall records (both storing recalls for all location thresholds and all recall numbers)
		positionRecalls = {locThr: [] for locThr in self.locationThresholds}						# e.g. { 20: [], 100: [], 1000: [] }
		positionAndOrientationRecalls = {locThr: [] for locThr in self.locationThresholds}			# e.g. { 20: [], 100: [], 1000: [] }

		# Distance recalls
		positionDistanceRecalls = {}
		positionAndOrientationDistanceRecalls = {}

		baseThresholds = list(range(self.locationThresholdStep, self.locationThresholds[-1] + 1, self.locationThresholdStep))	# [20, 40, ..., 980, 1000]
		distanceThresholds = [0] + baseThresholds						# [0, 20, ..., 980, 1000], corresponds to histogram values in individual bins
		binEdges = [0, 0.001] + baseThresholds							# [0, 0.001, 20, ..., 980, 1000], corresponds to histogram edges dividing individual bins

		positionDistanceRecalls['thresholds'] = distanceThresholds
		positionAndOrientationDistanceRecalls['thresholds'] = distanceThresholds

		if (not chanceSolution):
			perQueryIterations = 1
		else:
			perQueryIterations = 11

		# Query localizations
		querySuccesses = []
		queryFails = []
		shouldGatherQueryLocalizations = (not chanceSolution)
		shouldAnalyseFOVs = (not chanceSolution)

		# For each given number of database candidates
		for recallNumber in self.recallNumbers:

			# Initialize minimum location distance records for current recall number
			qMinLocDistancesByLoc = np.zeros(shape=(len(self.qDescriptors), perQueryIterations), dtype=np.float64)
			qMinLocDistancesByLocAndYaw = np.zeros(shape=(len(self.qDescriptors), perQueryIterations), dtype=np.float64)

			# For each query with its database nearest neighbors
			for qID, qNNs in enumerate(qNearestNeighbors):

				for perQueryIter in range(perQueryIterations):

					if (not chanceSolution):
						# Pick n nearest neighbor database IDs for query
						neighborDbIDs = qNNs[0:recallNumber]
					else:
						# Generate n (unique) random database IDs for query
						neighborDbIDs = random.sample(range(0, len(self.dbDescriptors)), recallNumber)

					# Compute location distances
					locationDistances = self.dataset.computeLocationDistances(qID, neighborDbIDs, squared=False)

					# Compute yaw distances
					yawDistances = self.dataset.computeYawDistances(qID, neighborDbIDs, inDegrees=False)

					# Find minimum location distance among retrieved candidates for current recall number
					minLocDistanceByLoc = self.__findMinimumLocationDistanceByLocation(locationDistances)
					minLocDistanceByLocAndYaw = self.__findMinimumLocationDistanceByLocationAndYaw(locationDistances, yawDistances, self.yawThreshold)

					qMinLocDistancesByLoc[qID][perQueryIter] = minLocDistanceByLoc
					qMinLocDistancesByLocAndYaw[qID][perQueryIter] = minLocDistanceByLocAndYaw

				if (shouldGatherQueryLocalizations):
					if (recallNumber == 1):
						# Store query ID and the closest 10 database candidate IDs
						if (qMinLocDistancesByLocAndYaw[qID][0] <= self.successfulLocalizationThreshold):
							# the closest candidate (out of [recallNumber] candidates) is within the successful localization threshold
							querySuccesses.append((qID, qNNs[0:10]))
						else:
							# the closest candidate (out of [recallNumber] candidates) is not within the successful localization threshold
							queryFails.append((qID, qNNs[0:10]))

			# Compute and store distance recalls for current recall number
			positionDistanceRecalls[recallNumber] = self.__computeDistanceRecalls(qMinLocDistancesByLoc, binEdges)
			positionAndOrientationDistanceRecalls[recallNumber] = self.__computeDistanceRecalls(qMinLocDistancesByLocAndYaw, binEdges)

			# Store (append) ratios of successful retrievals to total number of queries (for selected location thresholds) for current recall number
			for locationThreshold in self.locationThresholds:
				positionRecalls[locationThreshold].append(self.__getValueAtThreshold(locationThreshold, positionDistanceRecalls[recallNumber], positionDistanceRecalls['thresholds']))
				positionAndOrientationRecalls[locationThreshold].append(self.__getValueAtThreshold(locationThreshold, positionAndOrientationDistanceRecalls[recallNumber], positionAndOrientationDistanceRecalls['thresholds']))

			if (shouldAnalyseFOVs):
				if (recallNumber == 1):
					self.__analyzeFOVs(qMinLocDistancesByLocAndYaw)
					print('')

		if (withPrint):
			print('[{}] Position recalls:'.format(self.tag))
			self.__printRecalls(positionRecalls)
			print('')
			print('[{}] Position and Orientation recalls:'.format(self.tag))
			self.__printRecalls(positionAndOrientationRecalls)

		if (not chanceSolution):
			np.save(os.path.join(self.outputPath, '{}QNearestNeighbors'.format(self.dataset.getSet())), qNearestNeighbors)
			np.save(os.path.join(self.outputPath, '{}QuerySuccesses'.format(self.dataset.getSet())), querySuccesses)
			np.save(os.path.join(self.outputPath, '{}QueryFails'.format(self.dataset.getSet())), queryFails)

		# positionRecalls:					e.g. { 20: [@1, @10, @100], 100: [@1, @10, @100], 1000: [@1, @10, @100] }
		# positionAndOrientationRecalls:	e.g. { 20: [@1, @10, @100], 100: [@1, @10, @100], 1000: [@1, @10, @100] }
		return positionDistanceRecalls, positionAndOrientationDistanceRecalls, positionRecalls, positionAndOrientationRecalls, querySuccesses, queryFails

	def __findMinimumLocationDistanceByLocation(self, locationDistances):
		minLocationDistance = min(locationDistances)

		return minLocationDistance

	def __findMinimumLocationDistanceByLocationAndYaw(self, locationDistances, yawDistances, yawThreshold):
		# Keep only (location) distances with yaw distances less than yawThreshold
		closeLocationDistancesByYaw = [locDist for locDist, yawDist in zip(locationDistances, yawDistances) if (yawDist <= yawThreshold)]

		if (len(closeLocationDistancesByYaw) >= 1):
			minLocationDistance = min(closeLocationDistancesByYaw)
		else:
			minLocationDistance = 1e+300

		return minLocationDistance

	def __computeDistanceRecalls(self, distances, binEdges):
		medianDistances = np.median(distances, axis=1)

		histogram, edges = np.histogram(medianDistances, bins=binEdges)

		cumulativeHistogram = np.cumsum(histogram)

		distanceRecalls = cumulativeHistogram / len(distances)

		return distanceRecalls

	def __getValueAtThreshold(self, threshold, values, thresholds):
		index = thresholds.index(threshold)

		return values[index]

	def __printRecalls(self, recalls):
		for locationThreshold in recalls.keys():
			print('{} m:\t{}'.format(str(locationThreshold).rjust(4), recalls[locationThreshold]))

	def __analyzeFOVs(self, qMinLocDistances):
		qFOVs = self.dataset.getQueryFOVs(inDegrees=True)

		assert (len(qMinLocDistances) == len(qFOVs)), 'Number of distances and number of FOVs do not match!'

		FOVBinEdges = [0, 15, 30, 45, 60, 80, 360]

		for i in range(len(FOVBinEdges)-1):
			qIDsWithinFOVGroup = np.where((qFOVs > FOVBinEdges[i]) & (qFOVs <= FOVBinEdges[i+1]))[0]

			qMinLocDistancesWithinFOVGroup = qMinLocDistances[qIDsWithinFOVGroup]

			qSuccessIDsWithinFOVGroup = np.where(qMinLocDistancesWithinFOVGroup <= 10000)[0]
			qExactIDsWithinFOVGroup = np.where(qMinLocDistancesWithinFOVGroup <= 1000)[0]

			exactToSuccessfulRatio = (len(qExactIDsWithinFOVGroup) / len(qSuccessIDsWithinFOVGroup)) if (len(qSuccessIDsWithinFOVGroup)>0) else 0.0
			successToAllRatio = (len(qSuccessIDsWithinFOVGroup) / len(qIDsWithinFOVGroup)) if (len(qIDsWithinFOVGroup)>0) else 0.0

			print('[{}] FOV <{:3d};{:3d}>: Localized {:4d} ({:5.2f}% exactly) out of {:4d} ({:5.2f}%)'.format(self.tag,
																											  FOVBinEdges[i], FOVBinEdges[i+1],
																											  len(qSuccessIDsWithinFOVGroup), exactToSuccessfulRatio*100,
																											  len(qIDsWithinFOVGroup), successToAllRatio*100))
