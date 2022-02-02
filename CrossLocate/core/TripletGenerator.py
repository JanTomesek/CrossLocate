#!/usr/bin/python3
"""
	Triplet Generator

	Created by Jan Tomesek on 24.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import numpy as np
from sklearn.neighbors import NearestNeighbors

from core.NetworkBuilder import Branching


from enum import Enum

class MiningStrategy(Enum):
	NONE = 0
	SEMIHARD_AND_HARD_NEGATIVE = 1
	SEMIHARD_NEGATIVE = 2


class TripletGenerator():
	"""
		Generates triplet batches of query, positive and negative samples.
		Based on geographic and descriptor distances.
		Each sample is assembled based on specified tensor format.

		Assumptions:
		Descriptor distances (from NN) are already sorted (small to big).
	"""

	def __init__(self, dataset, tensorAssembler, dbAugmentator, qAugmentator, branching, batchSize, dimensions, channels, margin, numberOfNegatives,
				 negativeSampleAttempts, numberOfSampledNegatives, reusingNegatives, numberOfTripletReusages, miningStrategy,
				 shuffle):
		self.tag = 'TripletGenerator ({})'.format(dataset.getSet())

		self.dataset = dataset
		self.tensorAssembler = tensorAssembler
		self.dbAugmentator = dbAugmentator
		self.qAugmentator = qAugmentator

		self.databasePath = self.dataset.getDatabasePath()
		self.queriesPath = self.dataset.getQueriesPath()

		self.branching = branching

		self.batchSize = batchSize
		self.dimensions = dimensions
		self.channels = channels

		self.margin = margin

		self.numberOfNegatives = numberOfNegatives

		self.negativeSampleAttempts = negativeSampleAttempts
		self.numberOfSampledNegatives = numberOfSampledNegatives

		self.reusingNegatives = reusingNegatives
		self.numberOfTripletReusages = numberOfTripletReusages
		self.miningStrategy = miningStrategy

		self.shuffle = shuffle  # shuffle sample order (even within batches)

		self.numberOfTriplets = len(self.dataset)
		self.tripletIndex = 0
		self.queryIndexes = np.arange(self.numberOfTriplets)
		self.currentTripletReusage = 0

		if (self.shuffle):
			np.random.shuffle(self.queryIndexes)

		self.numberOfPositivesNotFound = 0
		self.numberOfNegativesNotFound = 0

		self.nnAlgorithm = 'ball_tree'
		self.nnMetric = 'euclidean'

		self.lastNegativeDbIDs = [[]] * len(self.dataset)

		#self.on_epoch_end()

	def onEpochEnd(self):
		self.tripletIndex = 0
		self.queryIndexes = np.arange(self.numberOfTriplets)
		self.currentTripletReusage = 0

		if (self.shuffle):
			np.random.shuffle(self.queryIndexes)

		self.numberOfPositivesNotFound = 0
		self.numberOfNegativesNotFound = 0

	# Returns the number of batches (per epoch)
	# Keras method (equivalent to steps_per_epoch)
	def __len__(self):
		return int(np.floor(len(self.dataset)*self.numberOfTripletReusages / float(self.batchSize)))		# floor and float not necessary (but explicit)

	def setDescriptors(self, dbDescriptors, qDescriptors):
		self.dbDescriptors = dbDescriptors
		self.qDescriptors = qDescriptors

	def __updateLastNegativeDbIDs(self, queryID, negativeDbIDs):
		if (len(negativeDbIDs) == self.numberOfNegatives):
			# Replace last negative IDs with new negative IDs
			self.lastNegativeDbIDs[queryID] = negativeDbIDs
		else:
			# From last negative IDs keep only those IDs that are not in new negative IDs (duplicate negative IDs from new negative IDs have priority)
			uniqueLastNegativeDbIDs = [negDbID for negDbID in self.lastNegativeDbIDs[queryID] if negDbID not in negativeDbIDs]
			# Prepend new negative IDs to last negative IDs
			self.lastNegativeDbIDs[queryID] = negativeDbIDs + uniqueLastNegativeDbIDs
			# Keep only (newest) numberOfNegatives negativeIDs
			self.lastNegativeDbIDs[queryID] = self.lastNegativeDbIDs[queryID][0:self.numberOfNegatives]

	def __getLastNegativeDbIDs(self, queryID):
		return self.lastNegativeDbIDs[queryID]

	def getBatch(self):
		inputBatch, targetBatch = self.__generateTripletBatch()
		return inputBatch, targetBatch

	def getFnsBatch(self):
		return self.tripletFnsBatch

	def getLocDistancesBatch(self):
		return self.tripletLocDistancesBatch

	def getYawDistancesBatch(self):
		return self.tripletYawDistancesBatch

	# TODO: create tensors possibly before WHILE to avoid allocation
	def __generateTripletBatch(self):
		queryBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)
		positiveBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)
		negativeBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)

		queryFnsBatch = []
		positiveFnsBatch = []
		negativeFnsBatch = []

		queryLocDistancesBatch = []
		positiveLocDistancesBatch = []
		negativeLocDistancesBatch = []

		queryYawDistancesBatch = []
		positiveYawDistancesBatch = []
		negativeYawDistancesBatch = []

		currentBatchSize = 0
		while (currentBatchSize < self.batchSize):

			if (self.tripletIndex == self.numberOfTriplets):
				print('')
				print('[{}] All {} possible triplets processed'.format(self.tag, self.numberOfTriplets))
				print('[{}] {} triplets skipped because of positive samples'.format(self.tag, self.numberOfPositivesNotFound))
				print('[{}] {} triplets skipped because of negative samples'.format(self.tag, self.numberOfNegativesNotFound))
				return None, None

			# Get query ID for current triplet
			queryID = self.queryIndexes[self.tripletIndex]

			if (self.currentTripletReusage == 0):
				# Get positive and negative database IDs for current triplet
				positiveDbID, negativeDbIDs = self.__getTripletDbIDs(queryID)

				# Go through triplets until successful
				if (positiveDbID is None) or (negativeDbIDs is None):
					print('[{}] Skipping triplet {} (ID {})'.format(self.tag, self.tripletIndex, queryID))
					self.tripletIndex += 1
					continue

				self.reusePositiveDbID = positiveDbID
				self.reuseNegativeDbIDs = negativeDbIDs
			else:
				positiveDbID = self.reusePositiveDbID
				negativeDbIDs = self.reuseNegativeDbIDs

			# Create tensors for current triplet
			queryTensor = np.ndarray(shape=(1, *self.dimensions, self.channels), dtype=np.float32)
			positiveTensor = np.ndarray(shape=(1, *self.dimensions, self.channels), dtype=np.float32)
			negativeTensor = np.ndarray(shape=(self.numberOfNegatives, *self.dimensions, self.channels), dtype=np.float32)

			# Fill query tensor
			queryFn = self.dataset.getQueryImageFn(queryID)
			if (self.currentTripletReusage == 0):
				queryTensor[0] = self.tensorAssembler.getQueryTensor(queryFn)
				self.reuseQueryTensor = queryTensor[0]
			else:
				queryTensor[0] = self.reuseQueryTensor
			if (self.qAugmentator):
				queryTensor[0] = self.qAugmentator.augmentImage(queryTensor[0])
				isFlipped = self.qAugmentator.wasFlipped()

			queryFnsBatch.append(queryFn)
			queryLocDistancesBatch.append(None)
			queryYawDistancesBatch.append(None)

			# Fill positive tensor
			positiveFn = self.dataset.getDatabaseImageFn(positiveDbID[0])
			if (self.currentTripletReusage == 0):
				positiveTensor[0] = self.tensorAssembler.getDatabaseTensor(positiveFn)
				self.reusePositiveTensor = positiveTensor[0]
			else:
				positiveTensor[0] = self.reusePositiveTensor
			if (self.dbAugmentator):
				positiveTensor[0] = self.dbAugmentator.augmentImage(positiveTensor[0], applyFlip=isFlipped, disableFlip=(not isFlipped))

			positiveFnsBatch.append(positiveFn)
			positiveLocDistancesBatch += self.dataset.computeLocationDistances(queryID, positiveDbID, squared=False).tolist()
			positiveYawDistancesBatch += self.dataset.computeYawDistances(queryID, positiveDbID, inDegrees=True)

			# Fill negative tensor
			for negIndex, negativeDbID in enumerate(negativeDbIDs):
				negativeFn = self.dataset.getDatabaseImageFn(negativeDbID)
				if (self.currentTripletReusage == 0):
					negativeTensor[negIndex] = self.tensorAssembler.getDatabaseTensor(negativeFn)

				negativeFnsBatch.append(negativeFn)
			negativeLocDistancesBatch += self.dataset.computeLocationDistances(queryID, negativeDbIDs, squared=False).tolist()
			negativeYawDistancesBatch += self.dataset.computeYawDistances(queryID, negativeDbIDs, inDegrees=True)

			if (self.currentTripletReusage == 0):
				self.reuseNegativeTensor = negativeTensor
			else:
				negativeTensor = self.reuseNegativeTensor

			# Add remaining negatives by duplicating existing ones
			if (len(negativeDbIDs) < self.numberOfNegatives):
				remainingStartIndex = len(negativeDbIDs)
				for sampleIndex, remainingIndex in enumerate(range(remainingStartIndex, self.numberOfNegatives)):
					moduloSampleIndex = sampleIndex % remainingStartIndex
					negativeTensor[remainingIndex] = negativeTensor[moduloSampleIndex]

					negativeFnsBatch.append('repeat')
					negativeLocDistancesBatch += [None]
					negativeYawDistancesBatch += [None]

			if (self.dbAugmentator):
				for negIndex in range(len(negativeTensor)):
					negativeTensor[negIndex] = self.dbAugmentator.augmentImage(negativeTensor[negIndex], applyFlip=isFlipped, disableFlip=(not isFlipped))

			# Add current tensors to (sub)batches
			queryBatch = np.concatenate((queryBatch, queryTensor))
			positiveBatch = np.concatenate((positiveBatch, positiveTensor))
			negativeBatch = np.concatenate((negativeBatch, negativeTensor))

			self.currentTripletReusage += 1

			if (self.currentTripletReusage == self.numberOfTripletReusages):
				self.currentTripletReusage = 0
				self.tripletIndex += 1
			currentBatchSize += 1

		# Assemble triplet batch
		if (self.branching is Branching.ONE_BRANCH):
			tripletBatch = np.concatenate((queryBatch, positiveBatch, negativeBatch))
		elif (self.branching is Branching.TWO_BRANCH):
			tripletBatch = (queryBatch, np.concatenate((positiveBatch, negativeBatch)))

		self.tripletFnsBatch = queryFnsBatch + positiveFnsBatch + negativeFnsBatch
		self.tripletLocDistancesBatch = queryLocDistancesBatch + positiveLocDistancesBatch + negativeLocDistancesBatch
		self.tripletYawDistancesBatch = queryYawDistancesBatch + positiveYawDistancesBatch + negativeYawDistancesBatch

		return tripletBatch, None

	# TODO: remove manual losses
	def __getTripletDbIDs(self, queryID):
		# Get positive database IDs (by location and yaw distance)
		potentialPositiveDbIDs = self.dataset.getPositiveDbIDs(queryID=queryID)

		if not potentialPositiveDbIDs:
			print('[{}] No potential positive database IDs (return)'.format(self.tag))
			self.numberOfPositivesNotFound += 1
			return None, None

		# Collect violating negative database IDs across multiple sample attempts
		totalViolatingNegativeDbIDs = []
		negativeSampleAttempt = 0
		while ((len(totalViolatingNegativeDbIDs) < self.numberOfNegatives) and (negativeSampleAttempt < self.negativeSampleAttempts)):

			# Get negative database IDs (by location distance)
			negativeDbIDs = self.dataset.sampleNegativeDbIDsByLocation(queryID=queryID, n=self.numberOfSampledNegatives)

			# On the last attempt enrich current negative database IDs by negative database IDs from last epoch
			if ((self.reusingNegatives) and ((negativeSampleAttempt+1) == self.negativeSampleAttempts)):
				lastNegativeDbIDs = self.__getLastNegativeDbIDs(queryID)
				negativeDbIDs += lastNegativeDbIDs				# could introduce duplicate IDs (removed later in addUnique - removes duplicates even in new IDs)

			# Get descriptors corresponding to IDs
			queryDesc = self.qDescriptors[ [queryID] ]
			potentialPositiveDbDescs = self.dbDescriptors[ potentialPositiveDbIDs ]
			negativeDbDescs = self.dbDescriptors[ negativeDbIDs ]

			# Keep only positive and negative database IDs with smallest descriptor distances
			positiveDbID, posDistance = self.__filterPotentialPositiveDbIDs(potentialPositiveDbIDs, queryDesc, potentialPositiveDbDescs)
			negativeDbIDs, negDistances = self.__filterNegativeDbIDs(negativeDbIDs, queryDesc, negativeDbDescs)

			# Square the Euclidean descriptor distances
			posDistance = np.square(posDistance)
			negDistances = np.square(negDistances)

			if (self.miningStrategy is MiningStrategy.SEMIHARD_AND_HARD_NEGATIVE):
				# Keep only negative database IDs with descriptor distances less than positive distance + margin
				violatingNegativeDbIDs = [negativeDbID for negativeDbID, negativeDist in zip(negativeDbIDs, negDistances)
								if (negativeDist < posDistance[0]+self.margin)]

				# --------------- TODO: remove ------------------
				violatingNegativeDistances = [negativeDist for negativeDist in negDistances
								if (negativeDist < posDistance[0]+self.margin)]
				# -----------------------------------------------
			elif (self.miningStrategy is MiningStrategy.SEMIHARD_NEGATIVE):
				violatingNegativeDbIDs = [negativeDbID for negativeDbID, negativeDist in zip(negativeDbIDs, negDistances)
										  if ((negativeDist < posDistance[0]+self.margin) and (negativeDist >= posDistance[0]))]

				violatingNegativeDistances = [negativeDist for negativeDist in negDistances
										  if ((negativeDist < posDistance[0]+self.margin) and (negativeDist >= posDistance[0]))]		# TODO: remove
			elif (self.miningStrategy is MiningStrategy.NONE):
				violatingNegativeDbIDs = negativeDbIDs

				violatingNegativeDistances = negDistances		# TODO: remove

			self.dataset._Dataset__addUnique(totalViolatingNegativeDbIDs, violatingNegativeDbIDs)
			negativeSampleAttempt += 1

		violatingNegativeDbIDs = totalViolatingNegativeDbIDs

		numberOfViolatingNegatives = len(violatingNegativeDbIDs)

		if (numberOfViolatingNegatives == 0):
			print('[{}] No violating negative database IDs (return)'.format(self.tag))
			self.numberOfNegativesNotFound += 1
			return None, None

		# --------------- TODO: remove ------------------
		loss = 0
		for negativeDist in negDistances:
		    value = posDistance[0] + self.margin - negativeDist
		    loss += max(value, 0)
		print('[{}] Manual loss ({} negatives):'.format(self.tag, len(negDistances)), loss)
		# -----------------------------------------------

		# Keep max numberOfNegatives violating negative database IDs
		if (numberOfViolatingNegatives > self.numberOfNegatives):
			violatingNegativeDbIDs = violatingNegativeDbIDs[0:self.numberOfNegatives]
			
			# TODO: not required
			violatingNegativeDistances = violatingNegativeDistances[0:self.numberOfNegatives]

		# --------------- TODO: remove ------------------
		loss = 0
		for negativeDist in violatingNegativeDistances:
		    value = posDistance[0] + self.margin - negativeDist
		    loss += max(value, 0)
		print('[{}] Manual loss ({} violating negatives):'.format(self.tag, len(violatingNegativeDistances)), loss)
		# -----------------------------------------------

		self.__updateLastNegativeDbIDs(queryID, violatingNegativeDbIDs)

		return positiveDbID, violatingNegativeDbIDs

	# TODO: unite both filter function
	def __filterPotentialPositiveDbIDs(self, potentialPositiveDbIDs, queryDesc, potentialPositiveDbDescs):
		potentialPositive1NN = NearestNeighbors(n_neighbors=1, algorithm=self.nnAlgorithm, metric=self.nnMetric)
		potentialPositive1NN.fit(potentialPositiveDbDescs)

		posDistance, posIndex = potentialPositive1NN.kneighbors(queryDesc)	# [[distance]], [[index]]

		potentialPositiveDbIDs = np.array(potentialPositiveDbIDs)
		positiveDbID = potentialPositiveDbIDs[ posIndex[0] ]

		return positiveDbID, posDistance[0]									# [ID], [distance]

	def __filterNegativeDbIDs(self, negativeDbIDs, queryDesc, negativeDbDescs):
		negativeKNN = NearestNeighbors(n_neighbors=min(self.numberOfNegatives*10,len(negativeDbIDs)), algorithm=self.nnAlgorithm, metric=self.nnMetric)
		negativeKNN.fit(negativeDbDescs)

		negDistances, negIndices = negativeKNN.kneighbors(queryDesc)		# [[distances]], [[indices]]

		negativeDbIDs = np.array(negativeDbIDs)
		negativeDbIDs = negativeDbIDs[ negIndices[0] ]

		return negativeDbIDs, negDistances[0]								# [IDs], [distances]
