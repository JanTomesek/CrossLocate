#!/usr/bin/python3
"""
	Historian

	Created by Jan Tomesek on 14.11.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import os
import numpy as np

from core.MasterHistorian import MasterHistorian

class Historian(MasterHistorian):
	"""
		Stores, writes and loads history for losses and recalls.
	"""

	def __init__(self, recallNumbers, locationThresholds, outputPath):
		super().__init__(outputPath, tag='Historian')

		self.recallNumbers = recallNumbers
		self.locationThresholds = locationThresholds

		self.totalTrainBatchLossHistory = []
		self.totalValBatchLossHistory = []

		self.currentTrainBatchLossHistory = []
		self.currentValBatchLossHistory = []

		self.trainEpochLossHistory = []
		self.valEpochLossHistory = []

		# Initialize recall histories (all storing recalls for all (selected) location thresholds, all recall numbers and all epochs)
		self.trainPositionRecallsHistory = {locThr: [] for locThr in self.locationThresholds}					# { 20: [], 100: [], 1000: [] }
		self.valPositionRecallsHistory = {locThr:[] for locThr in self.locationThresholds}						# { 20: [], 100: [], 1000: [] }

		self.trainPositionAndOrientationRecallsHistory = {locThr: [] for locThr in self.locationThresholds}		# { 20: [], 100: [], 1000: [] }
		self.valPositionAndOrientationRecallsHistory = {locThr: [] for locThr in self.locationThresholds}		# { 20: [], 100: [], 1000: [] }

		# Chance recall records
		self.trainChancePositionRecalls = {}
		self.valChancePositionRecalls = {}

		self.trainChancePositionAndOrientationRecalls = {}
		self.valChancePositionAndOrientationRecalls = {}

		# Distance recalls
		self.trainPositionDistanceRecalls = {}
		self.valPositionDistanceRecalls = {}

		self.trainPositionAndOrientationDistanceRecalls = {}
		self.valPositionAndOrientationDistanceRecalls = {}

		# Chance distance recalls
		self.trainChancePositionDistanceRecalls = {}
		self.valChancePositionDistanceRecalls = {}

		self.trainChancePositionAndOrientationDistanceRecalls = {}
		self.valChancePositionAndOrientationDistanceRecalls = {}

	def onEpochStart(self, epoch):
		super().onEpochStart(epoch)

	def onEpochEnd(self, epoch):
		self.totalTrainBatchLossHistory.append(self.currentTrainBatchLossHistory)
		self.totalValBatchLossHistory.append(self.currentValBatchLossHistory)

		self.currentTrainBatchLossHistory = []
		self.currentValBatchLossHistory = []

		super().onEpochEnd(epoch)


	# Batch losses


	def addTrainBatchLoss(self, trainBatchLoss):
		self.currentTrainBatchLossHistory.append(trainBatchLoss)

	def addValBatchLoss(self, valBatchLoss):
		self.currentValBatchLossHistory.append(valBatchLoss)

	def getTrainBatchLossHistory(self):
		return self.currentTrainBatchLossHistory

	def getValBatchLossHistory(self):
		return self.currentValBatchLossHistory


	# Epoch losses


	def addTrainEpochLoss(self, trainEpochLoss):
		self.trainEpochLossHistory.append(trainEpochLoss)

	def addValEpochLoss(self, valEpochLoss):
		self.valEpochLossHistory.append(valEpochLoss)

	def getTrainEpochLossHistory(self):
		return self.trainEpochLossHistory

	def getValEpochLossHistory(self):
		return self.valEpochLossHistory


	# Position recalls


	def addTrainPositionRecalls(self, trainPositionRecalls):
		# Store (append) recall records for all (selected) location thresholds
		for locationThreshold in self.locationThresholds:
			self.trainPositionRecallsHistory[locationThreshold].append(trainPositionRecalls[locationThreshold])

	def addValPositionRecalls(self, valPositionRecalls):
		# Store (append) recall records for all (selected) location thresholds
		for locationThreshold in self.locationThresholds:
			self.valPositionRecallsHistory[locationThreshold].append(valPositionRecalls[locationThreshold])

	def getTrainPositionRecallsHistory(self, forPlotting=False):
		if (not forPlotting):
			return self.trainPositionRecallsHistory
		else:
			# Transpose recalls history for each (selected) location threshold individually
			plottingTrainPositionRecallsHistory = {}
			for locationThreshold in self.trainPositionRecallsHistory:
				plottingTrainPositionRecallsHistory[locationThreshold] = np.transpose(self.trainPositionRecallsHistory[locationThreshold])
			return plottingTrainPositionRecallsHistory

	def getValPositionRecallsHistory(self, forPlotting=False):
		if (not forPlotting):
			return self.valPositionRecallsHistory
		else:
			# Transpose recalls history for each (selected) location threshold individually
			plottingValPositionRecallsHistory = {}
			for locationThreshold in self.valPositionRecallsHistory:
				plottingValPositionRecallsHistory[locationThreshold] = np.transpose(self.valPositionRecallsHistory[locationThreshold])
			return plottingValPositionRecallsHistory


	# Position and Orientation recalls


	def addTrainPositionAndOrientationRecalls(self, trainPositionAndOrientationRecalls):
		# Store (append) recall records for all (selected) location thresholds
		for locationThreshold in self.locationThresholds:
			self.trainPositionAndOrientationRecallsHistory[locationThreshold].append(trainPositionAndOrientationRecalls[locationThreshold])

	def addValPositionAndOrientationRecalls(self, valPositionAndOrientationRecalls):
		# Store (append) recall records for all (selected) location thresholds
		for locationThreshold in self.locationThresholds:
			self.valPositionAndOrientationRecallsHistory[locationThreshold].append(valPositionAndOrientationRecalls[locationThreshold])

	def getTrainPositionAndOrientationRecallsHistory(self, forPlotting=False):
		if (not forPlotting):
			return self.trainPositionAndOrientationRecallsHistory
		else:
			# Transpose recalls history for each (selected) location threshold individually
			plottingTrainPositionAndOrientationRecallsHistory = {}
			for locationThreshold in self.trainPositionAndOrientationRecallsHistory:
				plottingTrainPositionAndOrientationRecallsHistory[locationThreshold] = np.transpose(self.trainPositionAndOrientationRecallsHistory[locationThreshold])
			return plottingTrainPositionAndOrientationRecallsHistory

	def getValPositionAndOrientationRecallsHistory(self, forPlotting=False):
		if (not forPlotting):
			return self.valPositionAndOrientationRecallsHistory
		else:
			# Transpose recalls history for each (selected) location threshold individually
			plottingValPositionAndOrientationRecallsHistory = {}
			for locationThreshold in self.valPositionAndOrientationRecallsHistory:
				plottingValPositionAndOrientationRecallsHistory[locationThreshold] = np.transpose(self.valPositionAndOrientationRecallsHistory[locationThreshold])
			return plottingValPositionAndOrientationRecallsHistory


	# Chance position recalls


	def storeTrainChancePositionRecalls(self, trainChancePositionRecalls):
		self.trainChancePositionRecalls = trainChancePositionRecalls

	def storeValChancePositionRecalls(self, valChancePositionRecalls):
		self.valChancePositionRecalls = valChancePositionRecalls

	def getTrainChancePositionRecalls(self):
		return self.trainChancePositionRecalls

	def getValChancePositionRecalls(self):
		return self.valChancePositionRecalls


	# Chance position and orientation recalls


	def storeTrainChancePositionAndOrientationRecalls(self, trainChancePositionAndOrientationRecalls):
		self.trainChancePositionAndOrientationRecalls = trainChancePositionAndOrientationRecalls

	def storeValChancePositionAndOrientationRecalls(self, valChancePositionAndOrientationRecalls):
		self.valChancePositionAndOrientationRecalls = valChancePositionAndOrientationRecalls

	def getTrainChancePositionAndOrientationRecalls(self):
		return self.trainChancePositionAndOrientationRecalls

	def getValChancePositionAndOrientationRecalls(self):
		return self.valChancePositionAndOrientationRecalls


	# Position distance recalls


	def storeTrainPositionDistanceRecalls(self, trainPositionDistanceRecalls):
		self.trainPositionDistanceRecalls = trainPositionDistanceRecalls

	def storeValPositionDistanceRecalls(self, valPositionDistanceRecalls):
		self.valPositionDistanceRecalls = valPositionDistanceRecalls

	def getTrainPositionDistanceRecalls(self):
		return self.trainPositionDistanceRecalls

	def getValPositionDistanceRecalls(self):
		return self.valPositionDistanceRecalls


	# Position and Orientation distance recalls


	def storeTrainPositionAndOrientationDistanceRecalls(self, trainPositionAndOrientationDistanceRecalls):
		self.trainPositionAndOrientationDistanceRecalls = trainPositionAndOrientationDistanceRecalls

	def storeValPositionAndOrientationDistanceRecalls(self, valPositionAndOrientationDistanceRecalls):
		self.valPositionAndOrientationDistanceRecalls = valPositionAndOrientationDistanceRecalls

	def getTrainPositionAndOrientationDistanceRecalls(self):
		return self.trainPositionAndOrientationDistanceRecalls

	def getValPositionAndOrientationDistanceRecalls(self):
		return self.valPositionAndOrientationDistanceRecalls


	# Chance position distance recalls


	def storeTrainChancePositionDistanceRecalls(self, trainChancePositionDistanceRecalls):
		self.trainChancePositionDistanceRecalls = trainChancePositionDistanceRecalls

	def storeValChancePositionDistanceRecalls(self, valChancePositionDistanceRecalls):
		self.valChancePositionDistanceRecalls = valChancePositionDistanceRecalls

	def getTrainChancePositionDistanceRecalls(self):
		return self.trainChancePositionDistanceRecalls

	def getValChancePositionDistanceRecalls(self):
		return self.valChancePositionDistanceRecalls


	# Chance position and orientation distance recalls


	def storeTrainChancePositionAndOrientationDistanceRecalls(self, trainChancePositionAndOrientationDistanceRecalls):
		self.trainChancePositionAndOrientationDistanceRecalls = trainChancePositionAndOrientationDistanceRecalls

	def storeValChancePositionAndOrientationDistanceRecalls(self, valChancePositionAndOrientationDistanceRecalls):
		self.valChancePositionAndOrientationDistanceRecalls = valChancePositionAndOrientationDistanceRecalls

	def getTrainChancePositionAndOrientationDistanceRecalls(self):
		return self.trainChancePositionAndOrientationDistanceRecalls

	def getValChancePositionAndOrientationDistanceRecalls(self):
		return self.valChancePositionAndOrientationDistanceRecalls


	# Summary


	def showSummary(self):
		super().showSummary()
		print('')


		trainPositionAndOrientationRecallsHistory = self.getTrainPositionAndOrientationRecallsHistory(forPlotting=True)
		valPositionAndOrientationRecallsHistory = self.getValPositionAndOrientationRecallsHistory(forPlotting=True)


		# Best recalls (position and orientation)
		print('[{}] ----- Best Recalls (Position and Orientation) -----'.format(self.tag))
		print('')

		for trainLocationThreshold, valLocationThreshold in zip(trainPositionAndOrientationRecallsHistory.keys(), valPositionAndOrientationRecallsHistory.keys()):
			assert (trainLocationThreshold == valLocationThreshold), 'Not matching location thresholds for training and validation recalls!'
			self.__showBestRecallsSummaryAtThreshold(trainPositionAndOrientationRecallsHistory[trainLocationThreshold], valPositionAndOrientationRecallsHistory[valLocationThreshold], self.recallNumbers, trainLocationThreshold)
			print('')


		# Final recalls (position and orientation)
		print('[{}] ----- Final Recalls (Position and Orientation) -----'.format(self.tag))
		print('')

		for trainLocationThreshold, valLocationThreshold in zip(trainPositionAndOrientationRecallsHistory.keys(), valPositionAndOrientationRecallsHistory.keys()):
			assert (trainLocationThreshold == valLocationThreshold), 'Not matching location thresholds for training and validation recalls!'
			trainFinalRecalls, valFinalRecalls = self.__showFinalRecallsSummaryAtThreshold(trainPositionAndOrientationRecallsHistory[trainLocationThreshold], valPositionAndOrientationRecallsHistory[valLocationThreshold], self.recallNumbers, trainLocationThreshold)
			print('')

			self.__showRawRecalls(trainFinalRecalls, valFinalRecalls)
			print('')


		# Best by validation recalls (position and orientation)
		print('[{}] ----- Best by validation Recalls (Position And Orientation) -----'.format(self.tag))
		print('')

		for trainLocationThreshold, valLocationThreshold in zip(trainPositionAndOrientationRecallsHistory.keys(), valPositionAndOrientationRecallsHistory.keys()):
			assert (trainLocationThreshold == valLocationThreshold), 'Not matching location thresholds for training and validation recalls!'
			trainBestByValRecalls, valBestByValRecalls = self.__showBestByValRecallsSummaryAtThreshold(trainPositionAndOrientationRecallsHistory[trainLocationThreshold], valPositionAndOrientationRecallsHistory[valLocationThreshold], self.recallNumbers, trainLocationThreshold)
			print('')

			self.__showRawRecalls(trainBestByValRecalls, valBestByValRecalls)
			print('')

	def __showBestRecallsSummaryAtThreshold(self, trainPositionAndOrientationRecallsHistory, valPositionAndOrientationRecallsHistory, recallNumbers, locationThreshold):
		print('[{}] Best Recalls at {} m'.format(self.tag, locationThreshold))

		for trainPosAndOrRecallsAtN, valPosAndOrRecallsAtN, recallNumber in zip(trainPositionAndOrientationRecallsHistory, valPositionAndOrientationRecallsHistory, recallNumbers):
			maxTrainPosAndOrRecallIndex = np.argmax(trainPosAndOrRecallsAtN)
			maxTrainPosAndOrRecallAtN = trainPosAndOrRecallsAtN[maxTrainPosAndOrRecallIndex]

			maxValPosAndOrRecallIndex = np.argmax(valPosAndOrRecallsAtN)
			maxValPosAndOrRecallAtN = valPosAndOrRecallsAtN[maxValPosAndOrRecallIndex]

			print('[{}] Train: {:.2f} @{} (epoch {}) | Val: {:.2f} @{} (epoch {})'
				  .format(self.tag,
						  maxTrainPosAndOrRecallAtN *100, str(recallNumber).rjust(3), maxTrainPosAndOrRecallIndex,
						  maxValPosAndOrRecallAtN *100, str(recallNumber).rjust(3), maxValPosAndOrRecallIndex))			# recalls are computed before training (no need to increment indexes)

	def __showFinalRecallsSummaryAtThreshold(self, trainPositionAndOrientationRecallsHistory, valPositionAndOrientationRecallsHistory, recallNumbers, locationThreshold):
		print('[{}] Final Recalls at {} m'.format(self.tag, locationThreshold))
		
		trainFinalRecalls = []
		valFinalRecalls = []

		for trainPosAndOrRecallsAtN, valPosAndOrRecallsAtN, recallNumber in zip(trainPositionAndOrientationRecallsHistory, valPositionAndOrientationRecallsHistory, recallNumbers):
			finalTrainPosAndOrRecallAtN = trainPosAndOrRecallsAtN[-1]
			finalValPosAndOrRecallAtN = valPosAndOrRecallsAtN[-1]

			print('[{}] Train: {:.2f} @{} (epoch {}) | Val: {:.2f} @{} (epoch {})'
				  .format(self.tag,
						  finalTrainPosAndOrRecallAtN *100, str(recallNumber).rjust(3), len(trainPosAndOrRecallsAtN)-1,
						  finalValPosAndOrRecallAtN *100, str(recallNumber).rjust(3), len(valPosAndOrRecallsAtN)-1))

			trainFinalRecalls.append(finalTrainPosAndOrRecallAtN *100)
			valFinalRecalls.append(finalValPosAndOrRecallAtN *100)

		return trainFinalRecalls, valFinalRecalls

	def __showBestByValRecallsSummaryAtThreshold(self, trainRecallsHistory, valRecallsHistory, recallNumbers, locationThreshold):
		bestValRecallsEpoch = 0
		bestValRecallsTotal = 0

		# Search for epoch with best validation recalls total
		numberOfEpochs = len(trainRecallsHistory[0])
		for epoch in range(numberOfEpochs):
			# Calculate total of validation recalls over all recall numbers for current epoch
			epochValRecallsTotal = 0
			for valRecallsAtN in valRecallsHistory:
				epochValRecallsTotal += valRecallsAtN[epoch]

			# Keep best validation recalls total and epoch
			if (epochValRecallsTotal > bestValRecallsTotal):
				bestValRecallsTotal = epochValRecallsTotal
				bestValRecallsEpoch = epoch

		print('[{}] Best by validation Recalls at {} m'.format(self.tag, locationThreshold))

		trainBestByValRecalls = []
		valBestByValRecalls = []

		for trainRecallsAtN, valRecallsAtN, recallNumber in zip(trainRecallsHistory, valRecallsHistory, recallNumbers):
			bestByValTrainRecallAtN = trainRecallsAtN[bestValRecallsEpoch]
			bestByValValRecallAtN = valRecallsAtN[bestValRecallsEpoch]

			print('[{}] Train {:.2f} @{} (epoch {}) | Val: {:.2f} @{} (epoch {})'
				  .format(self.tag,
						  bestByValTrainRecallAtN *100, str(recallNumber).rjust(3), bestValRecallsEpoch,
						  bestByValValRecallAtN *100, str(recallNumber).rjust(3), bestValRecallsEpoch))

			trainBestByValRecalls.append(bestByValTrainRecallAtN *100)
			valBestByValRecalls.append(bestByValValRecallAtN *100)

		return trainBestByValRecalls, valBestByValRecalls

	def __showRawRecalls(self, trainRecalls, valRecalls):
		print(str(trainRecalls).lstrip('[').rstrip(']').replace(',', '').replace('.', ','))
		print(str(valRecalls).lstrip('[').rstrip(']').replace(',', '').replace('.', ','))


	# Load


	def load(self):
		super().load(overridden=True)

		self.totalTrainBatchLossHistory = np.load(os.path.join(self.outputPath, 'trainBatchLossHistory'+'.npy')).tolist()
		self.totalValBatchLossHistory = np.load(os.path.join(self.outputPath, 'valBatchLossHistory'+'.npy')).tolist()

		self.trainEpochLossHistory = np.load(os.path.join(self.outputPath, 'trainEpochLossHistory'+'.npy')).tolist()
		self.valEpochLossHistory = np.load(os.path.join(self.outputPath, 'valEpochLossHistory'+'.npy')).tolist()

		# Recall histories
		self.trainPositionRecallsHistory = np.load(os.path.join(self.outputPath, 'trainPositionRecallsHistory'+'.npy')).tolist()
		self.valPositionRecallsHistory = np.load(os.path.join(self.outputPath, 'valPositionRecallsHistory'+'.npy')).tolist()

		self.trainPositionAndOrientationRecallsHistory = np.load(os.path.join(self.outputPath, 'trainPositionAndOrientationRecallsHistory'+'.npy')).tolist()
		self.valPositionAndOrientationRecallsHistory = np.load(os.path.join(self.outputPath, 'valPositionAndOrientationRecallsHistory'+'.npy')).tolist()

		# Chance recall records
		self.trainChancePositionRecalls = np.load(os.path.join(self.outputPath, 'trainChancePositionRecalls'+'.npy')).tolist()
		self.valChancePositionRecalls = np.load(os.path.join(self.outputPath, 'valChancePositionRecalls'+'.npy')).tolist()

		self.trainChancePositionAndOrientationRecalls = np.load(os.path.join(self.outputPath, 'trainChancePositionAndOrientationRecalls'+'.npy')).tolist()
		self.valChancePositionAndOrientationRecalls = np.load(os.path.join(self.outputPath, 'valChancePositionAndOrientationRecalls'+'.npy')).tolist()

		# Distance recalls
		self.trainPositionDistanceRecalls = np.load(os.path.join(self.outputPath, 'trainPositionDistanceRecalls'+'.npy')).tolist()
		self.valPositionDistanceRecalls = np.load(os.path.join(self.outputPath, 'valPositionDistanceRecalls'+'.npy')).tolist()

		self.trainPositionAndOrientationDistanceRecalls = np.load(os.path.join(self.outputPath, 'trainPositionAndOrientationDistanceRecalls'+'.npy')).tolist()
		self.valPositionAndOrientationDistanceRecalls = np.load(os.path.join(self.outputPath, 'valPositionAndOrientationDistanceRecalls'+'.npy')).tolist()

		# Chance distance recalls
		self.trainChancePositionDistanceRecalls = np.load(os.path.join(self.outputPath, 'trainChancePositionDistanceRecalls'+'.npy')).tolist()
		self.valChancePositionDistanceRecalls = np.load(os.path.join(self.outputPath, 'valChancePositionDistanceRecalls'+'.npy')).tolist()

		self.trainChancePositionAndOrientationDistanceRecalls = np.load(os.path.join(self.outputPath, 'trainChancePositionAndOrientationDistanceRecalls'+'.npy')).tolist()
		self.valChancePositionAndOrientationDistanceRecalls = np.load(os.path.join(self.outputPath, 'valChancePositionAndOrientationDistanceRecalls'+'.npy')).tolist()

		print('[{}] History successfully loaded'.format(self.tag))


	# Write


	def write(self):
		super().write(overridden=True)

		np.save(os.path.join(self.outputPath, 'trainBatchLossHistory'), self.totalTrainBatchLossHistory)
		np.save(os.path.join(self.outputPath, 'valBatchLossHistory'), self.totalValBatchLossHistory)

		np.save(os.path.join(self.outputPath, 'trainEpochLossHistory'), self.trainEpochLossHistory)
		np.save(os.path.join(self.outputPath, 'valEpochLossHistory'), self.valEpochLossHistory)

		# Recall histories
		np.save(os.path.join(self.outputPath, 'trainPositionRecallsHistory'), self.trainPositionRecallsHistory)
		np.save(os.path.join(self.outputPath, 'valPositionRecallsHistory'), self.valPositionRecallsHistory)

		np.save(os.path.join(self.outputPath, 'trainPositionAndOrientationRecallsHistory'), self.trainPositionAndOrientationRecallsHistory)
		np.save(os.path.join(self.outputPath, 'valPositionAndOrientationRecallsHistory'), self.valPositionAndOrientationRecallsHistory)

		# Chance recall records
		np.save(os.path.join(self.outputPath, 'trainChancePositionRecalls'), self.trainChancePositionRecalls)
		np.save(os.path.join(self.outputPath, 'valChancePositionRecalls'), self.valChancePositionRecalls)

		np.save(os.path.join(self.outputPath, 'trainChancePositionAndOrientationRecalls'), self.trainChancePositionAndOrientationRecalls)
		np.save(os.path.join(self.outputPath, 'valChancePositionAndOrientationRecalls'), self.valChancePositionAndOrientationRecalls)

		# Distance recalls
		np.save(os.path.join(self.outputPath, 'trainPositionDistanceRecalls'), self.trainPositionDistanceRecalls)
		np.save(os.path.join(self.outputPath, 'valPositionDistanceRecalls'), self.valPositionDistanceRecalls)

		np.save(os.path.join(self.outputPath, 'trainPositionAndOrientationDistanceRecalls'), self.trainPositionAndOrientationDistanceRecalls)
		np.save(os.path.join(self.outputPath, 'valPositionAndOrientationDistanceRecalls'), self.valPositionAndOrientationDistanceRecalls)

		# Chance distance recalls
		np.save(os.path.join(self.outputPath, 'trainChancePositionDistanceRecalls'), self.trainChancePositionDistanceRecalls)
		np.save(os.path.join(self.outputPath, 'valChancePositionDistanceRecalls'), self.valChancePositionDistanceRecalls)

		np.save(os.path.join(self.outputPath, 'trainChancePositionAndOrientationDistanceRecalls'), self.trainChancePositionAndOrientationDistanceRecalls)
		np.save(os.path.join(self.outputPath, 'valChancePositionAndOrientationDistanceRecalls'), self.valChancePositionAndOrientationDistanceRecalls)

		print('[{}] History successfully written'.format(self.tag))
