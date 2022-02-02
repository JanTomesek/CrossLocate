#!/usr/bin/python3
"""
	Descriptor Extractor

	Created by Jan Tomesek on 24.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import os
import numpy as np
from math import ceil, log10
import time

class DescriptorExtractor():
	"""
		Loads and extracts database and query descriptors from dataset images.

		Assumptions:
		Dataset images are in RGB format.
	"""

	def __init__(self, dataset, tensorAssembler, descriptorsPath, inputDimensions=None):
		self.tag = 'DescriptorExtractor ({})'.format(dataset.getSet())

		self.dataset = dataset
		self.tensorAssembler = tensorAssembler
		self.descriptorsPath = descriptorsPath

		self.inputDimensions = inputDimensions

		self.dbDescriptorsFn = dataset.getSet() + 'DbDescriptors'
		self.qDescriptorsFn = dataset.getSet() + 'QDescriptors'

		self.extractBatchSize = 10

	def loadDatasetDescriptors(self):
		dbDescriptors = None
		qDescriptors = None

		print('[{}] Loading existing database descriptors ...'.format(self.tag))
		dbDescriptorsPath = os.path.join(self.descriptorsPath, self.dbDescriptorsFn+'.npy')
		if (os.path.exists(dbDescriptorsPath)):
			dbDescriptors = np.load(dbDescriptorsPath)
			print('[{}] Database descriptors successfully loaded'.format(self.tag))
		else:
			print('[{}] Could not load database descriptors'.format(self.tag))

		print('')

		print('[{}] Loading existing query descriptors ...'.format(self.tag))
		qDescriptorsPath = os.path.join(self.descriptorsPath, self.qDescriptorsFn+'.npy')
		if (os.path.exists(qDescriptorsPath)):
			qDescriptors = np.load(qDescriptorsPath)
			print('[{}] Query descriptors successfully loaded'.format(self.tag))
		else:
			print('[{}] Could not load query descriptors'.format(self.tag))

		return dbDescriptors, qDescriptors

	def extractDatasetDescriptors(self, session, dbModel, qModel):
		if (self.inputDimensions is None):
			raise Exception('Input dimensions required!')

		startExtractTime = time.time()

		print('[{}] Extracting database descriptors ...'.format(self.tag))
		dbDescriptors = self.__extractDescriptors('db', session, dbModel, self.dataset.getDatabaseImageFns())
		np.save(os.path.join(self.descriptorsPath, self.dbDescriptorsFn), dbDescriptors)

		print('')

		print('[{}] Extracting query descriptors ...'.format(self.tag))
		qDescriptors = self.__extractDescriptors('q', session, qModel, self.dataset.getQueryImageFns())
		np.save(os.path.join(self.descriptorsPath, self.qDescriptorsFn), qDescriptors)

		print('')

		endExtractTime = time.time()
		self.__printExtractionTime(endExtractTime - startExtractTime)

		return dbDescriptors, qDescriptors

	def __extractDescriptors(self, type, session, model, imageFns):
		numberOfDescriptors = len(imageFns)
		descriptors = np.zeros(shape=(numberOfDescriptors, model.output.shape[1]), dtype=np.float32)

		numberOfExtractBatches = ceil(numberOfDescriptors / self.extractBatchSize)		# ceil to assure processing of all samples
		orderOfNumberOfDescriptors = int(log10(numberOfDescriptors))
		sampleProgressInterval = min(max(10**orderOfNumberOfDescriptors, 100), 10000)
		batchProgressInterval = ceil(sampleProgressInterval / self.extractBatchSize)

		for batch in range(numberOfExtractBatches):
			if (batch % batchProgressInterval == 0):
				print('[{}] {} / {}'.format(self.tag, batch*self.extractBatchSize, numberOfDescriptors))

			startBatchIndex = batch*self.extractBatchSize
			endBatchIndex = (batch+1)*self.extractBatchSize

			batchImageFns = imageFns[startBatchIndex : endBatchIndex]					# correctly takes only remaining samples in the final batch

			batchDescriptors = self.__getDescriptors(type, session, model, batchImageFns)

			descriptors[startBatchIndex : endBatchIndex] = batchDescriptors

		print('[{}] Total: {}'.	format(self.tag, startBatchIndex+len(batchDescriptors)))

		return descriptors

	def __getDescriptors(self, type, session, model, imageFns):
		if (type == 'db'):
			images = [self.tensorAssembler.getDatabaseTensor(imageFn) for imageFn in imageFns]
		elif (type == 'q'):
			images = [self.tensorAssembler.getQueryTensor(imageFn) for imageFn in imageFns]

		descriptors = session.run(model.output, feed_dict={model.input: images})

		return descriptors

	def __printExtractionTime(self, extractionTime):
		hours = round(extractionTime // 3600)
		remaining = extractionTime - (hours * 3600)
		minutes = round(remaining // 60)
		print('[{}] Extraction took {}h {}min'.format(self.tag, hours, minutes))
