#!/usr/bin/python3
"""
	CrossLocate: Cross-Modal Large-Scale Visual Geo-Localization in Natural Environments Using Rendered Modalities
	WACV 2022

	Created by Jan Tomesek on 24.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2022, Jan Tomesek"


# Properties
experimentName = 'CrossLocate'
tag = 'train'


# Parameters
from core.DatasetDescriptions import datasetRenderV1, datasetFCN8sV1, datasetDeeplabV1, datasetOriginalV1, datasetDepthV1, datasetSilhouettesV1, datasetAustriaQ1
from core.DatasetDescriptions import datasetSegments, datasetSilhouettes, datasetDepth, datasetOriginalToSegments, datasetOriginalToSilhouettes, datasetOriginalToDepth
from core.DatasetDescriptions import datasetUniformSegments2000, datasetUniformSegments500, datasetUniformOriginalToSegments500
from core.DatasetDescriptions import datasetSegmentsSwiss, datasetOriginalToSegmentsSwiss
from core.DatasetDescriptions import datasetOriginalToSilhouettesSwiss, datasetOriginalToDepthSwiss
from core.DatasetDescriptions import datasetOriginalToSegmentsSwissResolution
from core.DatasetDescriptions import datasetAlpsGP3KSegmentsCompact, datasetAlpsGP3KOriginalToSegmentsCompact, datasetAlpsPhotosToSegmentsCompact
from core.DatasetDescriptions import datasetAlpsPhotosToSegmentsResolutionCompact
from core.DatasetDescriptions import datasetAlpsPhotosToSilhouettesCompact
from core.DatasetDescriptions import datasetAlpsPhotosToDepthCompact, datasetAlpsPhotosToDepthResolutionCompact
from core.DatasetDescriptions import datasetAlpsCH1ToSegments, datasetAlpsCH1ToSegmentsResolution
from core.DatasetDescriptions import datasetAlpsCH1ToSilhouettes
from core.DatasetDescriptions import datasetAlpsCH1ToDepth, datasetAlpsCH1ToDepthResolution
from core.DatasetDescriptions import datasetAlpsCH2ToSegments, datasetAlpsCH2ToSegmentsResolution, datasetAlpsCH2ToSegmentsExt, datasetAlpsCH2ToSegmentsExtResolution
from core.DatasetDescriptions import datasetAlpsCH2ToSilhouettes, datasetAlpsCH2ToSilhouettesExt
from core.DatasetDescriptions import datasetAlpsCH2ToDepth, datasetAlpsCH2ToDepthResolution, datasetAlpsCH2ToDepthExt, datasetAlpsCH2ToDepthExtResolution
from core.NetworkBuilder import Branching, Metric
from networks.NeuralNetwork import Architecture, Pooling, FeatureNormalization
from core.TensorAssembler import TensorFormat
from core.TripletGenerator import MiningStrategy
from core.Plotter import RecallType

netParams = { 	'inputDimensions': (224, 224),
			 	'branching': Branching.ONE_BRANCH,
			 	'architecture': Architecture.VGG16,
			 	'preFeatNorm': FeatureNormalization.L2,
				'pooling': Pooling.MAC,
			 	'postFeatNorm': FeatureNormalization.L2,
			 	'pretrained': 'IMAGENET' }

trainParams = { 'startEpoch': 1,
				'endEpoch': 20,
				'batchSize': 4,
				'margin': 0.1,
				'queryExtractFeatsInterval': 250,
				'queryTensorFormat': TensorFormat.INPUT_RGB,
				'databaseTensorFormat': TensorFormat.INPUT_RGB,
				'metric': Metric.EUCLIDEAN }

learnParams = {	'optimizer': 'Adam',
				'learningRate': 0.00001,
			   	'learningRateDownscaleEpochs': [],
			   	'learningRateDownscaleFactors': [] }

mineParams = {  'positiveLocationThreshold': 20,
                'positiveYawThreshold': 15,
                'negativeLocationMinThreshold': 2000,
                'margin': 0.2,
                'numberOfNegatives': 10,
                'negativeSampleAttempts': 2,
                'numberOfSampledNegatives': 4000,
                'reusingNegatives': True,
                'trainMiningStrategy': MiningStrategy.SEMIHARD_AND_HARD_NEGATIVE,
				'numberOfTrainTripletReusages': 1 }

augmentParams = {   'augmentation': True }

evalParams = {	'recallNumbers' : [1, 10, 100],
			  	'locationThresholds': [20, 100, 1000],
			  	'locationThresholdStep': 20,
			  	'successfulLocalizationThreshold': 20 }

plotParams = { 'distanceRecallsXTicks': [0, 20, 100, 250, 500, 750, 1000] }

visualParams = { 'batchVisualEpochs': [trainParams['startEpoch'], trainParams['endEpoch']],
				 'successesVisualEpochs': [trainParams['endEpoch']],
				 'failsVisualEpochs': [trainParams['endEpoch']] }

techParams = { 'moveToSSD': False }

outputDirectories = {	'charts': 'charts',
						'descriptors': 'descriptors',
					 	'evaluations': 'evaluations',
						'history': 'history',
						'models': 'models',
					 	'visuals': 'visuals' }

datasetParams = datasetSegments


# Reproducibility
from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(123)
# still possible randomness from CuDNN


# Imports
import tensorflow as tf
import os
import numpy as np
from math import floor
from shutil import rmtree


# Output paths and directories
fullOutputPath = os.path.join('output', experimentName)
chartsPath = os.path.join(fullOutputPath, outputDirectories['charts'])
descriptorsPath = os.path.join(fullOutputPath, outputDirectories['descriptors'])
evaluationsPath = os.path.join(fullOutputPath, outputDirectories['evaluations'])
historyPath = os.path.join(fullOutputPath, outputDirectories['history'])
modelsPath = os.path.join(fullOutputPath, outputDirectories['models'])
visualsPath = os.path.join(fullOutputPath, outputDirectories['visuals'])

os.mkdir(chartsPath)
os.mkdir(descriptorsPath)
os.mkdir(evaluationsPath)
if (trainParams['startEpoch'] == 1):
	os.mkdir(historyPath)
	os.mkdir(modelsPath)
os.mkdir(visualsPath)


# GPU selection
from core.Environment import Environment
print('----- GPU -----')

gpuID = Environment.seizeGPU()
if (gpuID == -1):
    print('[{}] No GPU, exit'.format(tag))
    exit(1)
print('[{}] Seized GPU: {}'.format(tag, gpuID))

print('')


# Environment info
print('----- Environment -----')

Environment.showInfo()

print('')


# Configuration
print('----- Config -----')
# Do not allocate all of the memory

config = tf.ConfigProto()
config.gpu_options.allow_growth = True	# TODO: check

gpu_memory_usage = 1.0					# TODO: check
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_usage	# TODO: check

print('')


# Local experiment directory
if (techParams['moveToSSD']):
	userSSDPath = '/mnt/ssd/' + 'CrossLocate'
	try:
		os.mkdir(userSSDPath, mode=0o700)
	except FileExistsError:
		pass

	experimentSSDPath = os.path.join(userSSDPath, experimentName)
	os.mkdir(experimentSSDPath)


# Datasets
from core.Dataset import Dataset
print('----- Datasets -----')

datasetsPath = '../datasets'

fullDatasetPath = os.path.join(datasetsPath, datasetParams['name'])
databasePath = os.path.join(fullDatasetPath, datasetParams['database'])
queriesPath = os.path.join(fullDatasetPath, datasetParams['queries'])

# Move dataset to local experiment directory
if (techParams['moveToSSD']):
	print('Moving dataset to SSD ...')
	databasePath, queriesPath = Environment.moveDataset([databasePath, queriesPath], experimentSSDPath)
	print('')

trainDataset = Dataset(datasetParams['name'], datasetParams['meta'], 'train', databasePath, queriesPath, positiveLocationThreshold=mineParams['positiveLocationThreshold'], positiveYawThreshold=mineParams['positiveYawThreshold'], negativeLocationMinThreshold=mineParams['negativeLocationMinThreshold'])
trainDataset.loadFromMetaStructs()
trainDataset.precomputeLocationDistancesWGS()
trainDataset.findPositiveDbIDs()
trainDataset.findNonNegativeDbIDs()

print('')

valDataset = Dataset(datasetParams['name'], datasetParams['meta'], 'val', databasePath, queriesPath, positiveLocationThreshold=mineParams['positiveLocationThreshold'], positiveYawThreshold=mineParams['positiveYawThreshold'], negativeLocationMinThreshold=mineParams['negativeLocationMinThreshold'])
valDataset.loadFromMetaStructs()
valDataset.precomputeLocationDistancesWGS()
valDataset.findPositiveDbIDs()
valDataset.findNonNegativeDbIDs()

print('')


# Tensor assembler
from core.TensorAssembler import TensorAssembler
print('----- Tensor assembler -----')

tensorAssembler = TensorAssembler(datasetParams['meta'], databasePath, queriesPath, netParams['inputDimensions'], 3, trainParams['databaseTensorFormat'], trainParams['queryTensorFormat'])

fullDepthDatasetPath = os.path.join(datasetsPath, datasetDepth['name'])
fullSilhouetteDatasetPath = os.path.join(datasetsPath, datasetSilhouettes['name'])

if (trainParams['databaseTensorFormat'] is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
	depthDatabasePath = os.path.join(fullDepthDatasetPath, datasetDepth['database'])
	silhouetteDatabasePath = os.path.join(fullSilhouetteDatasetPath, datasetSilhouettes['database'])

	if (techParams['moveToSSD']):
		depthDatabasePath, silhouetteDatabasePath = Environment.moveDataset([depthDatabasePath, silhouetteDatabasePath], experimentSSDPath)

	tensorAssembler.setDepthDatabasePath(depthDatabasePath)
	tensorAssembler.setSilhouetteDatabasePath(silhouetteDatabasePath)

if (trainParams['queryTensorFormat'] is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
	depthQueriesPath = os.path.join(fullDepthDatasetPath, datasetDepth['queries'])
	silhouetteQueriesPath = os.path.join(fullSilhouetteDatasetPath, datasetSilhouettes['queries'])

	if (techParams['moveToSSD']):
		depthQueriesPath, silhouetteQueriesPath = Environment.moveDataset([depthQueriesPath, silhouetteQueriesPath], experimentSSDPath)

	tensorAssembler.setDepthQueriesPath(depthQueriesPath)
	tensorAssembler.setSilhouetteQueriesPath(silhouetteQueriesPath)

# Set path to original queries
if (trainParams['queryTensorFormat'] is TensorFormat.ORIG_GRAY_INPUT_GRAY_INPUT_GRAY):
	fullOriginalDatasetPath = os.path.join(datasetsPath, datasetOriginalV1['name'])
	originalQueriesPath = os.path.join(fullOriginalDatasetPath, datasetOriginalV1['queries'])

	tensorAssembler.setOriginalQueriesPath(originalQueriesPath)

print('')


# Network building
from core.NetworkBuilder import NetworkBuilder
print('----- Network -----')

networkBuilder = NetworkBuilder(netParams['inputDimensions'], trainParams['batchSize'], mineParams['numberOfNegatives'], trainParams['metric'], learnParams['optimizer'])

graph, model, trainLoss, learningRate, optimizer, initializer = networkBuilder.build(netParams['branching'], netParams['architecture'], netParams['preFeatNorm'], netParams['pooling'], netParams['postFeatNorm'], trainParams['margin'])

if (netParams['branching'] is Branching.ONE_BRANCH):
	dbModel = model
	qModel = model
elif (netParams['branching'] is Branching.TWO_BRANCH):
	dbModel = model[1]
	qModel = model[0]

print('')


# Descriptor extractors
from core.DescriptorExtractor import DescriptorExtractor
print('----- Descriptor extractors -----')

trainingDescriptorExtractor = DescriptorExtractor(trainDataset, tensorAssembler, descriptorsPath, netParams['inputDimensions'])
validationDescriptorExtractor = DescriptorExtractor(valDataset, tensorAssembler, descriptorsPath, netParams['inputDimensions'])

print('')


# Augmentator
from core.Augmentator import Augmentator
print('----- Augmentator -----')

dbAugmentator = Augmentator()
qAugmentator = Augmentator()

if (augmentParams['augmentation']):
	dbAugmentator.activateShiftAugmentation(maxHorizontalPercentage=0.05, maxVerticalPercentage=0.025)
	dbAugmentator.activateRotationAugmentation(maxAngle=5)
	dbAugmentator.activateFlipAugmentation()

	qAugmentator.activateShiftAugmentation(maxHorizontalPercentage=0.10, maxVerticalPercentage=0.050)
	qAugmentator.activateRotationAugmentation(maxAngle=5)
	qAugmentator.activateFlipAugmentation()
	qAugmentator.activateBrightnessAugmentation(maxMagnitude=40)

	qAugmentator.activateHueAugmentation(maxMagnitude=10)
	qAugmentator.activateSaturationAugmentation(maxMagnitude=50)
	qAugmentator.activateContrastAugmentation(maxPercentage=0.4)

	qAugmentator.activateBlurAugmentation(maxSigmaPercentage=0.004)
	qAugmentator.activateNoiseAugmentation(std=20)

print('')


# Triplet generators
from core.TripletGenerator import TripletGenerator
print('----- Triplet generators -----')

generatorParams = {	'branching': netParams['branching'],
				   	'dimensions': netParams['inputDimensions'],
		            'batchSize': trainParams['batchSize'],
		            'channels': 3,
		            'margin': mineParams['margin'],
				   	'numberOfNegatives': mineParams['numberOfNegatives'],
				   	'negativeSampleAttempts': mineParams['negativeSampleAttempts'],
				   	'numberOfSampledNegatives': mineParams['numberOfSampledNegatives'],
                    'reusingNegatives': mineParams['reusingNegatives']}

trainingGenerator = TripletGenerator(trainDataset, tensorAssembler, dbAugmentator, qAugmentator, **generatorParams, numberOfTripletReusages=mineParams['numberOfTrainTripletReusages'], miningStrategy=mineParams['trainMiningStrategy'], shuffle=True)
validationGenerator = TripletGenerator(valDataset, tensorAssembler, None, None, **generatorParams, numberOfTripletReusages=1, miningStrategy=MiningStrategy.NONE, shuffle=False)

print('')


# Objects
from core.Evaluator import Evaluator
from core.Historian import Historian
from core.Plotter import Plotter
from core.TripletVisualizer import TripletVisualizer

from core.TripletVisualizer import VisualMode

# Evaluators
trainEvaluator = Evaluator(trainDataset, evalParams['recallNumbers'], evalParams['locationThresholds'], evalParams['locationThresholdStep'], yawThreshold=30, successfulLocalizationThreshold=evalParams['successfulLocalizationThreshold'], outputPath=evaluationsPath)
valEvaluator = Evaluator(valDataset, evalParams['recallNumbers'], evalParams['locationThresholds'], evalParams['locationThresholdStep'], yawThreshold=30, successfulLocalizationThreshold=evalParams['successfulLocalizationThreshold'], outputPath=evaluationsPath)

# Historian
historian = Historian(evalParams['recallNumbers'], evalParams['locationThresholds'], historyPath)

# Plotter
plotter = Plotter(evalParams['recallNumbers'], plotParams['distanceRecallsXTicks'], chartsPath)

# Batch viewer
batchVisualizer = TripletVisualizer(VisualMode.BATCH, datasetParams['meta'], trainParams['batchSize'], mineParams['numberOfNegatives'], trainParams['databaseTensorFormat'], trainParams['queryTensorFormat'], visualsPath)

trainLocVisualizer = TripletVisualizer(VisualMode.LOC, datasetParams['meta'], trainParams['batchSize'], mineParams['numberOfNegatives'], trainParams['databaseTensorFormat'], trainParams['queryTensorFormat'], visualsPath)
trainLocVisualizer.setupForQueryLocalizationsVisualization(trainDataset, tensorAssembler, netParams['inputDimensions'], 3, evalParams['locationThresholds'])

valLocVisualizer = TripletVisualizer(VisualMode.LOC, datasetParams['meta'], trainParams['batchSize'], mineParams['numberOfNegatives'], trainParams['databaseTensorFormat'], trainParams['queryTensorFormat'], visualsPath)
valLocVisualizer.setupForQueryLocalizationsVisualization(valDataset, tensorAssembler, netParams['inputDimensions'], 3, evalParams['locationThresholds'])

print('')




# Preparation
print('----- Session -----')
with tf.Session(graph=graph, config=config) as sess:	# TODO: check
	print('')

	sess.run(initializer)								# TODO: check (but network seems to be pretrained)

	saver = tf.train.Saver(max_to_keep=2)

	if (trainParams['startEpoch'] == 1):
		if (netParams['pretrained'] == 'IMAGENET'):
			if (netParams['architecture'] is Architecture.ALEXNET_SIMPLE):
				raise Exception('ImageNet weights cannot be used for {} architecture!'.format(netParams['architecture'].name))
			elif (netParams['architecture'] is Architecture.ALEXNET_SPLIT):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					model.loadWeights('models/bvlc_alexnet.npy', sess)
					print('Loaded ImageNet weights for {} {}'.format(netParams['branching'].name, netParams['architecture'].name))
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					dbModel.loadWeights('models/bvlc_alexnet.npy', sess)
					qModel.loadWeights('models/bvlc_alexnet.npy', sess)
					print('Loaded ImageNet weights for {} {}'.format(netParams['branching'].name, netParams['architecture'].name))

			# Load weights
			if (netParams['architecture'] is Architecture.VGG16):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					model.loadWeights('models/vgg16_weights.npz', sess)
					print('Loaded ImageNet weights for {} {}'.format(netParams['branching'].name, netParams['architecture'].name))
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					dbModel.loadWeights('models/vgg16_weights.npz', sess)
					qModel.loadWeights('models/vgg16_weights.npz', sess)
					print('Loaded ImageNet weights for {} {}'.format(netParams['branching'].name, netParams['architecture'].name))

		elif (netParams['pretrained']):
			pretrainedModelsPath = os.path.join('output', netParams['pretrained'], outputDirectories['models'])
			saver.restore(sess, tf.train.latest_checkpoint(pretrainedModelsPath))
			print('Loaded latest model from', pretrainedModelsPath)

	else:
		# Load graph and latest model
		#modelFn = experimentName+'-model-'+str(trainParams['startEpoch']-1)
		#saver = tf.train.import_meta_graph(os.path.join(modelsPath, modelFn+'.meta'))
		saver.restore(sess, tf.train.latest_checkpoint(modelsPath))
		print('Restored latest model from', modelsPath)

		# Load history
		historian.load()

	fileWriter = tf.summary.FileWriter(os.path.join('tensorboard', experimentName), sess.graph)	# TODO: check

	print('')


	# Initial descriptor extraction
	print('----- Initial descriptor extraction -----')
	print('')

	trainDbDescriptors, trainQDescriptors = trainingDescriptorExtractor.loadDatasetDescriptors()
	if (trainDbDescriptors is None) or (trainQDescriptors is None):
		print('')
		print('Precomputing training descriptors ...')
		trainDbDescriptors, trainQDescriptors = trainingDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)

	print('')

	valDbDescriptors, valQDescriptors = validationDescriptorExtractor.loadDatasetDescriptors()
	if (valDbDescriptors is None) or (valQDescriptors is None):
		print('')
		print('Precomputing validation descriptors ...')
		valDbDescriptors, valQDescriptors = validationDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)

	trainingGenerator.setDescriptors(trainDbDescriptors, trainQDescriptors)
	validationGenerator.setDescriptors(valDbDescriptors, valQDescriptors)

	print('')


	# Initial evaluation
	print('----- Initial recalls -----')
	print('')

	trainEvaluator.setDescriptors(trainDbDescriptors, trainQDescriptors)
	valEvaluator.setDescriptors(valDbDescriptors, valQDescriptors)

	print('-- Chance recalls --')
	print('')
	trainChancePositionDistanceRecalls, trainChancePositionAndOrientationDistanceRecalls, trainChancePositionRecalls, trainChancePositionAndOrientationRecalls, _, _ = trainEvaluator.computeRecallsAtNs(chanceSolution=True, withPrint=True)
	print('')
	valChancePositionDistanceRecalls, valChancePositionAndOrientationDistanceRecalls, valChancePositionRecalls, valChancePositionAndOrientationRecalls, _, _ = valEvaluator.computeRecallsAtNs(chanceSolution=True, withPrint=True)
	print('')

	print('-- Real recalls --')
	print('')
	trainPositionDistanceRecalls, trainPositionAndOrientationDistanceRecalls, trainPositionRecalls, trainPositionAndOrientationRecalls, _, _ = trainEvaluator.computeRecallsAtNs(withPrint=True)
	print('')
	valPositionDistanceRecalls, valPositionAndOrientationDistanceRecalls, valPositionRecalls, valPositionAndOrientationRecalls, _, _ = valEvaluator.computeRecallsAtNs(withPrint=True)
	print('')

	# Store initial evaluation results only for new training
	if (trainParams['startEpoch'] == 1):
		# Recalls
		historian.addTrainPositionRecalls(trainPositionRecalls)
		historian.addValPositionRecalls(valPositionRecalls)
		historian.addTrainPositionAndOrientationRecalls(trainPositionAndOrientationRecalls)
		historian.addValPositionAndOrientationRecalls(valPositionAndOrientationRecalls)

		# Chance recalls
		historian.storeTrainChancePositionRecalls(trainChancePositionRecalls)
		historian.storeValChancePositionRecalls(valChancePositionRecalls)
		historian.storeTrainChancePositionAndOrientationRecalls(trainChancePositionAndOrientationRecalls)
		historian.storeValChancePositionAndOrientationRecalls(valChancePositionAndOrientationRecalls)

		# Distance recalls
		historian.storeTrainPositionDistanceRecalls(trainPositionDistanceRecalls)
		historian.storeValPositionDistanceRecalls(valPositionDistanceRecalls)
		historian.storeTrainPositionAndOrientationDistanceRecalls(trainPositionAndOrientationDistanceRecalls)
		historian.storeValPositionAndOrientationDistanceRecalls(valPositionAndOrientationDistanceRecalls)

		# Chance distance recalls
		historian.storeTrainChancePositionDistanceRecalls(trainChancePositionDistanceRecalls)
		historian.storeValChancePositionDistanceRecalls(valChancePositionDistanceRecalls)
		historian.storeTrainChancePositionAndOrientationDistanceRecalls(trainChancePositionAndOrientationDistanceRecalls)
		historian.storeValChancePositionAndOrientationDistanceRecalls(valChancePositionAndOrientationDistanceRecalls)

	plotter.plotRecalls(RecallType.POS, historian.getTrainPositionRecallsHistory(True), historian.getValPositionRecallsHistory(True), historian.getTrainChancePositionRecalls(), historian.getValChancePositionRecalls())
	plotter.plotRecalls(RecallType.POS_AND_ORIENT, historian.getTrainPositionAndOrientationRecallsHistory(True), historian.getValPositionAndOrientationRecallsHistory(True), historian.getTrainChancePositionAndOrientationRecalls(), historian.getValChancePositionAndOrientationRecalls())

	plotter.plotDistanceRecalls(RecallType.POS, historian.getTrainPositionDistanceRecalls(), historian.getValPositionDistanceRecalls(), historian.getTrainChancePositionDistanceRecalls(), historian.getValChancePositionDistanceRecalls())
	plotter.plotDistanceRecalls(RecallType.POS_AND_ORIENT, historian.getTrainPositionAndOrientationDistanceRecalls(), historian.getValPositionAndOrientationDistanceRecalls(), historian.getTrainChancePositionAndOrientationDistanceRecalls(), historian.getValChancePositionAndOrientationDistanceRecalls())




	# Model training
	print('')
	print('-------------------- Training model --------------------')

	print('Train on {} samples, validate on {} samples (queries)'.format(len(trainDataset), len(valDataset)))
	print('')

	numberOfTrainBatches = len(trainingGenerator)
	numberOfValBatches = len(validationGenerator)

	currentLearningRate = learnParams['learningRate']
	batchExtractFeatsInterval = floor(trainParams['queryExtractFeatsInterval']/trainParams['batchSize'])

	for epoch in range(trainParams['startEpoch'], trainParams['endEpoch']+1):

		print('')
		print('--------------------')
		print('Starting epoch {}'.format(epoch))
		print('--------------------')

		historian.onEpochStart(epoch)
		print('')

		# Learning rate adjustment
		if (epoch in learnParams['learningRateDownscaleEpochs']):
			print('Changing learning rate')
			index = learnParams['learningRateDownscaleEpochs'].index(epoch)
			downscaleFactor = learnParams['learningRateDownscaleFactors'][index]
			currentLearningRate = currentLearningRate / downscaleFactor
		print('Learning rate {} ({:.20f})'.format(currentLearningRate, currentLearningRate).rstrip('0'))
		print('')

		# Prepare directories for batch visualization
		if (epoch in visualParams['batchVisualEpochs']):
			trainBatchVisualDirName = 'train_batch_epoch{}'.format(epoch)
			valBatchVisualDirName = 'val_batch_epoch{}'.format(epoch)
			os.mkdir(os.path.join(visualsPath, trainBatchVisualDirName))
			os.mkdir(os.path.join(visualsPath, valBatchVisualDirName))




		# Training phase
		print('')
		print('----- Training phase -----')
		print('')

		trainSummary = tf.summary.scalar('train_batch_loss_epoch{}'.format(epoch), trainLoss)	# TODO: check

		trainLossAcc = 0

		for batch in range(1, numberOfTrainBatches+1):
			print('Processing training batch {} out of {}'.format(batch, numberOfTrainBatches))

			tripletBatch, _ = trainingGenerator.getBatch()					# TODO: batch index

			if (tripletBatch is not None):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					feedDict = {model.input: tripletBatch, learningRate: currentLearningRate}
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					qBatch, dbBatch = tripletBatch
					feedDict = {qModel.input: qBatch, dbModel.input: dbBatch, learningRate: currentLearningRate}

				_, trainBatchLoss, summary = sess.run([optimizer, trainLoss, trainSummary], feed_dict=feedDict)	# TODO: check
				print('Train (batch) loss:', trainBatchLoss)
				trainLossAcc += trainBatchLoss
				historian.addTrainBatchLoss(trainBatchLoss)
				fileWriter.add_summary(summary, batch)						# TODO: check, change counter
			else:
				print('')
				print('Premature end of training epoch')
				print('')
				break

			if (epoch in visualParams['batchVisualEpochs']):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					imageBatch = tripletBatch
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					imageBatch = np.concatenate(tripletBatch)
				tripletFnsBatch = trainingGenerator.getFnsBatch()
				tripletLocDistancesBatch = trainingGenerator.getLocDistancesBatch()
				tripletYawDistancesBatch = trainingGenerator.getYawDistancesBatch()
				batchVisualizer.visualizeBatch(imageBatch, tripletFnsBatch, tripletLocDistancesBatch, tripletYawDistancesBatch, trainBatchVisualDirName, batch)

			if (batch % batchExtractFeatsInterval == 0):
				print('')
				print('----- Intermediate descriptor extraction -----')
				print('')

				trainDbDescriptors, trainQDescriptors = None, None

				trainDbDescriptors, trainQDescriptors = trainingDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)

				trainingGenerator.setDescriptors(trainDbDescriptors, trainQDescriptors)

			print('')

		trainEpochLoss = trainLossAcc / batch		# TODO: check
		trainEpochLossSummary = tf.Summary(value=[ tf.Summary.Value(tag='train_epoch_loss', simple_value=trainEpochLoss) ])
		historian.addTrainEpochLoss(trainEpochLoss)
		fileWriter.add_summary(trainEpochLossSummary, epoch)

		trainingGenerator.onEpochEnd()


		# Save network
		saver.save(sess, os.path.join(modelsPath, experimentName+'-model'), global_step=epoch, write_meta_graph=True)
		print('Saved network for epoch', epoch)
		print('')




		# Epoch descriptor extraction
		print('----- Epoch descriptor extraction -----')
		print('')

		trainDbDescriptors, trainQDescriptors = None, None
		valDbDescriptors, valQDescriptors = None, None

		trainDbDescriptors, trainQDescriptors = trainingDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)
		valDbDescriptors, valQDescriptors = validationDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)

		trainingGenerator.setDescriptors(trainDbDescriptors, trainQDescriptors)
		validationGenerator.setDescriptors(valDbDescriptors, valQDescriptors)

		print('')




		# Validation phase
		print('')
		print('---- Validation phase ----')
		print('')

		valSummary = tf.summary.scalar('val_batch_loss_epoch{}'.format(epoch), trainLoss)

		valLossAcc = 0

		for batch in range(1, numberOfValBatches+1):
			print('Processing validation batch {} out of {}'.format(batch, numberOfValBatches))

			tripletBatch, _ = validationGenerator.getBatch()				# TODO: batch index

			if (tripletBatch is not None):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					feedDict = {model.input: tripletBatch}
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					qBatch, dbBatch = tripletBatch
					feedDict = {qModel.input: qBatch, dbModel.input: dbBatch}

				valBatchLoss, summary = sess.run([trainLoss, valSummary], feed_dict=feedDict)
				print('Val (batch) loss', valBatchLoss)
				valLossAcc += valBatchLoss
				historian.addValBatchLoss(valBatchLoss)
				fileWriter.add_summary(summary, batch)
			else:
				print('')
				print('Premature end of validation epoch')
				print('')
				break

			if (epoch in visualParams['batchVisualEpochs']):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					imageBatch = tripletBatch
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					imageBatch = np.concatenate(tripletBatch)
				tripletFnsBatch = validationGenerator.getFnsBatch()
				tripletLocDistancesBatch = validationGenerator.getLocDistancesBatch()
				tripletYawDistancesBatch = validationGenerator.getYawDistancesBatch()
				batchVisualizer.visualizeBatch(imageBatch, tripletFnsBatch, tripletLocDistancesBatch, tripletYawDistancesBatch, valBatchVisualDirName, batch)

			print('')

		valEpochLoss = valLossAcc / batch			# TODO: check
		valEpochLossSummary = tf.Summary(value=[ tf.Summary.Value(tag='val_epoch_loss', simple_value=valEpochLoss) ])
		historian.addValEpochLoss(valEpochLoss)
		fileWriter.add_summary(valEpochLossSummary, epoch)

		validationGenerator.onEpochEnd()




		# Epoch evaluation
		print('----- Epoch {} recalls -----'.format(epoch))
		print('')

		trainEvaluator.setDescriptors(trainDbDescriptors, trainQDescriptors)
		valEvaluator.setDescriptors(valDbDescriptors, valQDescriptors)

		trainPositionDistanceRecalls, trainPositionAndOrientationDistanceRecalls, trainPositionRecalls, trainPositionAndOrientationRecalls, trainQuerySuccesses, trainQueryFails = trainEvaluator.computeRecallsAtNs(withPrint=True)
		print('')
		valPositionDistanceRecalls, valPositionAndOrientationDistanceRecalls, valPositionRecalls, valPositionAndOrientationRecalls, valQuerySuccesses, valQueryFails = valEvaluator.computeRecallsAtNs(withPrint=True)
		print('')

		# Recalls
		historian.addTrainPositionRecalls(trainPositionRecalls)
		historian.addValPositionRecalls(valPositionRecalls)
		historian.addTrainPositionAndOrientationRecalls(trainPositionAndOrientationRecalls)
		historian.addValPositionAndOrientationRecalls(valPositionAndOrientationRecalls)

		# Distance recalls
		historian.storeTrainPositionDistanceRecalls(trainPositionDistanceRecalls)
		historian.storeValPositionDistanceRecalls(valPositionDistanceRecalls)
		historian.storeTrainPositionAndOrientationDistanceRecalls(trainPositionAndOrientationDistanceRecalls)
		historian.storeValPositionAndOrientationDistanceRecalls(valPositionAndOrientationDistanceRecalls)


		# Plots
		plotter.plotTrainBatchLoss(historian.getTrainBatchLossHistory(), epoch)
		plotter.plotValBatchLoss(historian.getValBatchLossHistory(), epoch)

		plotter.plotTrainEpochLoss(historian.getTrainEpochLossHistory())
		plotter.plotValEpochLoss(historian.getValEpochLossHistory())

		plotter.plotEpochLossComparison([historian.getTrainEpochLossHistory(), historian.getValEpochLossHistory()])

		plotter.plotRecalls(RecallType.POS, historian.getTrainPositionRecallsHistory(True), historian.getValPositionRecallsHistory(True), historian.getTrainChancePositionRecalls(), historian.getValChancePositionRecalls())
		plotter.plotRecalls(RecallType.POS_AND_ORIENT, historian.getTrainPositionAndOrientationRecallsHistory(True), historian.getValPositionAndOrientationRecallsHistory(True), historian.getTrainChancePositionAndOrientationRecalls(), historian.getValChancePositionAndOrientationRecalls())

		plotter.plotDistanceRecalls(RecallType.POS, historian.getTrainPositionDistanceRecalls(), historian.getValPositionDistanceRecalls(), historian.getTrainChancePositionDistanceRecalls(), historian.getValChancePositionDistanceRecalls())
		plotter.plotDistanceRecalls(RecallType.POS_AND_ORIENT,historian.getTrainPositionAndOrientationDistanceRecalls(), historian.getValPositionAndOrientationDistanceRecalls(), historian.getTrainChancePositionAndOrientationDistanceRecalls(), historian.getValChancePositionAndOrientationDistanceRecalls())


		if (epoch in visualParams['successesVisualEpochs']):
			trainSuccessesVisualDirName = 'train_successes_epoch{}'.format(epoch)
			valSuccessesVisualDirName = 'val_successes_epoch{}'.format(epoch)
			os.mkdir(os.path.join(visualsPath, trainSuccessesVisualDirName))
			os.mkdir(os.path.join(visualsPath, valSuccessesVisualDirName))

			trainLocVisualizer.visualizeQueryLocalizations(trainQuerySuccesses, trainSuccessesVisualDirName)
			valLocVisualizer.visualizeQueryLocalizations(valQuerySuccesses, valSuccessesVisualDirName)

		if (epoch in visualParams['failsVisualEpochs']):
			trainFailsVisualDirName = 'train_fails_epoch{}'.format(epoch)
			valFailsVisualDirName = 'val_fails_epoch{}'.format(epoch)
			os.mkdir(os.path.join(visualsPath, trainFailsVisualDirName))
			os.mkdir(os.path.join(visualsPath, valFailsVisualDirName))

			trainLocVisualizer.visualizeQueryLocalizations(trainQueryFails, trainFailsVisualDirName)
			valLocVisualizer.visualizeQueryLocalizations(valQueryFails, valFailsVisualDirName)


		# Epoch finish
		print('----- Finishing epoch {} -----'.format(epoch))
		print('')

		historian.onEpochEnd(epoch)
		historian.write()
		print('')

		# On-command end
		if (os.path.exists(os.path.join(fullOutputPath, 'end.command'))):
			print('Premature end of training with epoch', epoch)
			print('')
			break

	# Show training summary
	historian.showSummary()

	fileWriter.close()

	# Remove local experiment directory
	if (techParams['moveToSSD']):
		rmtree(experimentSSDPath)

	print('')
	print('Training successfully finished')
