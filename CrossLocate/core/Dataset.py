#!/usr/bin/python3
"""
	Dataset

	Created by Jan Tomesek on 24.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import os
import numpy as np
from pymap3d import Ellipsoid
from pymap3d.vincenty import vdist
import h5py
from math import pi, degrees, radians
import random

class Dataset():
	"""
		Represents dataset set (training/validation/test) by holding its metadata.
		Provides methods for loading metadata from Matlab structures and database place names from info files.
		Provides methods for computing location and yaw distances between samples.
		Provides methods for mining positive and negative samples.

		Parameters:
			positiveYawThreshold ... None for dynamic threshold, Integer for fixed threshold (in degrees)

		Assumptions:
		Coordinates are in UTM format.
		Database yaw angles are in 0..2pi range.
		Loading database place names: Database filenames contain database IDs.
	"""

	def __init__(self, name, meta, set, databasePath, queriesPath, positiveLocationThreshold, positiveYawThreshold, negativeLocationMinThreshold, datasetsMetaPath='../datasets_meta'):
		self.tag = 'Dataset ({})'.format(set)

		self.name = name
		self.meta = meta
		self.set = set
		
		self.databasePath = databasePath
		self.queriesPath = queriesPath

		self.positiveLocationThreshold = positiveLocationThreshold
		self.positiveYawThreshold = positiveYawThreshold
		self.negativeLocationMinThreshold = negativeLocationMinThreshold

		self.datasetsMetaPath = datasetsMetaPath
		self.infoPath = '../datasets_info'

		self.dbImageFns = []
		self.dbUTM = []						# easting, northing
		self.dbCameraParams = []			# yaw [rad], pitch [rad], roll [rad], horizontal FOV [rad]

		self.qImageFns = []
		self.qUTM = []						# easting, northing
		self.qCameraParams = []				# yaw [rad], pitch [rad], roll [rad], horizontal FOV [rad]

		# created

		self.dbPlaceNames = []

		print('[{}] Dataset {}'.format(self.tag, self.meta))


	# Loading


	def loadFromMatlabStructs(self):
		structsPath = os.path.join(self.infoPath, self.meta)
		structFilename = '{}_{}.mat'.format(self.meta, self.set)
		structFile = h5py.File(os.path.join(structsPath, structFilename), 'r')

		print('[{}] Loading dataset ...'.format(self.tag))

		self.__loadStructProperty(structFile, 'dbImageFns')
		self.__loadStructProperty(structFile, 'utmDb')
		self.__loadStructProperty(structFile, 'dbCamParams')

		self.__loadStructProperty(structFile, 'qImageFns')
		self.__loadStructProperty(structFile, 'utmQ')
		self.__loadStructProperty(structFile, 'qCamParams')

		self.__loadStructProperty(structFile, 'posDistThr')
		self.__loadStructProperty(structFile, 'nonTrivPosDistSqThr')

		assert (len(self.dbImageFns) == len(self.dbUtm) and len(self.dbUtm) == len(self.dbCameraParams)), 'Dimensions of individual database properties do not match!'
		assert (len(self.qImageFns) == len(self.qUtm) and len(self.qUtm) == len(self.qCameraParams)), 'Dimensions of individual query properties do not match!'

		# derived

		self.positiveDistSqThr = self.positiveDistThr **2

	def __loadStructProperty(self, structFile, structPropertyName):
		structProperty = structFile['dbStruct/' + structPropertyName]

		if (structPropertyName == 'dbImageFns'):
			for objRef in structProperty[0]:
				filenameObj = structFile[objRef]
				filename = ''.join(chr(c) for c in filenameObj[:])
				self.dbImageFns.append(filename)

			print('[{}] Loaded dbImageFns \t\t\t ({})'.format(self.tag, len(self.dbImageFns)))

		elif (structPropertyName == 'qImageFns'):
			for objRef in structProperty[0]:
				filenameObj = structFile[objRef]
				filename = ''.join(chr(c) for c in filenameObj[:])
				self.qImageFns.append(filename)

			print('[{}] Loaded qImageFns \t\t\t ({})'.format(self.tag, len(self.qImageFns)))

		elif (structPropertyName == 'utmDb'):
			# latitude, longitude
			for dbUtm in structProperty:
				self.dbUtm.append(dbUtm)

			print('[{}] Loaded dbUtm \t\t\t\t ({}, {})'.format(self.tag, len(self.dbUtm), len(self.dbUtm[0])))

		elif (structPropertyName == 'utmQ'):
			# latitude, longitude
			for qUtm in structProperty:
				self.qUtm.append(qUtm)

			print('[{}] Loaded qUtm \t\t\t\t ({}, {})'.format(self.tag, len(self.qUtm), len(self.qUtm[0])))

		elif (structPropertyName == 'dbCamParams'):
			# yaw [rad], pitch [rad], roll [rad], horizontal FOV [rad]
			for i in range(structProperty.shape[1]):
				self.dbCameraParams.append([structProperty[0][i], structProperty[1][i], structProperty[2][i], structProperty[3][i]])

			print('[{}] Loaded dbCamParams \t\t\t ({}, {})'.format(self.tag, len(self.dbCameraParams), len(self.dbCameraParams[0])))

		elif (structPropertyName == 'qCamParams'):
			# yaw [rad], pitch [rad], roll [rad], horizontal FOV [rad]
			for i in range(structProperty.shape[1]):
				self.qCameraParams.append([structProperty[0][i], structProperty[1][i], structProperty[2][i], structProperty[3][i]])

			print('[{}] Loaded qCamParams \t\t\t ({}, {})'.format(self.tag, len(self.qCameraParams), len(self.qCameraParams[0])))

		elif (structPropertyName == 'posDistThr'):
			self.positiveDistThr = structProperty[0][0]

			print('[{}] Loaded positiveDistThr \t\t\t {} m'.format(self.tag, self.positiveDistThr))

		elif (structPropertyName == 'nonTrivPosDistSqThr'):
			self.nonTrivPositiveDistSqThr = structProperty[0][0]

			print('[{}] Loaded nonTrivPositiveDistSqThr \t {} m'.format(self.tag, self.nonTrivPositiveDistSqThr))

	# Loads database place names by mapping set filenames to datasetInfoClean records
	def loadDbPlaceNames(self, databasePath):
		# Load datasetInfoClean file for the (whole) database
		with open(os.path.join(databasePath, 'datasetInfoClean.csv'), 'r') as dbDatasetInfoCleanFile:
			dbInfo = dbDatasetInfoCleanFile.readlines()

		# Parse IDs and place names for the (whole) database
		dbInfoIDs = []
		dbInfoPlaceNames = []
		for line in dbInfo:
			splits = line.split(', ')
			dbInfoIDs.append(splits[0])
			dbInfoPlaceNames.append(splits[1])

		self.dbPlaceNames = []

		# For each database filename (in the current set)
		for dbImageFn in self.dbImageFns:
			dbID, _ = dbImageFn.split('_')

			# Locate database ID in database info
			index = dbInfoIDs.index(dbID)

			# Get corresponding database place name
			self.dbPlaceNames.append(dbInfoPlaceNames[index])

		print('[{}] Loaded dbPlaceNames \t\t\t ({})'.format(self.tag, len(self.dbPlaceNames)))

		assert (len(self.dbImageFns) == len(self.dbPlaceNames)), 'Dimensions of individual database properties do not match!'

	def loadFromMetaStructs(self):
		structsPath = os.path.join(self.datasetsMetaPath, self.name)
		structFn = '{}_{}.npy'.format(self.meta, self.set)
		struct = np.load(os.path.join(structsPath, structFn)).item()

		print('[{}] Loading dataset from {} ...'.format(self.tag, structFn))

		self.dbImageFns = struct['dbImageFns']
		self.dbIDs = struct['dbIDs']
		self.dbPlaceNames = struct['dbPlaceNames']
		self.dbWGS = struct['dbWGS']
		self.dbUTM = struct['dbUTM']
		self.dbElevations = struct['dbElevations']
		self.dbCameraParams = struct['dbCameraParams']

		self.__printDatabaseProperties('Loaded')

		self.qImageFns = struct['qImageFns']
		self.qIDs = struct['qIDs']
		self.qPlaceNames = struct['qPlaceNames']
		self.qWGS = struct['qWGS']
		self.qUTM = struct['qUTM']
		self.qElevations = struct['qElevations']
		self.qCameraParams = struct['qCameraParams']

		self.__printQueryProperties('Loaded')

		self.dbWGS = np.array(self.dbWGS)					# for WGS distance computation

		assert (len(self.dbImageFns) == len(self.dbIDs) == len(self.dbPlaceNames) == len(self.dbWGS) == len(self.dbUTM) == len(self.dbElevations) == len(self.dbCameraParams)), 'Dimensions of individual database properties do not match!'
		assert (len(self.qImageFns) == len(self.qIDs) == len(self.qPlaceNames) == len(self.qWGS) == len(self.qUTM) == len(self.qElevations) == len(self.qCameraParams)), 'Dimensions of individual query properties do not match!'

	def filterDatabaseByQueryLocations(self, locationThreshold):
		totalCloseDbIDs = []

		for index in range(len(self.qImageFns)):
			closeDbIDs, _ = self.__getPositiveDbIDsByLocation(index, locationThreshold, precomputed=False)

			totalCloseDbIDs = self.__addUniqueAndReorder(totalCloseDbIDs, closeDbIDs)		# not really sorted

		totalCloseDbIDs = sorted(totalCloseDbIDs)

		self.dbImageFns = np.array(self.dbImageFns)[totalCloseDbIDs]
		self.dbIDs = np.array(self.dbIDs)[totalCloseDbIDs]
		self.dbPlaceNames = np.array(self.dbPlaceNames)[totalCloseDbIDs]
		self.dbWGS = np.array(self.dbWGS)[totalCloseDbIDs]
		self.dbUTM = np.array(self.dbUTM)[totalCloseDbIDs]
		self.dbElevations = np.array(self.dbElevations)[totalCloseDbIDs]
		self.dbCameraParams = np.array(self.dbCameraParams)[totalCloseDbIDs]

	def saveToMetaStructs(self, datasetParams):
		structsPath = os.path.join(self.datasetsMetaPath, datasetParams['name'])
		structFn = '{}_{}.npy'.format(datasetParams['meta'], self.set)
		struct = {}

		print('[{}] Saving dataset to {} ...'.format(self.tag, structFn))

		struct['dbImageFns'] = self.dbImageFns
		struct['dbIDs'] = self.dbIDs
		struct['dbPlaceNames'] = self.dbPlaceNames
		struct['dbWGS'] = self.dbWGS
		struct['dbUTM'] = self.dbUTM
		struct['dbElevations'] = self.dbElevations
		struct['dbCameraParams'] = self.dbCameraParams

		self.__printDatabaseProperties('Saved')

		struct['qImageFns'] = self.qImageFns
		struct['qIDs'] = self.qIDs
		struct['qPlaceNames'] = self.qPlaceNames
		struct['qWGS'] = self.qWGS
		struct['qUTM'] = self.qUTM
		struct['qElevations'] = self.qElevations
		struct['qCameraParams'] = self.qCameraParams

		self.__printQueryProperties('Saved')

		assert (len(self.dbImageFns) == len(self.dbIDs) == len(self.dbPlaceNames) == len(self.dbWGS) == len(self.dbUTM) == len(self.dbElevations) == len(self.dbCameraParams)), 'Dimensions of individual database properties do not match!'
		assert (len(self.qImageFns) == len(self.qIDs) == len(self.qPlaceNames) == len(self.qWGS) == len(self.qUTM) == len(self.qElevations) == len(self.qCameraParams)), 'Dimensions of individual query properties do not match!'

		fullStructPath = os.path.join(structsPath, structFn)

		assert (not os.path.exists(fullStructPath)), 'Error: Meta structure already exists!'

		np.save(fullStructPath, struct)

	def __printDatabaseProperties(self, verb):
		print('[{}] {} dbImageFns \t\t\t ({})'.format(self.tag, verb, len(self.dbImageFns)))
		print('[{}] {} dbIDs \t\t\t\t ({})'.format(self.tag, verb, len(self.dbIDs)))
		print('[{}] {} dbPlaceNames \t\t\t ({})'.format(self.tag, verb, len(self.dbPlaceNames)))
		print('[{}] {} dbWGS \t\t\t\t ({}, {})'.format(self.tag, verb, len(self.dbWGS), len(self.dbWGS[0])))
		print('[{}] {} dbUTM \t\t\t\t ({}, {})'.format(self.tag, verb, len(self.dbUTM), len(self.dbUTM[0])))
		print('[{}] {} dbElevations \t\t\t ({})'.format(self.tag, verb, len(self.dbElevations)))
		print('[{}] {} dbCameraParams\t\t\t ({}, {})'.format(self.tag, verb, len(self.dbCameraParams), len(self.dbCameraParams[0])))

	def __printQueryProperties(self, verb):
		print('[{}] {} qImageFns \t\t\t ({})'.format(self.tag, verb, len(self.qImageFns)))
		print('[{}] {} qIDs \t\t\t\t ({})'.format(self.tag, verb, len(self.qIDs)))
		print('[{}] {} qPlaceNames \t\t\t ({})'.format(self.tag, verb, len(self.qPlaceNames)))
		print('[{}] {} qWGS \t\t\t\t ({}, {})'.format(self.tag, verb, len(self.qWGS), len(self.qWGS[0])))
		print('[{}] {} qUTM \t\t\t\t ({}, {})'.format(self.tag, verb, len(self.qUTM), len(self.qUTM[0])))
		print('[{}] {} qElevations \t\t\t ({})'.format(self.tag, verb, len(self.qElevations)))
		print('[{}] {} qCameraParams \t\t\t ({}, {})'.format(self.tag, verb, len(self.qCameraParams), len(self.qCameraParams[0])))


	def __len__(self):
		if (len(self.qImageFns) == len(self.qUTM)) and (len(self.qUTM) == len(self.qCameraParams)):
			return len(self.qImageFns)
		else:
			raise Exception('Dimensions of individual (query) properties do not match!')


	# Getters


	def getSet(self):
		return self.set

	def getDatabasePath(self):
		return self.databasePath

	def getQueriesPath(self):
		return self.queriesPath

	def getDatabaseImageFns(self):
		return self.dbImageFns

	def getDatabaseImageFn(self, databaseID):
		return self.dbImageFns[databaseID]

	def getDatabasePlaceName(self, databaseID):
		return self.dbPlaceNames[databaseID]

	def getDatabaseYaw(self, databaseID):
		return self.dbCameraParams[databaseID][0]

	def getQueryImageFns(self):
		return self.qImageFns

	def getQueryImageFn(self, queryID):
		return self.qImageFns[queryID]

	def getQueryYaw(self, queryID, inZeroTwoPi=False):
		qYaw = self.qCameraParams[queryID][0]
		if (not inZeroTwoPi):
			return qYaw
		else:
			return self.__toZeroTwoPi(qYaw)

	def getQueryFOVs(self, inDegrees=False):
		qFOVs = np.array(self.qCameraParams)[:, 3]
		if (not inDegrees):
			return qFOVs
		else:
			return np.degrees(qFOVs)

	def getQueryFOV(self, queryID, inDegrees=False):
		qFOV = self.qCameraParams[queryID][3]
		if (not inDegrees):
			return qFOV
		else:
			return degrees(qFOV)

	# TODO: class method
	def __toZeroTwoPi(self, angle):
		while (angle < 0):
			angle += 2*pi
		while (angle > 2*pi):
			angle -= 2*pi
		return angle


	# Distances


	def closestDatabaseImageByYaw(self, queryID, databaseIDs, inDegrees=False):
		yawDistances = self.computeYawDistances(queryID, databaseIDs, inDegrees)
		
		minYawDistance = min(yawDistances)
		relativeMinDbIndex = yawDistances.index(minYawDistance)

		return relativeMinDbIndex, minYawDistance

	def precomputeLocationDistancesUTM(self):
		queryUTMs = np.array(self.qUTM)
		queryUTMs = queryUTMs[:, None]

		databaseUTMs = np.array(self.dbUTM)
		databaseUTMs = databaseUTMs[None, :]

		diffs = queryUTMs - databaseUTMs
		diffsSquared = np.square(diffs)
		distsSquared = np.sum(diffsSquared, axis=2)

		self.locDistancesSquared = distsSquared
		self.locDistances = np.sqrt(distsSquared)

		print('[{}] Computed location distances (UTM) \t {}'.format(self.tag, self.locDistancesSquared.shape))

	def precomputeLocationDistancesWGS(self):
		self.locDistances = np.zeros((len(self.qWGS), len(self.dbWGS)))

		for qIndex in range(len(self.qWGS)):
			dists = self.computeDistancesWGS(qIndex)

			self.locDistances[qIndex] = dists

		self.locDistancesSquared = np.square(self.locDistances)

		print('[{}] Computed location distances (WGS) \t {}'.format(self.tag, self.locDistances.shape))

	def computeDistancesWGS(self, queryID):
		refEllipsoid = Ellipsoid(model='wgs84')  # unnecessary

		qWGS = self.qWGS[queryID]

		dists, _, _ = vdist(np.repeat(qWGS[1], len(self.dbWGS)), np.repeat(qWGS[0], len(self.dbWGS)),
							self.dbWGS[:, 1], self.dbWGS[:, 0],
							ell=refEllipsoid)
		dists[ np.isnan(dists) ] = 0.0  # Replace NaNs (distances between identical points) with zeros

		return dists

	def loadLocationDistancesWGS(self, loadPath):
		self.locDistances = np.load(loadPath)
		print('[{}] Loaded location distances (WGS) \t {}'.format(self.tag, self.locDistances.shape))

	def saveLocationDistancesWGS(self, savePath):
		np.save(savePath, self.locDistances)
		print('[{}] Saved location distances (WGS) \t {}'.format(self.tag, self.locDistances.shape))

		locDistancesF32 = self.locDistances.astype(np.float32)
		np.save(savePath+'_f32', locDistancesF32)
		print('[{}] Saved location distances (WGS, f32) \t {}'.format(self.tag, locDistancesF32.shape))

	def computeLocationDistances(self, queryID, databaseIDs, squared=True, precomputed=True):
		if (precomputed):
			if (squared):
				locDistances = self.locDistancesSquared[queryID][databaseIDs]
			else:
				locDistances = self.locDistances[queryID][databaseIDs]
		else:
			if (squared):
				raise Exception('Not available!')
			else:
				locDistances = self.computeDistancesWGS(queryID)[databaseIDs]

		return locDistances

	def computeYawDistances(self, queryID, databaseIDs, inDegrees=False):
		queryYaw = self.getQueryYaw(queryID, inZeroTwoPi=True)
		databaseYaws = [self.getDatabaseYaw(dbID) for dbID in databaseIDs]

		diff1 = np.absolute(queryYaw - databaseYaws)
		diff2 = 2*pi - diff1
		dists = np.amin(np.stack((diff1, diff2)), axis=0)

		if (not inDegrees):
			yawDistances = dists
		else:
			yawDistances = np.degrees(dists)

		return yawDistances.tolist()


	# Mining


	def findPositiveDbIDs(self):
		self.positiveDbIDs = []

		for queryID in range(len(self)):
			nonTrivialPositiveDbIDs = self.__getNonTrivialPositiveDbIDs(queryID)
			self.positiveDbIDs.append(nonTrivialPositiveDbIDs)

		print('[{}] Found positive database IDs \t\t ({})'.format(self.tag, len(self.positiveDbIDs)))

	def getPositiveDbIDs(self, queryID, multiple=True):
		if (multiple):
			return self.positiveDbIDs[queryID]
		else:
			if (len(self.positiveDbIDs[queryID]) >= 1):
				return [self.positiveDbIDs[queryID][0]]
			else:
				return self.positiveDbIDs[queryID]

	def loadPositiveDbIDs(self, loadPath):
		self.positiveDbIDs = np.load(loadPath)
		print('[{}] Loaded positive database IDs \t\t ({})'.format(self.tag, len(self.positiveDbIDs)))

	def savePositiveDbIDs(self, savePath):
		np.save(savePath, self.positiveDbIDs)
		print('[{}] Saved positive database IDs \t\t ({})'.format(self.tag, len(self.positiveDbIDs)))

	def findNonNegativeDbIDs(self):
		self.nonNegativeDbIDs = []

		for queryID in range(len(self)):
			nonNegativeDbIDs, _ = self.__getPositiveDbIDsByLocation(queryID, self.negativeLocationMinThreshold)
			self.nonNegativeDbIDs.append(nonNegativeDbIDs)

		print('[{}] Found non-negative database IDs \t ({})'.format(self.tag, len(self.nonNegativeDbIDs)))

	def getNonNegativeDbIDs(self, queryID):
		return self.nonNegativeDbIDs[queryID]

	def loadNonNegativeDbIDs(self, loadPath):
		self.nonNegativeDbIDs = np.load(loadPath)
		print('[{}] Loaded non-negative database IDs \t ({})'.format(self.tag, len(self.nonNegativeDbIDs)))

	def saveNonNegativeDbIDs(self, savePath):
		np.save(savePath, self.nonNegativeDbIDs)
		print('[{}] Saved non-negative database IDs \t ({})'.format(self.tag, len(self.nonNegativeDbIDs)))

	# TODO: nonTriv >=0
	# TODO: yawDistThr <
	def __getNonTrivialPositiveDbIDs(self, queryID):
		# Get positive database IDs (and their distances) by location distance
		positiveDbIDs, distances = self.__getPositiveDbIDsByLocation(queryID, self.positiveLocationThreshold)

		# Keep only positive database IDs with location distances less than positiveLocationThreshold
		positiveDbIDs = [positiveDbID for positiveDbID, dist in zip(positiveDbIDs, distances)
						 if (dist>=0) and (dist <= self.positiveLocationThreshold)]

		# Compute yaw distances between query and positive database
		yawDistances = self.computeYawDistances(queryID, positiveDbIDs)

		# Keep only positive database IDs with yaw distances less than yawDistThr
		if (self.positiveYawThreshold is None):
			qFOV = self.getQueryFOV(queryID)
			yawDistThr = qFOV/4
		else:
			yawDistThr = radians(self.positiveYawThreshold)
		positiveDbIDs = [positiveDbID for positiveDbID, yawDist in zip(positiveDbIDs, yawDistances) if (yawDist <= yawDistThr)]

		return positiveDbIDs

	# TODO: positiveDbIDs: getInds, hash
	def __getPositiveDbIDsByLocation(self, queryID, positiveDistThr, precomputed=True):
		# Get database IDs
		positiveDbIDs = range(len(self.dbImageFns))

		# Compute location distances between query and database
		distances = self.computeLocationDistances(queryID, positiveDbIDs, squared=False, precomputed=precomputed)

		keepPositiveDbIDs = []
		keepDistances = []
		# Keep only positive database IDs (and their distances) with location distances less than positiveDistThr
		for dbID, dist in zip(positiveDbIDs, distances):
			if (dist <= positiveDistThr):
				keepPositiveDbIDs.append(dbID)
				keepDistances.append(dist)

		return keepPositiveDbIDs, keepDistances

	def sampleNegativeDbIDsByLocation(self, queryID, n):
		negativeDbIDs = []

		while (len(negativeDbIDs) < n):
			# Generate random database IDs
			randomDbIDs = random.sample(range(len(self.dbImageFns)), round(n*1.1))

			# Keep only database IDs that are negative for the query (by location)
			randomNegativeDbIDs = self.__filterNegativeDbIDs(queryID, randomDbIDs)

			# Add new negative database IDs
			if (len(randomNegativeDbIDs) > 0):
				negativeDbIDs = self.__addUniqueAndReorder(negativeDbIDs, randomNegativeDbIDs)

		# Keep only n negative database IDs
		if (len(negativeDbIDs) > n):
			negativeDbIDs = random.sample(negativeDbIDs, n)

		return negativeDbIDs

	def __filterPositiveDbIDs(self, queryID, databaseIDs):
		positiveDbIDs, _ = self.__getPositiveDbIDsByLocation(queryID, self.positiveLocationThreshold)
		filteredDbIDs = [dbID for dbID in databaseIDs if dbID in positiveDbIDs]

		return filteredDbIDs

	def __filterNegativeDbIDs(self, queryID, databaseIDs):
		positiveDbIDs = self.getNonNegativeDbIDs(queryID)
		filteredDbIDs = np.setdiff1d(databaseIDs, positiveDbIDs).tolist()

		return filteredDbIDs

	def __addUnique(self, existingIDs, newIDs):
		for newID in newIDs:
			if newID not in existingIDs:
				existingIDs.append(newID)

	def __addUniqueAndReorder(self, existingIDs, newIDs):
		return list((set(existingIDs + newIDs)))
