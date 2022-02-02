#!/usr/bin/python3
"""
	Tensor Assembler

	Created by Jan Tomesek on 28.3.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2019, Jan Tomesek"


import os
import numpy as np
import cv2
import imageio as io
import OpenEXR, Imath


from enum import Enum

class TensorFormat(Enum):
    INPUT_RGB = 1
    INPUT_GRAY_INPUT_GRAY_INPUT_GRAY = 2
    ORIG_GRAY_INPUT_GRAY_INPUT_GRAY = 3
    INPUT_GRAY_DEPTH_1CH_SILH_1CH = 4


class TensorAssembler():
    """
        Assembles tensors from different image modalities.

        Assumptions:
        Image filenames have 3-characters long extensions (jpg, png, exr).
    """

    def __init__(self, datasetName, databasePath, queriesPath, dimensions, channels, databaseTensorFormat, queryTensorFormat):
        self.datasetName = datasetName

        self.databasePath = databasePath
        self.queriesPath = queriesPath

        self.dimensions = dimensions
        self.channels = channels

        self.databaseTensorFormat = databaseTensorFormat
        self.queryTensorFormat = queryTensorFormat

    def setDepthDatabasePath(self, depthDatabasePath):
        self.depthDatabasePath = depthDatabasePath

    def setDepthQueriesPath(self, depthQueriesPath):
        self.depthQueriesPath = depthQueriesPath

    def setSilhouetteDatabasePath(self, silhouetteDatabasePath):
        self.silhouetteDatabasePath = silhouetteDatabasePath

    def setSilhouetteQueriesPath(self, silhouetteQueriesPath):
        self.silhouetteQueriesPath = silhouetteQueriesPath

    def setOriginalQueriesPath(self, originalQueriesPath):
        self.originalQueriesPath = originalQueriesPath

    def getQueryTensor(self, queryFn):
        queryTensor = np.ndarray(shape=(*self.dimensions, self.channels), dtype=np.float32)

        if (self.queryTensorFormat is TensorFormat.INPUT_RGB) and (self.datasetName == 'GeoPose3K_v2_depth'):
            queryImage = self.__loadExr(os.path.join(self.queriesPath, queryFn))
            querySample = cv2.resize(queryImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            queryTensor[:, :, 0] = querySample
            queryTensor[:, :, 1] = querySample
            queryTensor[:, :, 2] = querySample

        elif (self.queryTensorFormat is TensorFormat.INPUT_RGB):
            queryImage = io.imread(os.path.join(self.queriesPath, queryFn), pilmode='RGB')
            querySample = cv2.resize(queryImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            queryTensor = querySample

        elif (self.queryTensorFormat is TensorFormat.INPUT_GRAY_INPUT_GRAY_INPUT_GRAY):
            queryImage = io.imread(os.path.join(self.queriesPath, queryFn), pilmode='RGB', as_gray=True)
            querySample = cv2.resize(queryImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            queryTensor[:, :, 0] = querySample  # R
            queryTensor[:, :, 1] = querySample  # G
            queryTensor[:, :, 2] = querySample  # B

        # elif (self.queryTensorFormat is TensorFormat.ORIG_GRAY_INPUT_GRAY_INPUT_GRAY):
        #     queryImage = io.imread(os.path.join(self.queriesPath, queryFn), pilmode='RGB', as_gray=True)
        #     querySample = cv2.resize(queryImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)
        #
        #     origImage = io.imread(os.path.join(self.originalQueriesPath, queryFn), pilmode='RGB', as_gray=True)
        #     origSample = cv2.resize(origImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)
        #
        #     queryTensor[:, :, 0] = origSample   # R
        #     queryTensor[:, :, 1] = querySample  # G
        #     queryTensor[:, :, 2] = querySample  # B

        elif (self.queryTensorFormat is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
            queryImage = io.imread(os.path.join(self.queriesPath, queryFn), pilmode='RGB', as_gray=True)
            querySample = cv2.resize(queryImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            depthImage = self.__loadExr(os.path.join(self.depthQueriesPath, queryFn[:-4]+'.exr'))
            depthSample = cv2.resize(depthImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            silhouetteImage = io.imread(os.path.join(self.silhouetteQueriesPath, queryFn[:-4]+'.png'))                          # assuming 1-channel uint8 silhouettes (loaded as such)
            silhouetteSample = cv2.resize(silhouetteImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            queryTensor[:, :, 0] = querySample
            queryTensor[:, :, 1] = depthSample
            queryTensor[:, :, 2] = silhouetteSample

        else:
            raise Exception('Unsupported query TensorFormat!')

        return queryTensor

    def getDatabaseTensor(self, databaseFn):
        databaseTensor = np.ndarray(shape=(*self.dimensions, self.channels), dtype=np.float32)

        if (self.databaseTensorFormat is TensorFormat.INPUT_RGB) and (self.datasetName in ['GeoPose3K_v2_depth', 'GeoPose3K_v2_original_to_depth', 'GeoPose3K_v2_original_to_depth_swiss', 'Alps_photos_to_depth_compact', 'Alps_CH1_to_depth', 'Alps_CH2_to_depth_ext']):
            databaseImage = self.__loadExr(os.path.join(self.databasePath, databaseFn))
            databaseSample = cv2.resize(databaseImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            databaseTensor[:, :, 0] = databaseSample
            databaseTensor[:, :, 1] = databaseSample
            databaseTensor[:, :, 2] = databaseSample

        elif (self.databaseTensorFormat is TensorFormat.INPUT_RGB):
            databaseImage = io.imread(os.path.join(self.databasePath, databaseFn), pilmode='RGB')
            databaseSample = cv2.resize(databaseImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            databaseTensor = databaseSample

        elif (self.databaseTensorFormat is TensorFormat.INPUT_GRAY_INPUT_GRAY_INPUT_GRAY):
            databaseImage = io.imread(os.path.join(self.databasePath, databaseFn), pilmode='RGB', as_gray=True)
            databaseSample = cv2.resize(databaseImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            databaseTensor[:, :, 0] = databaseSample  # R
            databaseTensor[:, :, 1] = databaseSample  # G
            databaseTensor[:, :, 2] = databaseSample  # B

        elif (self.databaseTensorFormat is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
            databaseImage = io.imread(os.path.join(self.databasePath, databaseFn), pilmode='RGB', as_gray=True)
            databaseSample = cv2.resize(databaseImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            depthImage = self.__loadExr(os.path.join(self.depthDatabasePath, '_'.join(databaseFn.split('_')[:-1])+'_depth.exr'))
            depthSample = cv2.resize(depthImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            silhouetteImage = io.imread(os.path.join(self.silhouetteDatabasePath, '_'.join(databaseFn.split('_')[:-1])+'_silhouettes.png'))     # assuming 1-channel uint8 silhouettes (loaded as such)
            silhouetteSample = cv2.resize(silhouetteImage, dsize=self.dimensions, interpolation=cv2.INTER_AREA)

            databaseTensor[:, :, 0] = databaseSample
            databaseTensor[:, :, 1] = depthSample
            databaseTensor[:, :, 2] = silhouetteSample

        else:
            raise Exception('Unsupported database TensorFormat!')

        return databaseTensor

    def __loadExr(self, fullImagePath):
        file = OpenEXR.InputFile(fullImagePath)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        dataChannel = file.channel('R', pt)
        image = np.fromstring(dataChannel, dtype=np.float32)

        header = file.header()

        dw = header['dataWindow']
        dimensions = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        image.shape = (dimensions[1], dimensions[0])

        return image
