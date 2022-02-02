#!/usr/bin/python3
"""
	Triplet Visualizer

	Created by Jan Tomesek on 26.3.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2019, Jan Tomesek"


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import ceil

from core.TensorAssembler import TensorFormat


from enum import Enum

class VisualMode(Enum):
    BATCH = 1
    LOC = 2


class TripletVisualizer():
    """
        Visualizes triplet batches.
    """

    def __init__(self, mode, datasetName, batchSize, numberOfNegatives, databaseTensorFormat, queryTensorFormat, visualsPath):
        self.mode = mode

        self.datasetName = datasetName

        self.batchSize = batchSize
        self.numberOfNegatives = numberOfNegatives

        self.databaseTensorFormat = databaseTensorFormat
        self.queryTensorFormat = queryTensorFormat

        self.visualsPath = visualsPath

        self.tripletSize = 1 + 1 + self.numberOfNegatives

        self.innerGridHSpace = -0.25

        if (self.mode is VisualMode.BATCH):
            self.positiveLabel = 'positive'
            self.negativeLabel = 'negative'
        elif (self.mode is VisualMode.LOC):
            self.positiveLabel = 'correct'
            self.negativeLabel = 'candidate'

    def setupForQueryLocalizationsVisualization(self, dataset, tensorAssembler, dimensions, channels, locationThresholds):
        self.dataset = dataset
        self.tensorAssembler = tensorAssembler
        self.dimensions = dimensions
        self.channels = channels
        self.locationThresholds = locationThresholds

    def visualizeQueryLocalizations(self, queryLocalizations, visualDirName):
        numberOfBatches = ceil(len(queryLocalizations) / self.batchSize)

        for batch in range(numberOfBatches):
            startBatchIndex = batch*self.batchSize
            endBatchIndex = (batch+1)*self.batchSize

            batchQueryLocalizations = queryLocalizations[startBatchIndex : endBatchIndex]

            currentBatchSize = len(batchQueryLocalizations)

            queryBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)
            correctBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)
            candidateBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)

            queryFnsBatch = []
            correctFnsBatch = []
            candidateFnsBatch = []

            queryLocDistancesBatch = []
            correctLocDistancesBatch = []
            candidateLocDistancesBatch = []

            queryYawDistancesBatch = []
            correctYawDistancesBatch = []
            candidateYawDistancesBatch = []

            for queryLoc in batchQueryLocalizations:
                queryID = queryLoc[0]
                correctID = self.dataset.getPositiveDbIDs(queryID, multiple=False)
                candidateIDs = queryLoc[1]

                queryTensor = np.ndarray(shape=(1, *self.dimensions, self.channels), dtype=np.float32)
                correctTensor = np.ndarray(shape=(1, *self.dimensions, self.channels), dtype=np.float32)
                candidateTensor = np.ndarray(shape=(10, *self.dimensions, self.channels), dtype=np.float32)

                # Fill query tensor
                queryFn = self.dataset.getQueryImageFn(queryID)
                queryTensor[0] = self.tensorAssembler.getQueryTensor(queryFn)

                queryFnsBatch.append(queryFn)
                queryLocDistancesBatch.append(None)
                queryYawDistancesBatch.append(None)

                # Fill correct tensor
                if (len(correctID) >= 1):
                    correctFn = self.dataset.getDatabaseImageFn(correctID[0])
                    correctTensor[0] = self.tensorAssembler.getDatabaseTensor(correctFn)

                    correctFnsBatch.append(correctFn)
                    correctLocDistancesBatch += self.dataset.computeLocationDistances(queryID, correctID, squared=False).tolist()
                    correctYawDistancesBatch += self.dataset.computeYawDistances(queryID, correctID, inDegrees=True)
                else:
                    correctFn = ''
                    correctTensor[0] = np.zeros(shape=(*self.dimensions, self.channels), dtype=np.float32)

                    correctFnsBatch.append(correctFn)
                    correctLocDistancesBatch += [None]
                    correctYawDistancesBatch += [None]

                # Fill candidate tensor
                for candIndex, candID in enumerate(candidateIDs):
                    candidateFn = self.dataset.getDatabaseImageFn(candID)
                    candidateTensor[candIndex] = self.tensorAssembler.getDatabaseTensor(candidateFn)

                    candidateFnsBatch.append(candidateFn)
                candidateLocDistancesBatch += self.dataset.computeLocationDistances(queryID, candidateIDs, squared=False).tolist()
                candidateYawDistancesBatch += self.dataset.computeYawDistances(queryID, candidateIDs, inDegrees=True)

                queryBatch = np.concatenate((queryBatch, queryTensor))
                correctBatch = np.concatenate((correctBatch, correctTensor))
                candidateBatch = np.concatenate((candidateBatch, candidateTensor))

            tripletBatch = np.concatenate((queryBatch, correctBatch, candidateBatch))
            tripletFnsBatch = queryFnsBatch + correctFnsBatch + candidateFnsBatch
            tripletLocDistancesBatch = queryLocDistancesBatch + correctLocDistancesBatch + candidateLocDistancesBatch
            tripletYawDistancesBatch = queryYawDistancesBatch + correctYawDistancesBatch + candidateYawDistancesBatch

            self.visualizeBatch(tripletBatch, tripletFnsBatch, tripletLocDistancesBatch, tripletYawDistancesBatch, visualDirName, batch, batchSize=currentBatchSize, numberOfNegatives=10, tripletSize=1+1+10)


    def visualizeBatch(self, batch, fnsBatch, locDistancesBatch, yawDistancesBatch, visualDirName, batchIndex, batchSize=None, numberOfNegatives=None, tripletSize=None):
        if (batchSize == None):
            batchSize = self.batchSize
        if (numberOfNegatives == None):
            numberOfNegatives = self.numberOfNegatives
        if (tripletSize == None):
            tripletSize = self.tripletSize

        # Assertions on batch size
        assert (batch.shape[0] == (batchSize*tripletSize)), 'Batch size does not correspond with batch size parameter!'
        assert (len(fnsBatch) == (batchSize*tripletSize)), 'Filenames batch size does not correspond with batch size parameter!'
        assert (len(locDistancesBatch) == (batchSize*tripletSize)), 'Location distances batch size does not correspond with batch size parameter!'
        assert (len(yawDistancesBatch) == (batchSize*tripletSize)), 'Yaw distances batch size does not correspond with batch size parameter!'

         # Query tensor format check
        if (self.queryTensorFormat is TensorFormat.INPUT_RGB) and (self.datasetName == 'GeoPose3K_v2_depth'):
            qInnerGridDim = 2
        elif (self.queryTensorFormat is TensorFormat.INPUT_RGB):
            qInnerGridDim = 1
        elif (self.queryTensorFormat is TensorFormat.INPUT_GRAY_INPUT_GRAY_INPUT_GRAY):
            qInnerGridDim = 2
        elif (self.queryTensorFormat is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
            qInnerGridDim = 2
        else:
            raise Exception('Unsupported query TensorFormat!')

        # Database tensor format check
        if (self.databaseTensorFormat is TensorFormat.INPUT_RGB) and (self.datasetName in ['GeoPose3K_v2_depth', 'GeoPose3K_v2_original_to_depth_swiss', 'Alps_photos_to_depth_compact', 'Alps_CH1_to_depth', 'Alps_CH2_to_depth_ext']):
            dbInnerGridDim = 2
        elif (self.databaseTensorFormat is TensorFormat.INPUT_RGB):
            dbInnerGridDim = 1
        elif (self.databaseTensorFormat is TensorFormat.INPUT_GRAY_INPUT_GRAY_INPUT_GRAY):
            dbInnerGridDim = 2
        elif (self.databaseTensorFormat is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
            dbInnerGridDim = 2
        else:
            raise Exception('Unsupported database TensorFormat!')

        # Type conversion for correct visualization of multi-channel images
        batch  = batch.astype(int)

        # Batch split
        queryBatch = batch[0*batchSize : 1*batchSize]
        positiveBatch = batch[1*batchSize : 2*batchSize]
        negativeBatch = batch[2*batchSize : tripletSize*batchSize]

        # Filenames batch split
        queryFnsBatch = fnsBatch[0*batchSize : 1*batchSize]
        positiveFnsBatch = fnsBatch[1*batchSize : 2*batchSize]
        negativeFnsBatch = fnsBatch[2*batchSize : tripletSize*batchSize]

        # Location distances batch split
        queryLocDistancesBatch = locDistancesBatch[0*batchSize : 1*batchSize]
        positiveLocDistancesBatch = locDistancesBatch[1*batchSize : 2*batchSize]
        negativeLocDistancesBatch = locDistancesBatch[2*batchSize : tripletSize*batchSize]

        # Yaw distances batch split
        queryYawDistancesBatch = yawDistancesBatch[0*batchSize : 1*batchSize]
        positiveYawDistancesBatch = yawDistancesBatch[1*batchSize : 2*batchSize]
        negativeYawDistancesBatch = yawDistancesBatch[2*batchSize : tripletSize*batchSize]

        # Figure
        fig = plt.figure('Batch', figsize=(18, 2.5*batchSize))

        # Outer grid
        outerGrid = gridspec.GridSpec(batchSize, tripletSize)

        # For all triplets in the batch
        for tripletIndex in range(batchSize):
            # Triplet split
            queryImage = queryBatch[tripletIndex]
            positiveImage = positiveBatch[tripletIndex]
            negativeImages = negativeBatch[tripletIndex*numberOfNegatives : (tripletIndex+1)*numberOfNegatives]

            # Filenames triplet split
            queryFn = queryFnsBatch[tripletIndex]
            positiveFn = positiveFnsBatch[tripletIndex]
            negativeFns = negativeFnsBatch[tripletIndex*numberOfNegatives : (tripletIndex+1)*numberOfNegatives]

            # Location distances triplet split
            positiveLocDistance = positiveLocDistancesBatch[tripletIndex]
            negativeLocDistances = negativeLocDistancesBatch[tripletIndex*numberOfNegatives : (tripletIndex+1)*numberOfNegatives]

            # Yaw distances triplet split
            positiveYawDistance = positiveYawDistancesBatch[tripletIndex]
            negativeYawDistances = negativeYawDistancesBatch[tripletIndex*numberOfNegatives : (tripletIndex+1)*numberOfNegatives]

            # Show query sample
            queryInnerGrid = gridspec.GridSpecFromSubplotSpec(qInnerGridDim, qInnerGridDim, subplot_spec=outerGrid[0 + (tripletIndex*tripletSize)], hspace=self.innerGridHSpace)
            self.__visualizeQueryTensor(fig, queryInnerGrid, queryImage, queryFn)

            # Show positive sample
            positiveInnerGrid = gridspec.GridSpecFromSubplotSpec(dbInnerGridDim, dbInnerGridDim, subplot_spec=outerGrid[1 + (tripletIndex*tripletSize)], hspace=self.innerGridHSpace)
            self.__visualizeDatabaseTensor(self.positiveLabel, fig, positiveInnerGrid, positiveImage, positiveFn, positiveLocDistance, positiveYawDistance)

            # Show negative samples
            for index, (negativeImage, negativeFn, negativeLocDistance, negativeYawDistance) in enumerate(zip(negativeImages, negativeFns, negativeLocDistances, negativeYawDistances)):
                negativeInnerGrid = gridspec.GridSpecFromSubplotSpec(dbInnerGridDim, dbInnerGridDim, subplot_spec=outerGrid[2 + index + (tripletIndex*tripletSize)], hspace=self.innerGridHSpace)
                self.__visualizeDatabaseTensor(self.negativeLabel, fig, negativeInnerGrid, negativeImage, negativeFn, negativeLocDistance, negativeYawDistance)

        plt.subplots_adjust(left=0.05, top=0.92, right=0.98, bottom=0.05, wspace=0.10, hspace=0.50)

        #plt.show()
        plt.savefig(os.path.join(self.visualsPath, visualDirName, 'batch_{}.png'.format(batchIndex)), dpi=200)

        plt.close()

    def __visualizeQueryTensor(self, figure, queryInnerGrid, queryImage, queryFn):
        title = 'query\n{}'.format(queryFn[:-4])

        if (self.queryTensorFormat is TensorFormat.INPUT_RGB) and (self.datasetName == 'GeoPose3K_v2_depth'):
            self.__visualizeImage(figure, queryInnerGrid[0], queryImage[:, :, 0], title=title)
            self.__visualizeImage(figure, queryInnerGrid[1], queryImage[:, :, 1])
            self.__visualizeImage(figure, queryInnerGrid[2], queryImage[:, :, 2])
            self.__visualizeImage(figure, queryInnerGrid[3], queryImage)

        elif (self.queryTensorFormat is TensorFormat.INPUT_RGB):
            self.__visualizeImage(figure, queryInnerGrid[0], queryImage, title=title, titleLoc='right')

        elif (self.queryTensorFormat is TensorFormat.INPUT_GRAY_INPUT_GRAY_INPUT_GRAY):
            self.__visualizeImage(figure, queryInnerGrid[0], queryImage[:, :, 0], title=title)
            self.__visualizeImage(figure, queryInnerGrid[1], queryImage[:, :, 1])
            self.__visualizeImage(figure, queryInnerGrid[2], queryImage[:, :, 2])
            self.__visualizeImage(figure, queryInnerGrid[3], queryImage)

        elif (self.queryTensorFormat is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
            self.__visualizeImage(figure, queryInnerGrid[0], queryImage[:, :, 0], title=title)
            self.__visualizeImage(figure, queryInnerGrid[1], queryImage[:, :, 1])
            self.__visualizeImage(figure, queryInnerGrid[2], queryImage[:, :, 2])
            self.__visualizeImage(figure, queryInnerGrid[3], queryImage)

        else:
            raise Exception('Unsupported query TensorFormat!')

    def __visualizeDatabaseTensor(self, type, figure, databaseInnerGrid, databaseImage, databaseFn, databaseLocDistance, databaseYawDistance):
        title = type + '\n'
        if ('/' in databaseFn):
            tileName, imageFn = databaseFn.split('/')
            tileNameParts = tileName.split('_')
            imageFnParts = imageFn.split('_')
            title += tileNameParts[-2]+'_'+tileNameParts[-1] +'/'+ imageFnParts[0]+'_'+imageFnParts[1][:3]   # keep only "<tile_numbers>/<image_ID>_<mod>"
        else:
            title += (databaseFn[:-4] if (databaseFn != 'repeat') else databaseFn)

        subTitle = '{:,.1f} m'.format(databaseLocDistance).replace(',', ' ') if (databaseLocDistance is not None) else ''
        subTitleStyled = True if (databaseLocDistance is not None) else False

        subSubTitle = '{:,.0f} \xb0'.format(databaseYawDistance) if (databaseYawDistance is not None) else ''
        subSubTitleStyled = True if (databaseYawDistance is not None) else False

        subTitleColor = 'antiquewhite'
        subSubTitleColor = 'antiquewhite'
        borderStyled = False
        borderColor = None
        if ((self.mode == VisualMode.LOC) and (type == self.negativeLabel)):
            if (databaseLocDistance <= self.locationThresholds[-1]):
                subTitleColor = 'limegreen'
                subSubTitleColor = 'orange'
            if ((databaseLocDistance <= self.locationThresholds[-1]) and (databaseYawDistance <= 30.0)):
                subSubTitleColor = 'limegreen'

            if (databaseLocDistance <= self.locationThresholds[0]):
                subTitleColor = 'lime'
                subSubTitleColor = 'orange'
            if ((databaseLocDistance <= self.locationThresholds[0]) and (databaseYawDistance <= 30.0)):
                subSubTitleColor = 'lime'
                borderStyled = True
                borderColor = 'lime'

        if (self.databaseTensorFormat is TensorFormat.INPUT_RGB) and (self.datasetName in ['GeoPose3K_v2_depth', 'GeoPose3K_v2_original_to_depth_swiss', 'Alps_photos_to_depth_compact', 'Alps_CH1_to_depth', 'Alps_CH2_to_depth_ext']):
            self.__visualizeImage(figure, databaseInnerGrid[0], databaseImage[:, :, 0])
            self.__visualizeImage(figure, databaseInnerGrid[1], databaseImage[:, :, 1], title=title)
            self.__visualizeImage(figure, databaseInnerGrid[2], databaseImage[:, :, 2])
            self.__visualizeImage(figure, databaseInnerGrid[3], databaseImage,
                                  subTitle=subTitle, subTitleStyled=subTitleStyled, subTitleColor=subTitleColor,
                                  subSubTitle=subSubTitle, subSubTitleStyled=subSubTitleStyled, subSubTitleColor=subSubTitleColor,
                                  borderStyled=borderStyled, borderColor=borderColor)

        elif (self.databaseTensorFormat is TensorFormat.INPUT_RGB):
            self.__visualizeImage(figure, databaseInnerGrid[0], databaseImage, title=title,
                                  subTitle=subTitle, subTitleStyled=subTitleStyled, subTitleColor=subTitleColor,
                                  subSubTitle=subSubTitle, subSubTitleStyled=subSubTitleStyled, subSubTitleColor=subSubTitleColor,
                                  borderStyled=borderStyled, borderColor=borderColor)

        elif (self.databaseTensorFormat is TensorFormat.INPUT_GRAY_INPUT_GRAY_INPUT_GRAY):
            self.__visualizeImage(figure, databaseInnerGrid[0], databaseImage[:, :, 0])
            self.__visualizeImage(figure, databaseInnerGrid[1], databaseImage[:, :, 1], title=title)
            self.__visualizeImage(figure, databaseInnerGrid[2], databaseImage[:, :, 2])
            self.__visualizeImage(figure, databaseInnerGrid[3], databaseImage,
                                  subTitle=subTitle, subTitleStyled=subTitleStyled, subTitleColor=subTitleColor,
                                  subSubTitle=subSubTitle, subSubTitleStyled=subSubTitleStyled, subSubTitleColor=subSubTitleColor,
                                  borderStyled=borderStyled, borderColor=borderColor)

        elif (self.databaseTensorFormat is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
            self.__visualizeImage(figure, databaseInnerGrid[0], databaseImage[:, :, 0])
            self.__visualizeImage(figure, databaseInnerGrid[1], databaseImage[:, :, 1], title=title)
            self.__visualizeImage(figure, databaseInnerGrid[2], databaseImage[:, :, 2])
            self.__visualizeImage(figure, databaseInnerGrid[3], databaseImage,
                                  subTitle=subTitle, subTitleStyled=subTitleStyled, subTitleColor=subTitleColor,
                                  subSubTitle=subSubTitle, subSubTitleStyled=subSubTitleStyled, subSubTitleColor=subSubTitleColor,
                                  borderStyled=borderStyled, borderColor=borderColor)

        else:
            raise Exception('Unsupported database TensorFormat!')

    def __visualizeImage(self, figure, plot, image, title='', titleLoc='center',
                         subTitle='', subTitleStyled=False, subTitleColor='antiquewhite',
                         subSubTitle='', subSubTitleStyled=False, subSubTitleColor='antiquewhite',
                         borderStyled=False, borderColor='lime'):
        axes = plt.Subplot(figure, plot)
        axes.set_title(title, loc=titleLoc, fontsize=8)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        # Sub title
        subTitleProps = dict(boxstyle='round', facecolor=subTitleColor) if subTitleStyled else None
        axes.text(0.5, -0.08, subTitle,
                  horizontalalignment='center', verticalalignment='top', transform=axes.transAxes,
                  bbox=subTitleProps, fontsize=8)

        # Sub sub title
        subSubTitleProps = dict(boxstyle='round', facecolor=subSubTitleColor) if subSubTitleStyled else None
        axes.text(0.5, -0.24, subSubTitle,
                  horizontalalignment='center', verticalalignment='top', transform=axes.transAxes,
                  bbox=subSubTitleProps, fontsize=8)

        if (borderStyled):
            axes.patch.set_edgecolor(borderColor)
            axes.patch.set_linewidth('8')

        figure.add_subplot(axes)
        plt.imshow(image)
