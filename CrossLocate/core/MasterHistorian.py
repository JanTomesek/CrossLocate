"""
	Master Historian

	Created by Jan Tomesek on 4.9.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "tomesek.j@gmail.com"
__copyright__ = "Copyright 2019, Jan Tomesek"


import os
import numpy as np
import time
from statistics import mean

class MasterHistorian:
    """
        Superclass for storing, writing and loading history.
        Provides epoch time measuring.
    """

    def __init__(self, outputPath, tag='MasterHistorian'):
        self.outputPath = outputPath

        self.tag = tag

        self.startEpochTime = 0
        self.epochTimeHistory = []

        print('[{}] Initialized'.format(self.tag))

    def onEpochStart(self, epoch):
        # Start epoch time
        currentTime = time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())
        print('[{}] {}: Epoch {} start'.format(self.tag, currentTime, epoch))

        self.startEpochTime = time.time()

    def onEpochEnd(self, epoch):
        # End epoch time
        endEpochTime = time.time()

        epochTime = endEpochTime - self.startEpochTime
        self.epochTimeHistory.append(epochTime)

        hours, minutes = self.__toHoursAndMinutes(epochTime)

        currentTime = time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())
        print('[{}] {}: Epoch {} end (epoch took {}h {}min)'.format(self.tag, currentTime, epoch, hours, minutes))

    def __toHoursAndMinutes(self, seconds):
        hours = round(seconds // 3600)
        remaining = seconds - (hours * 3600)
        minutes = round(remaining // 60)

        return hours, minutes

    def showSummary(self):
        print('[{}] -------------------- Summary --------------------'.format(self.tag))
        print('')


        # Times
        print('[{}] -------- Times --------'.format(self.tag))

        # Epoch times
        for index, epochTime in enumerate(self.epochTimeHistory):
            hours, minutes = self.__toHoursAndMinutes(epochTime)
            print('[{}] Epoch {}: \t {}h {}min'.format(self.tag, index+1, hours, minutes))
        print('')

        # Average epoch time
        averageEpochTime = mean(self.epochTimeHistory)
        averageHours, averageMinutes = self.__toHoursAndMinutes(averageEpochTime)
        print('[{}] Average: \t {}h {}min'.format(self.tag, averageHours, averageMinutes))

        # Total epoch time
        totalEpochTime = sum(self.epochTimeHistory)
        totalHours, totalMinutes = self.__toHoursAndMinutes(totalEpochTime)
        print('[{}] Total: \t {}h {}min'.format(self.tag, totalHours, totalMinutes))

    def load(self, overridden=False):
        self.epochTimeHistory = np.load(os.path.join(self.outputPath, 'epochTimeHistory'+'.npy')).tolist()

        if (not overridden):
            print('[{}] History successfully loaded'.format(self.tag))

    def write(self, overridden=False):
        np.save(os.path.join(self.outputPath, 'epochTimeHistory'), self.epochTimeHistory)

        if (not overridden):
            print('[{}] History successfully written'.format(self.tag))
