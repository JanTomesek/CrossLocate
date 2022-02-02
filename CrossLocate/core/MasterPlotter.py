"""
    Master Plotter

    Created by Jan Tomesek on 24.9.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "tomesek.j@gmail.com"
__copyright__ = "Copyright 2019, Jan Tomesek"


import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MasterPlotter:
    """
        Superclass for plotting charts.
    """

    def __init__(self, outputPath, tag='MasterPlotter'):
        self.outputPath = outputPath

        self.tag = tag

        print('[{}] Initialized'.format(self.tag))

    def _plotSingle(self, xAxis, values, title, xLabel, yLabel, color, fileName, xNorm=False, yNorm=False, grid=False, marker=''):
        if (xAxis is None):
            xAxis = range(1, len(values)+1)

        plt.figure()

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        if (xNorm):
            plt.xlim(-0.05, 1.05)
            xTicks = np.arange(0.0, 1.0 + 0.01, step=0.1)
            plt.xticks(xTicks)

        if (yNorm):
            plt.ylim(-0.05, 1.05)
            yTicks = np.arange(0.0, 1.0 + 0.01, step=0.1)
            plt.yticks(yTicks)

        if (grid):
            plt.grid()

        plt.plot(xAxis, values, color, marker=marker)

        plt.savefig(os.path.join(self.outputPath, fileName+'.png'))

        plt.close()

    def _plotMultiple(self, xAxis, multiValues, colors, labels, markers, title, xLabel, yLabel, xStart, fileName, xTicks=None, yNorm=False):
        if (xAxis is None):
            xAxis = range(xStart, len(multiValues[0])+xStart)

        plt.figure()

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        if (xTicks is None):
            # dynamic x ticks
            xTicksStep = int(np.ceil((len(multiValues[0]) -1) / 20))    # increment x ticks step after every 20 values (2-21, 22-41, 42-61, ... ranges)
            xTicksStep = max(xTicksStep, 1)                             # case of single value
            xTicks = list(xAxis[0::xTicksStep])
            xTicks[-1] = xAxis[-1]                                      # ensure presence of the last x tick
        plt.xticks(xTicks, fontsize=8)

        if (yNorm):
            plt.ylim(-0.05, 1.05)
            yTicks = np.arange(0.0, 1.0 + 0.01, step=0.1)
            plt.yticks(yTicks)

        for values, color, label, marker in zip(multiValues, colors, labels, markers):
            plt.plot(xAxis, values, color, label=label, marker=marker, markersize=4)

        plt.legend(fontsize=8)

        plt.savefig(os.path.join(self.outputPath, fileName+'.png'), dpi=200)

        plt.close()
