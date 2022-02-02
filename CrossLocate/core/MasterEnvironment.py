"""
    Master Environment

    Created by Jan Tomesek on 1.7.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "tomesek.j@gmail.com"
__copyright__ = "Copyright 2019, Jan Tomesek"


import os
import subprocess
import tensorflow as tf
from shutil import copyfile
from zipfile import ZipFile

class MasterEnvironment:
    """
        Superclass representing software and hardware environment.
    """

    tag = 'MasterEnvironment'

    @staticmethod
    def seizeGPU():
        freeGPU = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
        freeGPU = freeGPU.decode().strip()

        if (len(freeGPU) == 0):
            print('[{}] No free GPU available!'.format(MasterEnvironment.tag))
            return -1

        os.environ['CUDA_VISIBLE_DEVICES'] = freeGPU                            # only this device will be seen
        print('[{}] Seizing GPU: {}'.format(MasterEnvironment.tag, freeGPU))

        return freeGPU

    @staticmethod
    def showInfo(tensorflow=True, numpy=True, opencv=True, imageio=True, sklearn=True):
        if (tensorflow):
            print('[{}] TensorFlow: \t{}'.format(MasterEnvironment.tag, tf.__version__))
        if (numpy):
            import numpy as np
            print('[{}] NumPy:\t\t{}'.format(MasterEnvironment.tag, np.__version__))
        if (opencv):
            import cv2
            print('[{}] OpenCV:\t\t{}'.format(MasterEnvironment.tag, cv2.__version__))
        if (imageio):
            import imageio as io
            print('[{}] Imageio:\t\t{}'.format(MasterEnvironment.tag, io.__version__))
        if (sklearn):
            import sklearn
            print('[{}] Scikit-learn:\t{}'.format(MasterEnvironment.tag, sklearn.__version__))

        print('')
        print('[{}] Available GPUs:\t{}'.format(MasterEnvironment.tag, tf.test.gpu_device_name()))

    @staticmethod
    def moveDataset(srcDatasetPartPaths, dstPath, dstSubdirName='dataset'):
        # Create destination dataset directory
        dstDatasetPath = os.path.join(dstPath, dstSubdirName)
        try:
            os.mkdir(dstDatasetPath)
        except FileExistsError:
            pass

        dstDatasetPartPaths = []

        # For each dataset part
        for fullSrcPartPath in srcDatasetPartPaths:
            # Create destination part path
            fullDstPartPath = os.path.join(dstDatasetPath, os.path.basename(fullSrcPartPath))

            # Create part ZIP paths
            fullSrcPartZipPath = fullSrcPartPath + '.zip'
            fullDstPartZipPath = fullDstPartPath + '.zip'

            # Copy part ZIP file to destination dataset directory
            copyfile(fullSrcPartZipPath, fullDstPartZipPath)

            # Extract part ZIP file
            with ZipFile(fullDstPartZipPath, 'r') as partZipFile:
                partZipFile.extractall(dstDatasetPath)

            # Remove part ZIP file
            os.remove(fullDstPartZipPath)

            # Store destination part path
            dstDatasetPartPaths.append(fullDstPartPath)

        return dstDatasetPartPaths
