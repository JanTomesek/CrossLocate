#!/usr/bin/python3
"""
	Augmentator

	Created by Jan Tomesek on 16.12.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2019, Jan Tomesek"


import numpy as np
import cv2
import random

class Augmentator():

    def __init__(self):
        self.augmentations = {}

        self.lastFlipped = None

        self.appearanceAugmentation = False

    def activateShiftAugmentation(self, maxHorizontalPercentage, maxVerticalPercentage, probability=0.5):
        self.augmentations['shift'] = {}
        self.augmentations['shift']['maxHorizontalPercentage'] = maxHorizontalPercentage
        self.augmentations['shift']['maxVerticalPercentage'] = maxVerticalPercentage
        self.augmentations['shift']['prob'] = probability

    def activateRotationAugmentation(self, maxAngle, probability=0.5):
        self.augmentations['rotation'] = {}
        self.augmentations['rotation']['maxAngle'] = maxAngle
        self.augmentations['rotation']['prob'] = probability

    def activateFlipAugmentation(self, probability=0.5):
        self.augmentations['flip'] = {}
        self.augmentations['flip']['prob'] = probability

    def activateBrightnessAugmentation(self, maxMagnitude, probability=0.5):
        self.augmentations['brightness'] = {}
        self.augmentations['brightness']['maxMagnitude'] = maxMagnitude
        self.augmentations['brightness']['prob'] = probability

        self.appearanceAugmentation = True

    def activateHueAugmentation(self, maxMagnitude, probability=0.5):
        self.augmentations['hue'] = {}
        self.augmentations['hue']['maxMagnitude'] = maxMagnitude
        self.augmentations['hue']['prob'] = probability

        self.appearanceAugmentation = True

    def activateSaturationAugmentation(self, maxMagnitude, probability=0.5):
        self.augmentations['saturation'] = {}
        self.augmentations['saturation']['maxMagnitude'] = maxMagnitude
        self.augmentations['saturation']['prob'] = probability

        self.appearanceAugmentation = True

    def activateContrastAugmentation(self, maxPercentage, probability=0.5):
        self.augmentations['contrast'] = {}
        self.augmentations['contrast']['maxPercentage'] = maxPercentage
        self.augmentations['contrast']['prob'] = probability

        self.appearanceAugmentation = True

    def activateBlurAugmentation(self, maxSigmaPercentage, probability=0.5):
        self.augmentations['blur'] = {}
        self.augmentations['blur']['maxSigmaPercentage'] = maxSigmaPercentage
        self.augmentations['blur']['prob'] = probability

        self.appearanceAugmentation = True

    def activateNoiseAugmentation(self, std, probability=0.5):
        self.augmentations['noise'] = {}
        self.augmentations['noise']['std'] = std
        self.augmentations['noise']['prob'] = probability

        self.appearanceAugmentation = True

    def augmentImage(self, image, applyFlip=False, disableFlip=False):
        if (self.appearanceAugmentation):
            image = image.astype(np.uint8)

        height, width, _ = image.shape

        if ('shift' in self.augmentations):
            if (random.random() < self.augmentations['shift']['prob']):
                maxHorizontalShift = int(width * self.augmentations['shift']['maxHorizontalPercentage'])
                maxVerticalShift = int(height * self.augmentations['shift']['maxVerticalPercentage'])
                image = self.__shiftImage(image, maxHorizontalShift, maxVerticalShift)

        if ('rotation' in self.augmentations):
            if (random.random() < self.augmentations['rotation']['prob']):
                image = self.__rotateImage(image, self.augmentations['rotation']['maxAngle'])

        if ('flip' in self.augmentations) and (not disableFlip):
            if (random.random() < self.augmentations['flip']['prob']) or (applyFlip):
                image = self.__flipImage(image)
                self.lastFlipped = True
            else:
                self.lastFlipped = False

        if ('brightness' in self.augmentations):
            if (random.random() < self.augmentations['brightness']['prob']):
                image = self.__changeBrightness(image, self.augmentations['brightness']['maxMagnitude'])

        if ('hue' in self.augmentations):
            if (random.random() < self.augmentations['hue']['prob']):
                image = self.__changeHue(image, self.augmentations['hue']['maxMagnitude'])

        if ('saturation' in self.augmentations):
            if (random.random() < self.augmentations['saturation']['prob']):
                image = self.__changeSaturation(image, self.augmentations['saturation']['maxMagnitude'])

        if ('contrast' in self.augmentations):
            if (random.random() < self.augmentations['contrast']['prob']):
                image = self.__changeContrast(image, self.augmentations['contrast']['maxPercentage'])

        if ('blur' in self.augmentations):
            if (random.random() < self.augmentations['blur']['prob']):
                maxSigma = width * self.augmentations['blur']['maxSigmaPercentage']
                image = self.__applyBlur(image, maxSigma)

        if ('noise' in self.augmentations):
            if (random.random() < self.augmentations['noise']['prob']):
                image = self.__applyNoise(image, self.augmentations['noise']['std'])

        if (self.appearanceAugmentation):
            image = image.astype(np.float32)

        return image

    def __shiftImage(self, image, maxHorizontalShift, maxVerticalShift):
        xShift = random.randint(-maxHorizontalShift, maxHorizontalShift)
        yShift = random.randint(-maxVerticalShift, maxVerticalShift)

        hShiftedImage = np.zeros_like(image)

        if (xShift > 0):
            hShiftedImage[:, xShift:] = image[:, 0:-xShift]             # shift right
        elif (xShift < 0):
            xShift = -xShift
            hShiftedImage[:, 0:-xShift] = image[:, xShift:]             # shift left
        else:
            hShiftedImage = image

        vShiftedImage = np.zeros_like(image)

        if (yShift > 0):
            vShiftedImage[yShift:, :] = hShiftedImage[0:-yShift, :]     # shift down
        elif (yShift < 0):
            yShift = -yShift
            vShiftedImage[0:-yShift, :] = hShiftedImage[yShift:, :]     # shift up
        else:
            vShiftedImage = hShiftedImage

        return vShiftedImage    # TODO: create copy

    def __rotateImage(self, image, maxAngle):
        angle = random.randint(-maxAngle, maxAngle)

        height, width, _ = image.shape

        rotationMatrix = cv2.getRotationMatrix2D(center=(width/2, height/2), angle=angle, scale=1)
        rotatedImage = cv2.warpAffine(image, rotationMatrix, dsize=(width, height))

        return rotatedImage

    def __flipImage(self, image):
        flippedImage = np.fliplr(image)

        return flippedImage

    def __changeBrightness(self, image, maxMagnitude):
        return self.__changeHSV(image, maxMagnitude, channelIndex=2)

    def __changeHue(self, image, maxMagnitude):
        return self.__changeHSV(image, maxMagnitude, channelIndex=0)

    def __changeSaturation(self, image, maxMagnitude):
        return self.__changeHSV(image, maxMagnitude, channelIndex=1)

    # copied (generalized by channelIndex)
    def __changeHSV(self, image, maxMagnitude, channelIndex):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        magnitude = random.randint(-maxMagnitude, maxMagnitude)

        v = image[:, :, channelIndex]
        if (magnitude > 0):
            v = np.where(v <= 255-magnitude, v+magnitude, 255)
        else:
            v = np.where(v >= 0-magnitude, v+magnitude, 0)
        image[:, :, channelIndex] = v

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return image

    def __changeContrast(self, image, maxPercentage):
        percentage = random.randint(100-(maxPercentage*100), 100+(maxPercentage*100)) / 100

        image = cv2.addWeighted(image, percentage, image, 0, 0)

        return image

    # copied
    def __applyBlur(self, image, maxSigma):
        sigma = np.random.uniform(0.0, maxSigma)
        image = cv2.GaussianBlur(image, (0, 0), sigma)

        return image

    # copied (added image copy)
    def __applyNoise(self, image, std):
        image = image.copy()

        noise = cv2.randn(np.ndarray(shape=image.shape, dtype=image.dtype), mean=(0, 0, 0), stddev=(std, std, std))
        image += np.minimum(255-image, noise)               # add without overflow

        return image

    def wasFlipped(self):
        return self.lastFlipped
