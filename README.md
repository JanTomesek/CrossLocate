# CrossLocate:</br>Cross-modal Large-scale Visual Geo-Localization in Natural Environments using Rendered Modalities

![alt text](http://cphoto.fit.vutbr.cz/crosslocate/data/images/CrossLocate_overview.png)

## About this repository

This is the official repository for the WACV 2022 paper [CrossLocate](https://ieeexplore.ieee.org/document/9706931).
The work solves the task of determining the geographic origin of photographs captured in natural areas (e.g. mountains).

**Authors**
- [Jan Tomešek](https://www.linkedin.com/in/jan-tome%C5%A1ek-95bb26149/) (itomesek@fit.vutbr.cz)
- [Martin Čadík](http://cadik.posvete.cz/) (cadik@fit.vutbr.cz)
- Jan Brejcha

CPhoto@FIT, Brno University of Technology, Czechia

If you use the code or datasets in a scientific work, please, cite it:

```
J. Tomešek, M. Čadík and J. Brejcha, 
"CrossLocate: Cross-modal Large-scale Visual Geo-Localization in Natural Environments using Rendered Modalities," 
2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 2193-2202, 
doi: 10.1109/WACV51458.2022.00225.
```

```
@INPROCEEDINGS{CrossLocate,  
author={Tomešek, Jan and Čadík, Martin and Brejcha, Jan},  
booktitle={2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},   
title={CrossLocate: Cross-modal Large-scale Visual Geo-Localization in Natural Environments using Rendered Modalities},   
year={2022},  
pages={2193-2202},  
doi={10.1109/WACV51458.2022.00225}}
```

## Useful links
- [CrossLocate project website](http://cphoto.fit.vutbr.cz/crosslocate/) (includes short video presentation, datasets, meta structures and additional materials)
- [Paper (pdf)](http://cphoto.fit.vutbr.cz/crosslocate/data/paper/CrossLocate.pdf)
- [Supplementary Material (pdf)](http://cphoto.fit.vutbr.cz/crosslocate/data/supplement/CrossLocate-supp.pdf)

## Project overview

The project solution requires several components to run
- code (*available in this repository*)
- datasets (*available on the project website*)
- meta structures (containing meta information for the data) (*available on the project website*)
- (pre)trained models (*optional*) (*available in this repository and on the project website*)

The expected directory structure, beginning with the root of this repository, is
```
CrossLocate/
    CrossLocate/
        <code>
    datasets/
        <datasets>
    datasets_meta/
        <meta structures>
```

## Code

The implementation is done in **Python** and **TensorFlow** with several Python dependencies.

The implementation was tested with
- Python 3.6
- TensorFlow 1.14.0
- NumPy 1.16.0
- OpenCV 4.4.0
- Imageio 2.5.0
- Scikit-learn 0.24.1

Training (and related evaluation) can be run with the script `train.py`.

Parameters of the training and evaluation procedure can be set directly in the script `train.py`.
```
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

augmentParams = { 'augmentation': True }

evalParams = {	'recallNumbers' : [1, 10, 100],
                'locationThresholds': [20, 100, 1000],
                'locationThresholdStep': 20,
                'successfulLocalizationThreshold': 20 }

plotParams = { 'distanceRecallsXTicks': [0, 20, 100, 250, 500, 750, 1000] }

visualParams = {  'batchVisualEpochs': [trainParams['startEpoch'], trainParams['endEpoch']],
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
```

By default, the solution searches for an available GPU with
```
Environment.seizeGPU()
```

Pre-computation of geographic distances within datasets takes some time when precise WGS coordinate system is used
```
trainDataset.precomputeLocationDistancesWGS()
valDataset.precomputeLocationDistancesWGS()
```
This can be speeded up by using less precise UTM coordinate system
```
trainDataset.precomputeLocationDistancesUTM()
valDataset.precomputeLocationDistancesUTM()
```

## Datasets

CrossLocate provides two unique datasets, each consisting of various image modalities.

### Sparse dataset

Designed for small, fast and simple experimentation.

#### Sparse database
- 37 332 synthetically rendered views (3 111 locations, 12 views per location)
  - 3 different image modalities (semantic segmentations, silhouette maps, depth maps)

#### GeoPose3K queries
- 3 111 photographs (3 111 locations)
- 3 111 synthetically rendered views (3 111 locations)
  - 3 different image modalities (semantic segmentations, silhouette maps, depth maps)

### Uniform dataset

Designed for the real-world task of localization across a large natural area - the Alps.

#### Uniform database
- 10.72M synthetically rendered views (1M locations, 12 views per location)
  - 3 different image modalities (semantic segmentations, silhouette maps, depth maps)

#### CrossLocate queries
- 12 353 photographs (12 353 locations)

The expected directory structure (with a specific example) is
```
datasets/
    Uniform/
        database_segments/
        query_photos_to_segments_result/
        ...
    Sparse/
        database_depth/
        query_original_result/
        ...
```

The path to the root directory of datasets can be set in `train.py` with variable `datasetsPath`. By default
```
datasetsPath = '../datasets'
```

## Meta structures

Meta information for the data (latitude, longitude, field of view, ...) is available in the form of "meta structures".

There are three meta structures (for training, validation and testing sets respectively) for a given combination of a database and queries (e.g. semantic segmentations from Uniform database and photographs from CrossLocate queries).

The meta structures are loaded by the `Dataset.py` class.

The expected directory structure (with a specific example) is
```
datasets_meta/
    Uniform/
        Alps_photos_to_segments_train.npy
        Alps_photos_to_segments_val.npy
        Alps_photos_to_segments_test.npy
        ...
    Sparse/
        GeoPose3K_v2_original_to_depth_train.npy
        GeoPose3K_v2_original_to_depth_val.npy
        GeoPose3K_v2_original_to_depth_test.npy
        ...
```

The path to the root directory of meta structures can be passed as a parameter to the constructor of `Dataset.py`. By default
```
datasetsMetaPath='../datasets_meta'
```

Specific meta structures (and thereby specific datasets) to be used in an experiment are set in `train.py` with variable `datasetParams`
```
datasetParams = datasetSegments
```
The applicable values for this variable are prepared in `DatasetDescriptions.py`

## (Pre)trained models

The solution can be initialized with available (pre)trained deep models (e.g. a model pre-trained on the ImageNet dataset or a model resulting from the CrossLocate training procedure).

The pre-trained models are stored in `CrossLocate/CrossLocate/models/`.

