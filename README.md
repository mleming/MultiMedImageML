![ze logo](.images/logo.png)

Multi-Input Medical Image Machine Learning Toolkit
==================================================

The Multi-Input Medical Image Machine Learning Toolkit (MultiMedImageML) is a library of Pytorch functions that can encode multiple 3D images (designed specifically for brain images) and offer a single- or multi-label output, such as a disease detection.

Thus, with a dataset of brain images and labels, you can train a model to predict dementia or multiple sclerosis from multiple input brain images.

To install Multi Med Image ML, simply type into a standard UNIX terminal

    pip install multi-med-image-ml


Overview
========

![ze figure](.images/model_diagram.png)

The core deep learning architecture is a Pytorch model that can take in variable numbers of 3D images (between one and 14 by default), then encodes them into a numerical vector and, through an adversarial training process, creates an intermediate representation that contains information about disease biomarkers but not confounds, like patient age and scanning site.

![ze regress figure](.images/regress_figure.png)

The confound regression process essentially disguises the intermediary representation to have disease biomarker features while imitating the confounding features of other groups.

Getting Started
===============

For best performance, download [https://github.com/rordenlab/dcm2niix](dcm2niix). This is the best program I've found for generally converting DICOMs into NIFTIs, but it's a UNIX command line program, not Python, so it needs to be installed separately.

Datasets
========

This may be used with either public benchmark datasets of brain images or internal hospital records, so long as they're represented as DICOM or NIFTI images. If they're represented as DICOM images, they are converted to NIFTI with metadata represented as a JSON file using [https://github.com/rordenlab/dcm2niix](dcm2niix). They may be further converted to NPY files, which are resized to a specific dimension, with the metadata represented in a pandas dataframe.

The MedImageLoader builds up this representation automatically, but it is space-intensive to do so.

Data may be represented with a folder structure.

# Dataset
```
.
 * [Samples](./Samples)
   * [control](./Samples/control)
     * [patient1](./Samples/control/patient1)
     * [patient2](./Samples/control/patient2)
   * [test](./Samples/test)
     * [patient3](./Samples/test/patient3)
     * [patient4](./Samples/test/patient4)
```

