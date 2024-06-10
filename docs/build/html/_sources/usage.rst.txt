Usage
=====

Installation
------------

To use MIMIM, first install it using pip:

.. code-block:: console

  $ pip install multi-med-image-ml

Data
----

Imaging data is required to get started with this. It was designed and tested with brain MRI/PET/CT data, though any 3D data is applicable. The simplest application is two folders of NIFTI images.

.. code-block:: console

  $ from multi_input_med_image_loader.MedImageLoader import *
  $ folder1 = '/path/to/data1'
  $ folder2 = '/path/to/data2'
  $ dataloader = MedImageLoader(folder1,folder2)
  $ for image,label in MedImageLoader:
  $   ...

Model
-----

.. code-block:: console

  $ 
  $ 
  $ 
