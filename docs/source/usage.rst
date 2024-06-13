Usage
=====

Installation
------------

To use MIMIM, first install it using pip:

.. code-block:: console

  $ pip install multi-med-image-ml

Dataloader
----------

Imaging data is required to get started with this. It was designed and tested with brain MRI/PET/CT data, though any 3D data is applicable. The simplest application is two folders of NIFTI images.

.. code-block:: console

  $ from multi_input_med_image_loader.MedImageLoader import *
  $ folder1 = '/path/to/data1'
  $ folder2 = '/path/to/data2'
  $ dataloader = MedImageLoader(folder1,folder2)
  $ for image,label in MedImageLoader:
  $   ...

Sample datasets of brain images may be downloaded from sources like [OpenNeuro](https://openneuro.org/).

MedImageLoader by default builds up a database of all images accessed, as well as their metadata. This may be accessed in the ./pandas/ subdirectory.

By default, images are resized to 96x96x96. This may also be changed by specifying the X_dim parameter in the dataloader. Resized images are cached as .npy files.

Model and Training
------------------

.. code-block:: console

The simplest way to train the multi-input module, as other pytorch models are trained, is as follows:

  $ from multi_input_med_image_loader.models import *
  $ from src.multi_med_image_ml.MedImageLoader import *
  $
  $ dataloader = MedImageLoader(folder1,folder2)
  $ model = MultiInputModule()
  $
  $ optimizer = torch.optim.Adam(
  $   model.classifier_parameters(),
  $   betas = (0.5,0.999),
  $   lr= 1e-5
  $ )
  $
  $ loss_function = nn.MSELoss()
  $
  $ for image,label in dataloader:
  $   optimizer.zero_grad()
  $   y_pred,_ = model(image)
  $   loss = loss_function(label,y_pred)
  $   loss.backward()
	$		optimizer.step()
  $

  The MultiInputTrainer module allows for the confound regression functionalities and generally abstracts that process.

  $ from multi_input_med_image_loader.models import *
  $ from src.multi_med_image_ml.MedImageLoader import *
  $ model = MultiInputModule(Y_dim = (17,4),C_dim=(13,11))
  $
  $ dataloader = MedImageLoader(imfolder1,imfolder2,
  $   cache=True,
  $   label=["MRAcquisitionType"],
  $   confounds=["Slice Thickness","Repetition Time"],
  $   return_obj = True,
  $   batch_by_pid=True)
  $
  $ trainer = MultiInputTrainer(model,batch_size=2)
  $ for i in range(3):
  $   for p in dataloader:
  $     trainer.loop(p,dataloader=medim_loader)

  Testing
  -------

  TODO
