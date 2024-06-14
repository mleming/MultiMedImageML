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

.. code-block:: python

	from multi_med_image_ml.MedImageLoader import *
	folder1 = '/path/to/data1'
	folder2 = '/path/to/data2'
	dataloader = MedImageLoader(folder1,folder2)
	for image,label in dataloader:
		...
	

Sample `datasets <https://openneuro.org/datasets/ds005216/versions/1.0.1/download>`_ of brain images may be downloaded from sources like `OpenNeuro <https://openneuro.org/>`_.

Data may also be encapsulated in the BatchRecord class, which is recommended for very large datasets.

.. code-block:: python

	dataloader = MedImageLoader(folder1,folder2,return_obj=True)
	for b in dataloader:
		print(b.get_X()) # image
		print(b.get_Y()) # label


They may also be batched by the patient:

.. code-block:: python

	dataloader = MedImageLoader(folder1,
				folder2,
				return_obj=True,
				group_by_pid=True)
	for b in dataloader:
		...


MedImageLoader may also take in a pandas dataframe containing references to each cached image with the associated metadata:

.. code-block:: python

	pandas_path = '/path/to/dataframe.pkl'
	dataloader = MedImageLoader(pandas_path)


By default, it builds up this dataframe the first time it reads through a folder. The dataframe contains indices that are paths to image files and columns associated with metadata. To read in different variales from this dataframe, you may specify the labels as an argument:

.. code-block:: python

	pandas_path = '/path/to/dataframe.pkl'
	dataloader = MedImageLoader(pandas_path,
			label=["MRAcquisitionType"],
			return_obj=True)
	
	for p in dataloader:
		p.get_X() # Image
		p.get_Y() # Encoding of MRAcquisitionType



MedImageLoader by default builds up a database of all images accessed, as well as their metadata. This may be accessed in the designates directory.

By default, images are resized to 96x96x96. This may also be changed by specifying the X_dim parameter in the dataloader. Resized images are cached as .npy files.

Model and Training
------------------

The simplest way to train the multi-input module, as other pytorch models are trained, is as follows:

.. code-block:: python

	from multi_med_image_ml.models import *
	from multi_med_image_ml.MedImageLoader import *
	import torch
	
	dataloader = MedImageLoader(folder1,folder2)
	model = MultiInputModule()
	
	optimizer = torch.optim.Adam(
		model.classifier_parameters(),
		betas = (0.5,0.999),
		lr= 1e-5
	)
	loss_function = torch.nn.MSELoss()
	
	for image,label in dataloader:
		optimizer.zero_grad()
		y_pred,_ = model(image)
		loss = loss_function(label,y_pred)
		loss.backward()
		optimizer.step()


The `MultiInputTrainer`_ module allows for the confound regression functionalities and generally abstracts that process.

.. code-block:: python

	from multi_med_image_ml.models import *
	from multi_med_image_ml.MedImageLoader import *
	from multi_med_image_ml.MultiInputTrainer import *
	model = MultiInputModule()
	
	dataloader = MedImageLoader(imfolder1,imfolder2,
		cache=True,
		label=["MRAcquisitionType"],
		confounds=["Slice Thickness","Repetition Time"],
		return_obj = True,
		batch_by_pid = True
	)
	
	trainer = MultiInputTrainer(model)
	for i in range(3):
		for p in dataloader:
			trainer.loop(p,dataloader=medim_loader)


Testing
-------

`MultiInputTester`_ is a more complex module that allows a variety of tests to be performed on the ML model. One is model performance:

.. code-block:: python

	
	from multi_med_image_ml.models import *
	from multi_med_image_ml.MedImageLoader import *
	from multi_med_image_ml.MultiInputTester import *
	
	model = MultiInputModule()
	
	dataloader = MedImageLoader(imfolder1,imfolder2,
		cache=True,
		label=["MRAcquisitionType"],
		confounds=["Slice Thickness","Repetition Time"],
		return_obj = True,
		batch_by_pid = True
	)
	
	tester = MultiInputTester(model,dataloader.database)
	
	tester.grad_cam()
	
	for p in dataloader:
		tester.loop(p)
	
	
