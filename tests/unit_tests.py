#!/usr/bin/python
import unittest
import os,sys,time

import os,sys,time
import numpy as np
import torch
import shutil

wd         = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
im_root    = os.path.join(os.path.dirname(wd),'data','ADNI_sample')
imfolder1 = os.path.join(im_root,'002')
imfolder2 = os.path.join(im_root,'941')
im1 = os.path.join(imfolder1,'002_S_0295','MP-RAGE','2006-04-18_08_20_30.0','I13722')
nifti_im = os.path.join(im1,'I13722_ADNI_12M4_TS_2_20060418081744_3.nii.gz')
npy_im = os.path.join(im1,'I13722_ADNI_12M4_TS_2_20060418081744_3_resized_4_5_6.npy')
#'/Users/mleming/Desktop/MultiMedImageML/data/ADNI_sample/002/002_S_0295/MP-RAGE/2006-04-18_08_20_30.0/I13722/I13722_ADNI_12M4_TS_2_20060418081744_3_resized_4_5_6.npy'
#imfile_svs = os.path.join(im_root,'10447627_1.svs')
pandas_file = 'test_folder/pandas/database_48_32_24.pkl'
sys.path.insert(0,wd)
sys.path.insert(0,os.path.join(wd,'src'))
sys.path.insert(0,os.path.join(wd,'src','multi_med_image_ml'))

from src.multi_med_image_ml.MedImageLoader import *
from src.multi_med_image_ml.models import *
from src.multi_med_image_ml.MultiInputTrainer import *
from src.multi_med_image_ml.MultiInputTester import * 

def get_cache_file_list():
	filepaths = []
	for root, dirs, files in os.walk(im_root, topdown=False):
		for name in files:
			filepath = os.path.join(root,name)
			ext = os.path.splitext(filepath)[1]
			if ext in [".gz",".nii",".npy",".json"]:
				filepaths.append(filepath)
	return filepaths

def clear_files():
	for filename in get_cache_file_list():
		os.remove(filename)
	pandas_dir = 'test_folder/pandas'
	if os.path.isdir(pandas_dir):
		for root, dirs, files in os.walk(pandas_dir, topdown=False):
			for name in files:
				filepath = os.path.join(root,name)
				ext = os.path.splitext(filepath)[1]
				if ext == ".pkl":
					os.remove(filepath)

class TestSimple(unittest.TestCase):
	def test_dicom_compile(self):
		nifti_file,json_file = compile_dicom(im1)
		self.assertTrue(os.path.isfile(nifti_file))
		self.assertEqual(os.path.splitext(nifti_file)[1], ".gz")
		self.assertTrue(os.path.isfile(json_file))
		self.assertEqual(os.path.splitext(json_file)[1], ".json")
		len_glob = len(glob.glob(os.path.join(im1,'*')))
		nifti_file,json_file = compile_dicom(im1,cache=True)
		self.assertEqual(len_glob, len(glob.glob(os.path.join(im1,'*'))))
		nifti_file2,json_file2 = compile_dicom(im1,cache=False)
		self.assertTrue(nifti_file2 != nifti_file)
		self.assertTrue(json_file2 != json_file)
		if os.path.splitext(json_file)[1] == ".json":
			os.remove(json_file)
		if os.path.splitext(nifti_file)[1] == ".gz":
			os.remove(nifti_file)
		if os.path.splitext(json_file2)[1] == ".json":
			os.remove(json_file2)
		if os.path.splitext(nifti_file2)[1] == ".gz":
			os.remove(nifti_file2)
	def test_single_im_load(self):
		im = ImageRecord(im1,X_dim=(24,48,32),cache=False)
		img = im.get_X()
		self.assertEqual(img.shape[0], 24)
		self.assertEqual(img.shape[1], 48)
		self.assertEqual(img.shape[2], 32)
	def test_cache(self):
		im = ImageRecord(im1,X_dim=(33,16,3),cache=True)
		img = im.get_X()
		self.assertEqual(img.shape[0], 33)
		self.assertEqual(img.shape[1], 16)
		self.assertEqual(img.shape[2], 3)
		self.assertTrue(im.cached_record is not None)
		self.assertTrue(os.path.isfile(im.cached_record))
		self.assertEqual(os.path.splitext(im.cached_record)[1], ".npy")
		if os.path.isfile(im.cached_record):
			os.remove(im.cached_record)
	def test_single_nifti_load(self):
		im = ImageRecord(nifti_im,X_dim=(4,5,6),cache=True)
		img = im.get_X()
		self.assertEqual(img.shape[0], 4)
		self.assertEqual(img.shape[1], 5)
		self.assertEqual(img.shape[2], 6)
	def test_npy_load(self):
		#self.assertTrue(os.path.isfile(npy_im))
		im = ImageRecord(npy_im,X_dim=(4,5,6))
		img = im.get_X()
		self.assertEqual(img.shape[0], 4)
		self.assertEqual(img.shape[1], 5)
		self.assertEqual(img.shape[2], 6)	
	def test_single_nifti_load_torch(self):
		im = ImageRecord(nifti_im,X_dim=(4,5,6),cache=False,
			dtype='torch')
		img = im.get_X()
		self.assertTrue(torch.is_tensor(img))
		self.assertEqual(img.size()[0], 4)
		self.assertEqual(img.size()[1], 5)
		self.assertEqual(img.size()[2], 6)
	def test_basic_load_torch(self):
		medim_loader = MedImageLoader(
					imfolder1,
					imfolder2,
					X_dim=(48,32,24),
					dtype="torch",
					batch_size=16,
					cache = False,
					channels_first=False)
		for image,label in medim_loader:
			self.assertTrue(torch.is_tensor(image))
			self.assertTrue(torch.is_tensor(label))
			imsize = image.size()
			self.assertEqual(imsize[0], 16)
			self.assertEqual(imsize[1], 48)
			self.assertEqual(imsize[2], 32)
			self.assertEqual(imsize[3], 24)
	def test_basic_load_numpy(self):
		medim_loader = MedImageLoader(
					imfolder1,
					imfolder2,
					X_dim=(48,32,24),
					dtype="numpy",
					batch_size=16,
					cache = False,
					channels_first=False)
		for image,label in medim_loader:
			self.assertTrue(isinstance(image,np.ndarray))
			self.assertTrue(isinstance(label,np.ndarray))
			imsize = image.shape
			self.assertEqual(imsize[0], 16)
			self.assertEqual(imsize[1], 48)
			self.assertEqual(imsize[2], 32)
			self.assertEqual(imsize[3], 24)
	def test_pandas(self):
		medim_loader = MedImageLoader(
					imfolder1,
					imfolder2,
					X_dim = (48,32,24),
					dtype = "numpy",
					cache = True,
					channels_first=False)
		for image,label in medim_loader:
			self.assertTrue(medim_loader.cache)
			self.assertTrue(isinstance(image,np.ndarray))
			self.assertTrue(isinstance(label,np.ndarray))
			imsize = image.shape
			self.assertEqual(imsize[0], 14)
			self.assertEqual(imsize[1], 48)
			self.assertEqual(imsize[2], 32)
			self.assertEqual(imsize[3], 24)
		pandas_file = medim_loader.database_file
		self.assertTrue(os.path.isfile(pandas_file))
		df = pd.read_pickle(pandas_file)
		self.assertTrue(len(df) > 10)
		for filename in df.index:
			if not os.path.isfile(filename):
				print("ABOUT2FAIL: %s not existy" % filename)
			self.assertTrue(os.path.isfile(filename))
		rstr = "48_32_24_resized.npy"
		for root, dirs, files in os.walk(im_root):
			for name in files:
				fpath = os.path.join(root,name)
				if rstr in fpath:
					self.assertTrue(fpath in df.index)
		
	def test_pandas_2(self):
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			X_dim=(48,32,24),
			cache=True,
			dtype="numpy",
			channels_first=False)
		for image,label in medim_loader: continue
		pandas_file = medim_loader.database_file
		self.assertTrue(os.path.isfile(pandas_file))
		medim_loader = MedImageLoader(pandas_file,
			X_dim=(48,32,24),dtype="numpy",channels_first=False)
		self.assertEqual(medim_loader.mode,"iterate")
		for image in medim_loader:
			imsize = image.shape
			self.assertEqual(len(imsize),5)
			self.assertEqual(imsize[0], 14)
			self.assertEqual(imsize[1], 48)
			self.assertEqual(imsize[2], 32)
			self.assertEqual(imsize[3], 24)
	
	def test_match_label_confounds(self):
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			X_dim=(48,32,24),
			cache=True,
			dtype="numpy",
			channels_first=False)
		for image,label in medim_loader: continue 
		pandas_file = medim_loader.database_file
		self.assertTrue(os.path.isfile(pandas_file))
		medim_loader = MedImageLoader(pandas_file,
			X_dim=(48,32,24),
			dtype="numpy",
			label=["MRAcquisitionType"],
			confounds=["PercentSampling"],
			channels_first=False,
			batch_by_pid=False)
		self.assertEqual(medim_loader.mode,"match")
		for image,label in medim_loader:
			imsize = image.shape
			self.assertEqual(len(imsize),5)
			self.assertEqual(imsize[0], 14)
			self.assertEqual(imsize[1], 48)
			self.assertEqual(imsize[2], 32)
			self.assertEqual(imsize[3], 24)
	def test_grouping(self):
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			X_dim=(48,32,24),
			cache=True,
			dtype="torch",
			channels_first=False,
			batch_by_pid=False)
		pandas_file = medim_loader.database_file
		for image,label in medim_loader:
			imsize = image.size()
			self.assertEqual(len(imsize),5)
			self.assertEqual(imsize[1], 48)
			self.assertEqual(imsize[2], 32)
			self.assertEqual(imsize[3], 24)
		medim_loader = MedImageLoader(pandas_file,
			X_dim=(48,32,24),dtype="torch",
			label=["MRAcquisitionType"],
			confounds=["PercentSampling"],
			return_obj=True,
			channels_first=False)
		for patient in medim_loader:
			image = patient.get_X()
			imsize = image.size()
			self.assertEqual(imsize[1], 48)
			self.assertEqual(imsize[2], 32)
			self.assertEqual(imsize[3], 24)
		C = patient.get_C()
		C_dud = patient.get_C_dud()
		Y = patient.get_Y()
	def test_more(self):
		medim_loader = MedImageLoader(imfolder1)
		for image in medim_loader:
			imsize = image.shape
		medim_loader = MedImageLoader(imfolder2,augment=True)
		for image in medim_loader:
			continue
	def test_model(self):
		model = MultiInputModule(Y_dim=(32,32),C_dim=(32,32))
		#medim_loader = MedImageLoader(imfolder1,imfolder2,
		#	cache=True)
		#for image,label in medim_loader: continue
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			return_obj=True,
			cache=True,dtype="torch",
			Y_dim = (32,32),
			C_dim = (32,32),batch_by_pid=True)
		optimizer = torch.optim.Adam(
			model.classifier_parameters(),
			betas = (0.5,0.999),
			lr= 1e-5
		)
		loss_function = nn.MSELoss()
		
		for p in medim_loader:
			optimizer.zero_grad()
			y_pred,y_reg = model(p)
			loss = loss_function(p.get_Y(),y_pred)
			loss.backward()
			optimizer.step()
			break
	def test_trainer(self):
		model = MultiInputModule(Y_dim = (17,4),C_dim=(13,11))
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			cache=True,
			label=["MRAcquisitionType",
					"ImageOrientationPatientDICOM"],
			confounds=["Slice Thickness","Repetition Time"],
			return_obj = True,
			dtype="torch",
			batch_size=14,Y_dim=(17,4),C_dim=(13,11),
			batch_by_pid=True)
		trainer = MultiInputTrainer(model,batch_size=2)
		for i in range(3):
			#print(f"Epoch {i}")
			for p in medim_loader:
				trainer.loop(p,dataloader=medim_loader)
		
	def test_cache2(self):
		model = MultiInputModule(Y_dim = (13,14),C_dim=(19,21))
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			cache=False,
			label=["MRAcquisitionType",
					"Manufacturer"],
			confounds=["Slice Thickness","Repetition Time"],
			return_obj = True,
			dtype="torch",
			Y_dim = (13,14),
			C_dim=(19,21),
			batch_by_pid=True)
		trainer = MultiInputTrainer(model,batch_size=2)
		for i in range(3):
			#print(f"Epoch {i}")
			for p in medim_loader:
				trainer.loop(p,dataloader=medim_loader)	

	def test_dyn_input(self):
		model = MultiInputModule(Y_dim = (13,14),C_dim=(19,21))
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			cache=False,
			label=["MRAcquisitionType",
					"Manufacturer"],
			confounds=["Slice Thickness","Repetition Time"],
			return_obj = True,
			dtype="torch",
			batch_size=14,Y_dim = (13,14),C_dim=(19,21),
			static_inputs = ["User Data 7","Slice Thickness"],
			batch_by_pid=True)
		trainer = MultiInputTrainer(model,batch_size=2)
		for i in range(3):
			for p in medim_loader:
				trainer.loop(p,dataloader=medim_loader)

	def test_cache3(self):
		model = MultiInputModule((11,17),C_dim=(23,5))
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			cache=True,
			label=["MRAcquisitionType",
					"Manufacturer"],
			confounds=["Slice Thickness","Repetition Time"],
			return_obj = True,
			dtype="torch",
			batch_size=14,Y_dim=(11,17),C_dim=(23,5),
			batch_by_pid=True)
		trainer = MultiInputTrainer(model,
					batch_size=2,
					loss_image_dir = 'test_folder/loss_images',
					checkpoint_dir = 'test_folder/checkpoints',
					name = 'unit_test',
					verbose = False,
					save_latest_freq = 1)
		for i in range(3):
			for p in medim_loader:
				trainer.loop(p,dataloader=medim_loader)	
	def test_weight_download(self):
		model = MultiInputModule(weights="unit_test")
	def test_init(self):
		model = MultiInputModule((1,17),C_dim=(23,5))
		medim_loader = MedImageLoader(imfolder1,imfolder2,
			cache=True,
			label=["MRAcquisitionType"],
			confounds=["Slice Thickness","Repetition Time"],
			return_obj = True,
			dtype="torch",
			batch_size=14,Y_dim=(1,17),C_dim=(23,5),
			batch_by_pid=True)
		tester = MultiInputTester(medim_loader.database,model,
			out_record_folder='test_folder',
			checkpoint_dir='test_folder/checkpoints',
			database_key="Manufacturer")
		for pr in medim_loader:
			tester.loop(pr)
		tester.stats_record.record()
		tester.read_json()
		tester.plot()
		tester.test_grad_cam(pr)
if __name__ == "__main__":
	#clear_files()
	# Runs the tests twice, once with cached files and once without
	#unittest.main()
	unittest.main()
	
	
