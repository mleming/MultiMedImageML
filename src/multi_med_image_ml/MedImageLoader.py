#!/usr/bin/python3
import os,sys
import torchvision.transforms as transforms
from pdb import set_trace as st
import random
import numpy as np
import torch
import pandas as pd
import torch.multiprocessing
from .utils import *
torch.multiprocessing.set_sharing_strategy('file_system')
import psutil,shutil
import nibabel as nb
from nibabel.filebasedimages import *
from .Records import BatchRecord,ImageRecord,AllRecords
from .DataBaseWrapper import DataBaseWrapper

# Translates a filename to a key and back, for storing files as keys in the
# pandas dataframe. By default, the keys are the full filepaths. This function
# may need to be changed when switching to different systems
def key_to_filename_default(fkey,reverse=False):
	return fkey

class MedImageLoader:
	"""Loads medical images into a format that may be used by MultiInputModule.
	
	This loader preprocesses, reshapes, augments, and batches images and 
	metadata into a format that may be read by MultiInputModule. It additionally
	may apply a data matching algorithm to ensure that no overly confounded
	data is fed into the model during training. It is capable of maintaining
	different lists of images to balance classes for both the classifier and
	regressor.
	
	Attributes:
		database (DataBaseWrapper): Object used to store and access metadata about particular files. MedImageLoader builds this automatically from a folder, or it can read from one directly if it's already been built (default None)
		X_dim (tuple): Three-tuple dimension to which the images will be resized to upon output (default (96,96,96))
		Y_dim (tuple): A tuple indicating the dimension of the image's label. The first number is the number of labels associated with the image and the second is the number of choices that has. Extra choices will not affect the model but fewer will throw an error — thus, if Y_dim is (1,2) and the label has three classes, it will crash. But (1,4) will just result in an output that is always zero. This should match the Y_dim parameter in the associated MultiInputModule (default (1,32))
		C_dim (tuple): A tuple indicating the dimension of the image's confounds. This effectively operates the same way as Y_dim, except the default number of confounds is higher (default (16,32))
		batch_size (int): Max number of images that can be read in per batch. Note that if batch_by_pid is True, this is the maximum number of images that can be read in, and it's best to set it to the same value as n_dyn_inputs in MultiInputModule (default 14)
		augment (bool): Whether to augment images during training. Note that this only works if the images are returned directly (i.e. return_obj = False). Otherwise images are augmented when get_X is called from ImageRecord (default True)
		dtype (str): Type of image to be returned -- either "torch" or "numpy" (default "torch")
		label (list): List of labels that will be read in from DataBaseWrapper to the Y output. Must be smaller than the first value of Y_dim.
		confounds (list): List of confounds that will be read in from DataBaseWrapper to the C output. Must be smaller than the first value of C_dim.
		pandas_cache (str): Directory in which the database pandas file is stored
		cache (str): Whether to cache images of a particular dimension as .npy files, for faster reading and indexing in the database (default True)
		key_to_filename (callback): Function that translates a key to the DataBaseWrapper into a full filepath from which an image may be read. Needs to accept an additional parameter to reverse this as well (default key_to_filename_default)
		batch_by_pid (bool): Whether to batch images together by their Patient ID in a BatchRecord or not (default False)
		file_record_name (str): Path of the record of files that were read in by the MedImageLoader, if it needs to be examined later (default None)
		channels_first (bool): Whether to put channels in the first or last dimension of images (default True)
		save_ram (bool): Clears images from ImageRecords and applies garbage collection frequently to save RAM. Useful for very large datasets (default True)
		static_inputs (list): List of variables from DataBaseWrapper that will be input as static, per-patient text inputs (like Sex of Ethnicity) to the MultiInputModule (default None)
		val_ranges (dict): Dictionary that may be used to indicate ranges of values that may be loaded in. So, if you want to only study males, val_ranges could be {'SexDSC':'MALE'}, and of you only wanted to study people between ages 30 and 60, val_ranges could be {'Ages':(30,60)}; these can be combined, too. Note that 'Ages' and 'SexDSC' must be present in DataBaseWrapper as metadata variable names for this to work (default {})
		match_confounds (list): Used to apply data matching between the labels. So, if you wanted to distinguish between AD and Controls and wanted to match by age, match_confounds could be set to ['Ages'] and this would only return sets of AD and Control of the same age ranges. Note that this may severely limit the dataset or even return nothing if the match_confound variable and the label variable are mutually exclusive (default [])
		all_records (multi_med_image_loader.Records.AllRecords): Cache to store ImageRecords in and clear them if images in main memory get too high.
		n_dyn_inputs (int): Max number of inputs of the ML model, to be passed into BatchRecord when it's used as a patient record (default 14)
		precedence (list): Because labeling is by image in the database and diagnosis is by patient, this option allows "precedence" in labeling when assigning an overall label to a patient. So, if a patient has three images, two marked as "Healthy" and one marked as "Alzheimer's", you can pass "[Alzheimer's,Healthy]" into precedence and it would assign the whole patient the "Alzheimer's" label (default [])
	"""
	
	def __init__(self,*image_folders,
			pandas_cache = './pandas/',
			cache = True,
			key_to_filename = key_to_filename_default,
			batch_by_pid=False,
			file_record_name=None,
			database = None,
			batch_size = 14,
			X_dim = (96,96,96),
			get_encoded = False,
			static_inputs = None,
			confounds = [],
			match_confounds = [],
			label = [],
			augment = True,
			val_ranges = {},
			dtype="torch",
			Y_dim = (1,32),
			C_dim = (16,32),
			return_obj = False,
			channels_first = True,
			gpu_ids = "",
			save_ram = True,
			precedence = [],
			n_dyn_inputs=14,
			verbose = False):
		self.channels_first = channels_first
		self.image_folders = image_folders
		self.augment = augment
		self.X_dim=X_dim
		self.dtype=dtype
		self.cache = cache
		self.pandas_cache = pandas_cache
		self.val_ranges = val_ranges
		self.get_encoded = get_encoded
		self.batch_by_pid = batch_by_pid
		self.file_list_dict = {}
		self.static_inputs = static_inputs
		self.label = label
		self.confounds = confounds
		self.Y_dim = Y_dim
		self.C_dim = C_dim
		self.mode = None
		self.gpu_ids = gpu_ids
		self.save_ram = save_ram
		self.n_dyn_inputs = n_dyn_inputs
		self.verbose = verbose
		
		if self.verbose:
			print("Checking key to filename conversion function")
		check_key_to_filename(key_to_filename)

		# Stores images so that they aren't repeated in different stacks
		self.all_records = AllRecords()
		# If true, this uses one match confound at a time and cycles through
		# them
		self.zero_data_list = []
		self.match_confounds = match_confounds
		if self.batch_by_pid:
			self.batch_size = 1
		else:
			self.batch_size = batch_size
		self.return_obj = return_obj
		
		# Create or read in the image database
		if self._pickle_input():
			self.database_file = self.image_folders[0]
			#assert(self.cache or os.path.isfile(self.database_file))
		else:
			os.makedirs(os.path.dirname(self.pandas_cache),
				exist_ok=True)
			self.database_file = os.path.join(
				self.pandas_cache,
				"database_%s.pkl" % get_dim_str(X_dim=self.X_dim))
		
		if self.verbose:
			print("Initializing database wrapper %s" % self.database_file)
		
		self.database = DataBaseWrapper(
					filename=self.database_file,
					labels=self.label,
					confounds=self.confounds,
					X_dim=self.X_dim,
					key_to_filename=key_to_filename,
					val_ranges=self.val_ranges,
					precedence = precedence)
		if not self._pickle_input():
			self.build_pandas_database()
		self.database.build_metadata()
		
		# Determine which mode the scheduler is in
		if (self.label is not None and len(self.label) > 0):
			self.mode = "match"
		else:
			self.mode = "iterate"
		
		# For outputting the records of files that were read in
		self.file_record_name = file_record_name
		if self.file_record_name is not None:
			self.file_record = {}
			fd = os.path.join(wd,'json','dataloader_records')
			if not os.path.isdir(fd): os.makedirs(fd)
			self.file_record_output = os.path.join(fd,file_record_name)

		self.uniques = None
		self.n_buckets = 3
		self.rmatch_confounds = self.confounds
	
		# Regular stack variables
		self.file_list_dict = {}
	
		# Switch stack variables
		self.file_list_dict_hidden = {}
		self.match_confounds_hidden = []
		self.mode_hidden = "iterate"
		if self._return_labels():
			for l in self.label:
				if l not in self.val_ranges:
					self.match_confounds_hidden.append(l)
		self.index = 0
		if self.verbose:
			print("Loading image stack")
		self.load_image_stack()
		
	def build_pandas_database(self):
		"""Builds up the entire Pandas DataFrame from the filesystem in one go.
		May take a while."""
		if self.verbose:
			print("Building pandas database")
		assert(self.database is not None)
		assert(not self._pickle_input())
		old_mode = self.mode
		self.mode = "iterate"
		self._load_list_stack()
		if self.verbose:
			print("len(self.all_records.image_dict): %d" % (len(self.all_records.image_dict)))
		
		for i,filename in enumerate(self.all_records.image_dict):
			im = self.all_records.get(filename)
			if not self.database.has_im(im):
				try:
					im.get_X()
					im.clear_image()
				except (ImageFileError,FileNotFoundError) as e:
					if self.verbose:
						print("Failed to load %s" % filename)
					continue
			if i % 100 == 0:
				self.database.out_dataframe()
		self.all_records.clear_images()
		self.database.out_dataframe()
		self.mode = old_mode
	def _pickle_input(self):
		return len(self.image_folders) == 1 and \
			os.path.splitext(self.image_folders[0])[1] == ".pkl"
	def tl(self):
		"""Top label"""
		
		if len(self.label) == 0: tl = "Folder"
		else: tl = self.label[0]
		if tl not in self.file_list_dict:
			self.file_list_dict[tl] = []
		return tl
	
	def get_file_list(self):
		
		
		if self._pickle_input() and self.tl() == "Folder":
			return [[str(_) for _ in self.database.get_file_list()]]
		if self.mode == "iterate":
			if self._pickle_input():
				fname_list = [str(_) for _ in self.database.get_file_list()]
				return self.database.stack_list_by_label(fname_list,self.tl())
			else:
				all_filename_lists = []
				duplicate_test = set()
				for img in self.image_folders:
					flist = get_file_list(img)
					flist_set = set(flist)
					if len(flist_set.intersection(duplicate_test)) > 0:
						raise Exception(
							"Intersecting files found between labels"
							)
					duplicate_test = duplicate_test.union(flist_set)
					all_filename_lists.append(flist)
				assert(len(all_filename_lists) > 0)
				return all_filename_lists
		elif self.mode == "match":
			[fname_list],_ = get_balanced_filename_list(self.tl(),
				self.match_confounds,
				selection_ratios=[1], total_size_limit=np.inf,
				non_confound_value_ranges = self.val_ranges,verbose=False,
				database=self.database.database)
			fname_list = list(fname_list)
			fname_list = [self.database.key_to_filename(_) for _ in fname_list]
			if len(fname_list) == 0:
				print(self.database.database.loc[:,self.tl()])
				raise Exception("No valid files from %s" % self.tl())
			assert(isinstance(fname_list,list))
			return self.database.stack_list_by_label(fname_list,self.tl())
		else:
			raise Exception("Invalid mode: %s" % self.mode)
	def _load_list_stack(self):
		X_files = self.get_file_list()
		if self.verbose:
			print("X_files loaded: %d" % len(X_files))
			print([len(_) for _ in X_files])
		for i,filename_list in enumerate(X_files):
			for j,filename in enumerate(filename_list):
				if self.all_records.has(filename):
					X_files[i][j] = self.all_records.get(filename)
				else:
					X_files[i][j] = ImageRecord(filename,
								X_dim=self.X_dim,
								y_nums=[i] if self.tl() == "Folder" else None,
								Y_dim = self.Y_dim,
								C_dim = self.C_dim,
								dtype=self.dtype,
								database = self.database,
								cache=self.cache,
								static_inputs = self.static_inputs)
					self.all_records.add(filename, X_files[i][j])
		if self.verbose:
			print("Loading list stack")
			print("X_files loaded: %d" % len(X_files))
			print([len(_) for _ in X_files])
		return X_files
	def load_image_stack(self):
		"""Loads a stack of images to an internal queue"""
		
		self.all_records.check_mem()
		if self.tl() not in self.file_list_dict:
			self.file_list_dict[self.tl()] = []
		X_files = self._load_list_stack()
		if self.batch_by_pid:
			if self.verbose:
				print("Load image stack: Batching by PID")
			pdicts = []
			with_id,without_id = 0,0
			for images in X_files:
				pdict = {}
				for image in images:
					image.load_extra_info()
					if not is_nan(image.group_by):
						with_id += 1
						if image.group_by not in pdict:
							pdict[image.group_by] = []
						pdict[image.group_by].append(image)
					else:
						without_id += 1
				pdicts.append(pdict)
			if self.verbose:
				print("Load image stack: len(pdicts): %d" % len(pdicts))
				print([len(pdict) for pdict in pdicts])
				print(f"with id: {with_id}, without id: {without_id}")
			image_record_list = []
			for pdict in pdicts:
				image_record_list.append([])
				for ID in pdict:
					image_record_list[-1].append(
							BatchRecord(pdict[ID],
							batch_by_pid=True,
							sort=True,
							dtype=self.dtype,
							channels_first=self.channels_first,
							gpu_ids = self.gpu_ids,
							batch_size=self.n_dyn_inputs))
			X_files = image_record_list
		self.file_list_dict[self.tl()] = X_files
		if self.verbose:
			print("In %s loaded list of size %d" % (self.tl(),len(X_files)))
			print([len(_) for  _ in X_files])
	def _return_labels(self):
		"""Whether or not labels ought to be returned or just the images"""
		
		if self.tl() == "Folder":
			if len(self.image_folders) > 1:
				return True
			elif self._pickle_input():
				return False
		return self.label is not None and len(self.label) > 0
	def record(self,flist,index=None):
		self.file_record = set(self.file_record).union(set(flist))
		if index is None or index % 100 == 0:
			with open(self.file_record_output,"w") as fileobj:
				json.dump(list(self.file_record),fileobj,indent=4)
	def read_record(self):
		if os.path.isfile(self.file_record_output):
			with open(self.file_record_output,"r") as fileobj:
				self.file_record = set(json.load(fileobj))
	def rotate_labels(self,zero_list_addendum = None):
		if len(self.label) < 2: return
		if zero_list_addendum is not None:
			self.zero_data_list.append(zero_list_addendum)
		self.label = self.label[1:] + [self.label[0]]
		while self.label[0] in self.zero_data_list:
			self.label = self.label[1:] + [self.label[0]]
	def switch_stack(self):
		if self.verbose:
			print("Switching stack")
		self.label,self.rmatch_confounds = self.rmatch_confounds,self.label
		self.file_list_dict,self.file_list_dict_hidden = \
			self.file_list_dict_hidden,self.file_list_dict
		self.match_confounds,self.match_confounds_hidden = \
			self.match_confounds_hidden,self.match_confounds
		self.mode,self.mode_hidden = self.mode_hidden,self.mode
	def __next__(self):
		if self.verbose:
			print("self.tl(): %s" % self.tl())
			print("len(self.file_list_dict[self.tl()]): %d" % (len(self.file_list_dict[self.tl()])))
			print("len(self.file_list_dict[self.tl()][0]): %d" % (len(self.file_list_dict[self.tl()][0])))
			print("set(self.file_list_dict) : %s" % str(set(self.file_list_dict)))

		if len(self) == 0:
			self.load_image_stack()
		if self.index > len(self):
			self.database.out_dataframe()
			self.index = 0
			self.rotate_labels()
			self.load_image_stack()
			raise StopIteration
		# Temporary measure
		if self.index % 500 == 0 and self.index != 0:
			self.all_records.check_mem()
		temp = []
		for i in range(self.batch_size):
			b = self.index % len(self.file_list_dict[self.tl()])
			img = None
			if len(self.file_list_dict[self.tl()][b]) == 0: continue
			for j in range(len(self.file_list_dict[self.tl()][b])):
				im = self.file_list_dict[self.tl()][b].pop()
				self.file_list_dict[self.tl()][b] = [im] + \
					self.file_list_dict[self.tl()][b]
				try:
					if self.cache: assert self.database is not None
					img = im.get_X(augment=self.augment)
					temp.append(im)
					self.index += 1
					break
				except (ImageFileError, ValueError) as e:
					self.file_list_dict[self.tl()][b] = \
						self.file_list_dict[self.tl()][b][1:]
					continue
		if len(temp) != self.batch_size:
			print(len(temp))
			print(self.batch_size)
			#print(self.batch_size)
			#print(self.file_list_dict)
		assert(len(temp) == self.batch_size)
		if self.batch_by_pid:
			p = temp[0]
		else:
			p = BatchRecord(temp,
				dtype = self.dtype,
				sort=False,
				batch_by_pid=False,
				channels_first=self.channels_first,
				gpu_ids=self.gpu_ids,
				batch_size=self.batch_size)
		if self.return_obj:
			return p
		elif self._return_labels():
			return p.get_X(augment=self.augment),p.get_Y()
		else:
			return p.get_X(augment=self.augment)				

	def __len__(self):
		l = len(self.file_list_dict[self.tl()])
		if l == 0:
			return 0
		else:
			return l * max([len(_) for _ in self.file_list_dict[self.tl()]])
	def __iter__(self):
		return self
	def name(self):
		return 'MedImageLoader'
