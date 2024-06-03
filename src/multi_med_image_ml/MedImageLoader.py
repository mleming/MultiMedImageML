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
import gc
from .Records import BatchRecord,ImageRecord
from .DataBaseWrapper import DataBaseWrapper

# Translates a filename to a key and back, for storing files as keys in the
# pandas dataframe. By default, the keys are the full filepaths. This function
# may need to be changed when switching to different systems
def key_to_filename_default(fkey,reverse=False):
	return fkey

class MedImageLoader():
	def __init__(self,*image_folders,
			pandas_cache = '../pandas/',
			cache = True,
			key_to_filename = key_to_filename_default,
			batch_by_pid=True,
			file_record_name=None,
			all_vars = None,
			batch_size = 14,
			dim = (96,96,96),
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
			recycle=True,
			gpu_ids = "",
			save_ram = True):
		self.channels_first = channels_first
		self.image_folders = image_folders
		self.augment = augment
		self.dim=dim
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

		# If set to true, restacks images every time via the data 
		# matching function. Best for very large and imbalanced datasets
		self.recycle=recycle
		
		check_key_to_filename(key_to_filename)

		# Stores images so that they aren't repeated in different stacks
		self.image_dict = {}
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
		if self.pickle_input():
			self.all_vars_file = self.image_folders[0]
			#assert(self.cache or os.path.isfile(self.all_vars_file))
		else:
			os.makedirs(os.path.dirname(self.pandas_cache),
				exist_ok=True)
			self.all_vars_file = os.path.join(
				self.pandas_cache,
				"all_vars_%s.pkl" % get_dim_str(dim=self.dim))
		
		self.all_vars = DataBaseWrapper(
					filename=self.all_vars_file,
					labels=self.label,
					confounds=self.confounds,
					dim=self.dim,
					key_to_filename=key_to_filename,
					val_ranges=self.val_ranges)
		if not self.pickle_input():
			self.build_pandas_database()
		self.all_vars.build_metadata()
		
		# Determine which mode the scheduler is in
		if (self.label is not None and len(self.label) > 0):
			self.mode = "match"
		else:
			self.mode = "iterate"
		
		self.mem_limit = psutil.virtual_memory().available * 0.2
		
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
		if self.return_labels():
			for l in self.label:
				if l not in self.val_ranges:
					self.match_confounds_hidden.append(l)
		self.index = 0
		self.load_image_stack()
	# Builds up the entire cache in one go — may take a while
	def build_pandas_database(self):
		assert(self.all_vars is not None)
		assert(not self.pickle_input())
		old_mode = self.mode
		self.mode = "iterate"
		self._load_list_stack()
		
		for i,filename in enumerate(self.image_dict):
			im = self.image_dict[filename]
			if not self.all_vars.has_im(im):
				try:
					im.get_image()
					im.clear_image()
				except ImageFileError:
					continue
			if i % 100 == 0:
				self.all_vars.out_dataframe()
		self.clear_mem()
		self.all_vars.out_dataframe()
		self.mode = old_mode
	def pickle_input(self):
		return len(self.image_folders) == 1 and \
			os.path.splitext(self.image_folders[0])[1] == ".pkl"
	def tl(self):
		# Top label
		
		if len(self.label) == 0: tl = "Folder"
		else: tl = self.label[0]
		if tl not in self.file_list_dict:
			self.file_list_dict[tl] = []
		return tl
	def get_file_list(self):
		if self.pickle_input() and self.tl() == "Folder":
			return [[str(_) for _ in self.all_vars.get_file_list()]]
		if self.mode == "iterate":
			if self.pickle_input():
				fname_list = [str(_) for _ in self.all_vars.get_file_list()]
				return self.all_vars.stack_list_by_label(fname_list,self.tl())
			else:
				all_filename_lists = []
				duplicate_test = set()
				for img in self.image_folders:
					flist = get_file_list(img)#,db_builder=self.all_vars)
					flist_set = set(flist)
					if len(flist_set.intersection(duplicate_test))>0:
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
				all_vars=self.all_vars.all_vars)
			fname_list = list(fname_list)
			fname_list = [self.all_vars.key_to_filename(_) for _ in fname_list]
			if len(fname_list) == 0:
				print(self.all_vars.all_vars.loc[:,self.tl()])
				raise Exception("No valid files from %s" % self.tl())
			assert(isinstance(fname_list,list))
			return self.all_vars.stack_list_by_label(fname_list,self.tl())
		else:
			raise Exception("Invalid mode: %s" % self.mode)
	def _load_list_stack(self):
		X_files = self.get_file_list() 
		for i,filename_list in enumerate(X_files):
			for j,filename in enumerate(filename_list):
				if filename in self.image_dict:
					X_files[i][j] = self.image_dict[filename]
				else:
					X_files[i][j] = ImageRecord(filename,
								dim=self.dim,
								y_nums=[i] if len(X_files) == 1 else None,
								Y_dim = self.Y_dim,
								C_dim = self.C_dim,
								dtype=self.dtype,
								all_vars = self.all_vars,
								cache=self.cache,
								static_inputs = self.static_inputs)
					self.image_dict[filename] = X_files[i][j]
		return X_files
	def load_image_stack(self):
		if self.get_mem() > self.mem_limit:
			self.clear_mem()
		#self.rotate_labels()
		if self.tl() not in self.file_list_dict:
			self.file_list_dict[self.tl()] = []
		X_files = self._load_list_stack()
		if self.batch_by_pid:
			pdict = {}
			for images in X_files:
				for image in images:
					image.load_extra_info()
					if not is_nan(image.group_by):
						if image.group_by not in pdict:
							pdict[image.group_by] = []
						pdict[image.group_by].append(image)
			image_record_list = []
			for ID in pdict:
				image_record_list.append(
							BatchRecord(pdict[ID],
							batch_by_pid=True,
							sort=True,
							dtype=self.dtype,
							channels_first=self.channels_first,
							gpu_ids = self.gpu_ids,
							batch_size=self.batch_size))
			X_files = [image_record_list]
		self.file_list_dict[self.tl()] = X_files
	def return_labels(self):
		if self.tl() == "Folder":
			if len(self.image_folders) > 1:
				return True
			elif self.pickle_input():
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
		self.label,self.rmatch_confounds = self.rmatch_confounds,self.label
		self.file_list_dict,self.file_list_dict_hidden = \
			self.file_list_dict_hidden,self.file_list_dict
		self.match_confounds,self.match_confounds_hidden = \
			self.match_confounds_hidden,self.match_confounds
		self.mode,self.mode_hidden = self.mode_hidden,self.mode
	def __next__(self):
		if len(self) == 0: self.load_image_stack()
		if self.index > len(self):
			self.all_vars.out_dataframe()
			self.index = 0
			self.rotate_labels()
			self.load_image_stack()
			raise StopIteration
		# Temporary measure
		if self.index % 1000 == 0 and self.index != 0:
			self.clear_mem()
		temp = []
		for i in range(self.batch_size):
			b = i % len(self.file_list_dict[self.tl()])
			img = None
			if len(self.file_list_dict[self.tl()][b]) == 0: continue
			for j in range(len(self.file_list_dict[self.tl()][b])):
				im = self.file_list_dict[self.tl()][b].pop()
				self.file_list_dict[self.tl()][b] = [im] + \
					self.file_list_dict[self.tl()][b]
				try:
					if self.cache: assert self.all_vars is not None
					img = im.get_image(augment=self.augment)
					temp.append(im)
					self.index += 1
					break
				except (ImageFileError, ValueError) as e:
					self.file_list_dict[self.tl()][b] = \
						self.file_list_dict[self.tl()][b][1:]
					continue
		if len(temp) != self.batch_size:
			print(len(temp))
			print(temp)
			print(self.batch_size)
			print(self.file_list_dict)
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
		elif self.return_labels():
			return p.get_image(),p.get_Y()
		else:
			return p.get_image()				

	def __len__(self):
		l = len(self.file_list_dict[self.tl()])
		if l == 0:
			return 0
		else:
			return l * max([len(_) for _ in self.file_list_dict[self.tl()]])
	def __iter__(self):
		return self
	def clear_mem(self):
		cur_mem = self.get_mem()
		times_called = []
		ssize = 0
		for filename in self.image_dict:
			if self.image_dict[filename].image is not None:
				times_called.append(self.image_dict[filename].times_called)
				#if ssize == 0: ssize = self.image_dict[filename].get_mem()
		times_called = sorted(times_called,reverse=True)
		#n = int(self.mem_limit / ssize * 0.5)
		#if n > len(times_called):
		#	warnings.warn("Something strange")
		#	n = int(len(times_called)/2)
		n = 10
		median_times_called = 1000000000 if len(times_called) < n else times_called[n]
		for filename in self.image_dict:
			if self.image_dict[filename].image is not None:
				if self.save_ram or self.image_dict[filename].times_called < median_times_called:
					self.image_dict[filename].clear_image()
		if self.save_ram:
			del self.image_dict
			self.image_dict = {}
			gc.collect()
	def get_mem(self):
		total_mem = 0
		for filename in self.image_dict:
			total_mem += float(self.image_dict[filename].get_mem())
		return total_mem
	def name(self):
		return 'MedImageLoader'
