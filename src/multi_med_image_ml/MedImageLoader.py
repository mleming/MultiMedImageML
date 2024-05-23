#!/usr/bin/python3
import os,sys
import torchvision.transforms as transforms
from pdb import set_trace as st
import random
import numpy as np
import torch
import pandas as pd
import torch.multiprocessing
from utils import *
torch.multiprocessing.set_sharing_strategy('file_system')
import shutil
import nibabel as nb
from nibabel.filebasedimages import *

from Records import BatchRecord,ImageRecord
from DataBaseWrapper import DataBaseWrapper

class MedImageLoader():
	def __init__(self,*image_folders,
			pandas_cache = '../pandas/',
			cache = False,
			path_func = None,
			batch_by_pid=True,
			return_as_patient_record=False,
			file_record_name=None,
			n_out_y=None,
			forcealt=True,
			all_vars = None,
			batch_size = 16,
			dim = (96,96,96),
			get_encoded = False,
			static_inputs = None,
			confounds = [],
			match_confounds = [],
			label = [],
			group_by = None,
			augment = True,
			val_ranges = {},
			dtype="numpy",
			Y_dim = (32,32),
			C_dim = (32,32),
			return_obj = False,
			channels_first = True):
		self.channels_first = channels_first
		self.image_folders = image_folders
		self.augment = augment
		self.dim=dim
		self.dtype=dtype
		self.cache = cache
		self.pandas_cache = pandas_cache
		self.val_ranges = val_ranges
		self.get_encoded = get_encoded
		self.batch_by_pid = group_by is not None
		self.return_as_patient_record = return_as_patient_record
		self.file_list_dict = {}
		self.static_inputs = static_inputs
		self.label = label
		self.confounds = confounds
		self.path_func = path_func
		self.group_by = group_by
		self.n_out_y = n_out_y # Set dimension for Y
		self.forcealt = forcealt
		self.Y_dim = Y_dim
		self.C_dim = C_dim
		
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
			assert(self.cache or os.path.isfile(self.all_vars_file))
		else:
			os.makedirs(os.path.dirname(self.pandas_cache),
				exist_ok=True)
			self.all_vars_file = os.path.join(
				self.pandas_cache,
				"all_vars_%s.pkl" % get_dim_str(dim=self.dim))		
		if self.batch_by_pid and not os.path.isfile(self.all_vars_file):
			warnings.warn("Must have working all vars file to group files. One once without group_by set")
		self.all_vars = DataBaseWrapper(
					filename=self.all_vars_file,
					labels=self.label,
					confounds=self.confounds,
					dim=self.dim)
		
		# Determine which mode the scheduler is in
		if (self.label is not None and len(self.label) > 0):
			self.mode = "match"
		else:
			self.mode = "iterate"
		
		#if self.Y_dim is None:
		#	self.Y_dim = (len(self.labels)
		
		# For outputting the records of files that were read in
		self.file_record_name = file_record_name 
		if self.file_record_name is not None:
			self.file_record = {}
			fd = os.path.join(wd,'json','dataloader_records')
			if not os.path.isdir(fd): os.makedirs(fd)
			self.file_record_output = os.path.join(fd,file_record_name)
		
		if self.path_func is None:
			self.path_func = lambda k: k
		self.uniques = None
		self.n_buckets = 3
		self.rmatch_confounds = self.confounds
	
		# Regular stack variables
		self.file_list_dict = {}
	
		# Switch stack variables
		self.file_list_dict_hidden = {}
		self.match_confounds_hidden = []
		self.mode_hidden = "iterate"
		self.forcealt_hidden = False
		if self.return_labels():
			for l in self.label:
				if l not in self.val_ranges:
					self.match_confounds_hidden.append(l)
		self.index = 0
		self.load_image_stack()	
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
			return [[str(_) for _ in self.all_vars.all_vars.index]]
		if self.mode == "iterate":
			if self.pickle_input():
				fname_list = [str(_) for _ in self.all_vars.all_vars.index]
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
			if len(fname_list) == 0:
				raise Exception("No valid files from %s" % self.tl())
			assert(isinstance(fname_list,list))
			return self.all_vars.stack_list_by_label(fname_list,self.tl())
		else:
			raise Exception("Invalid mode: %s" % self.mode)
	def load_image_stack(self):
		if self.get_mem() > 10000000000:
			print("Clearing memory")
			self.clear_mem()
		#self.rotate_labels()
		if self.tl() not in self.file_list_dict:
			self.file_list_dict[self.tl()] = []
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
							channels_first=self.channels_first))
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
		self.forcealt,self.forcealt_hidden = self.forcealt_hidden,self.forcealt
		self.mode,self.mode_hidden = self.mode_hidden,self.mode
	def __next__(self):
		if len(self) == 0: self.load_image_stack()
		if self.index > len(self):
			if self.cache:
				self.all_vars.out_dataframe()
			self.index = 0
			self.rotate_labels()
			self.load_image_stack()
			raise StopIteration
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
					img = im.get_image()
					temp.append(im)
					self.index += 1
					break
				except ImageFileError:
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
				channels_first=self.channels_first)
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
		for filename in self.image_dict:
			self.image_dict[filename].clear_image()
	def get_mem(self):
		total_mem = 0
		for filename in self.image_dict:
			total_mem += float(self.image_dict[filename].get_mem())
		return total_mem
	def name(self):
		return 'MedImageLoader'