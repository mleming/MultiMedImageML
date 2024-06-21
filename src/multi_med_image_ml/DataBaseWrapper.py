import pandas as pd
import os
import numpy as np
from .utils import *
from .Records import ImageRecord
import dateutil

class DataBaseWrapper:
	"""Wrapper for Pandas table to cache some common and repeated functions
	
	DataBaseWrapper stores a pandas dataframe that contains metadata about 
	the images being read in. It also builds up this dataframe in real time
	if only DICOM or NIFTI/JSON files are present. One purpose is to 
	translate tokenized values in the dataframe to one-hot vectors that
	can be read by a DL model in the fastest way possible. Another purpose
	is to contain a storage of cached image files of a particular size.
	
	Attributes:
		database (pandas.DataFrame): The internal Pandas dataframe (default None)
		filename (str): The filename of the Pandas Dataframe pickle (default None)
		labels (list): Labels that are read in by the current MedImageLoader (default [])
		confounds (list): Confound names that are read in by the current MedImageLoader (default [])
		val_ranges (dict): List of values that can be returned. See MedImageLoader (default {})
		X_dim (tuple): Image dimensions (default None)
		key_to_filename (callback): Function to translate a key to a filename and back (default key_to_filename_default)
		jdict (dict): Dictionary of values read from JSON files that are accumulated and periodically merged with the pandas Dataframe (database) as it's build up. Used as an intermediary to prevent too much fragmenting in the DataFrame (default {})
		columns (set): Columns of the database. Used for quick reference.
		
	"""
	def __init__(self,
					database = None,
					filename=None,
					labels=[],
					confounds=[],
					X_dim=None,
					key_to_filename = key_to_filename_default,
					val_ranges={},
					precedence=[]):
		self.key_to_filename = key_to_filename
		check_key_to_filename(self.key_to_filename)
		self.filename = filename
		self.X_dim = X_dim
		self.val_ranges = val_ranges
		self.labels = [] if labels is None else labels
		self.confounds = [] if confounds is None else confounds
		self.precedence = precedence
			
		if database is not None:
			self.database = database
			self.columns = set(self.database.columns)
		elif os.path.isfile(filename):
			self.database = pd.read_pickle(self.filename)
			self.columns = set(self.database.columns)
		elif self.filename is not None:
			assert(self.X_dim is not None)
			self.database = pd.DataFrame()
			self.columns = set()
		else:
			raise Exception("DatabaseWrapper cannot have no inputs")
		if len(self.precedence) > 0:
			label_ID = {}
			if "PatientID" in self.columns:
				pid_key = "PatientID"	
			elif "Patient ID" in self.columns:
				pid_key = "Patient ID"
			for index in self.database.index:
				PID = self.database.loc[index,pid_key]
				l = self.database.loc[index,self.labels[0]]
				if is_nan(l):
					continue
				if PID not in label_ID or (self.precedence.index(l) < self.precedence.index(label_ID[PID])):
					label_ID[PID] = l
			for index in self.database.index:
				PID = self.database.loc[index,pid_key]
				#self.database.loc[index,self.labels[0]] = label_ID[PID]
				if PID in label_ID:
					self.database.at[index,self.labels[0]] = label_ID[PID]
		self.jdict = []
	def has_im(self,im: ImageRecord) -> bool:
		"""Determines if the DataBaseWrapper contains a given ImageRecord
		
		Args:
			im (ImageRecord): Input image
			
		Returns:
			bool - Whether image is present in database
		"""
		
		fkey = self.key_to_filename(im.npy_file,reverse=True)
		return fkey in self.database.index
	def in_val_ranges(self,fkey: str) -> bool:
		"""Determines if the input fkey is in a valid value range
		
		Args:
			fkey (str): Lookup key to the pandas DataFrame
		
		Returns:
			bool - Whether the key is present
		
		"""
		valid = True
		for column_name in self.val_ranges:
			val = self.database.loc[fkey,column_name]
			if is_nan(val,inc_null_str=True):
				return False
			elif isinstance(val,str):
				if val not in self.val_ranges[column_name]:
					return False
			elif isinstance(val,float) or isinstance(val,int):
				min_,max_ = self.val_ranges[column_name]
				if val < min_ or val > max_:
					return False
		return True
	def build_metadata(self):
		"""Builds internal lookup tables used to convert continuous and label-
		based variables to one-hot vectors that can be read by an ML model."""
		
		for l in self.labels:
			if l not in self.database.columns:
				print("%s not in database.columns" % l)
				print(self.database.columns)
			assert(l in self.database.columns)
		for c in self.confounds:
			assert(c in self.database.columns)
		assert(len(set(self.labels).intersection(set(self.confounds)))==0)
		self.uniques = {}
		self.n_buckets=10
		for c in self.confounds + self.labels:
			self.uniques[c] = {}
			lis = list(self.database.loc[:,c])
			if np.any([isinstance(_,str) for _ in lis]):
				self.uniques[c]["discrete"] = True
				u = {}
				for l in lis:
					if not is_nan(l):
						if l not in u:
							u[l] = 0
						u[l] += 1
				# This ensures that confounds are ordered by their frequency
				# of appearance in the real data. Thus, if there were 7500
				# females and 2500 males in a dataset, females would be
				# encoded as [1,0] and males as [0,1].
				if c in self.confounds:
					u = sorted(
								[(_,u[_]) for _ in u],
								key=lambda k: k[1],
								reverse=True)
					u = [_[0] for _ in u]
				else:
					try:
						u = sorted(list(u))
					except:
						print(u)
						exit()
				self.uniques[c]["unique"] = u
				self.n_buckets = max(self.n_buckets,len(u))
			else:
				self.uniques[c]["discrete"] = False
				max_ = -np.inf
				min_ = np.inf
				nonnan_list = []
				for l in lis:
					if not is_nan(l):
						max_ = max(max_,l)
						min_ = min(min_,l)
						nonnan_list.append(l)
				self.uniques[c]["max"] = max_
				self.uniques[c]["min"] = min_
				self.uniques[c]["nonnan_list"] = sorted(nonnan_list)

				self.n_buckets_cont = min([self.n_buckets,
									10,
									int(len(self.uniques[c]["nonnan_list"])/10)
								])
				skips = int(len(self.uniques[c]["nonnan_list"])/\
					float(self.n_buckets_cont)) + 1
				self.uniques[c]["nonnan_list"] = \
					self.uniques[c]["nonnan_list"][::skips]
				# Get mean between density and range dists
				max_ = self.uniques[c]["max"]
				min_ = self.uniques[c]["min"]
				rd = np.arange(self.n_buckets_cont)
				rd = rd / float(self.n_buckets_cont-1)
				rd = list((rd * (max_ - min_)) + min_)
				self.uniques[c]["nonnan_list"] = \
					[(rd[i] + \
					self.uniques[c]["nonnan_list"][i])/2 \
					for i in range(self.n_buckets_cont)]
				self.uniques[c]["nonnan_list"][-1] = max_
				self.uniques[c]["nonnan_list"][0] = min_
				assert(len(self.uniques[c]["nonnan_list"]) == \
					self.n_buckets_cont)
	def get_confound_encode(self,npy_file: str) -> list:
		"""Returns list of integers that represent the confounds of a given input file
	
		Args:
			npy_file (str): Numpy file of the given record, which is converted into a key
		
		Returns:
			cnum_list (list): A list of integers indicating the nth confound of the datapoint
	
		"""
		
		assert(os.path.splitext(npy_file)[1] == ".npy")
		confound_strs = [self.loc_val(npy_file,c) for c in self.confounds]
		cnum_list = []
		for j,c in enumerate(self.confounds):
			if self.uniques[c]["discrete"]:
				c_uniques = self.uniques[c]["unique"]
				if is_nan(confound_strs[j]):
					cnum_list.append(-1)
				else:
					cnum_list.append(c_uniques.index(confound_strs[j]))
			else:
				max_ = self.uniques[c]["max"]
				min_ = self.uniques[c]["min"]
				if is_nan(confound_strs[j]):
					cnum_list.append(-1)
				else:
					unnl = self.uniques[c]["nonnan_list"]
					for kk in range(len(unnl)-1):
						if unnl[kk] <= confound_strs[j] and \
							unnl[kk+1] >= confound_strs[j]:
							cnum_list.append(kk)
							break
		assert(len(cnum_list) == len(self.confounds))
		return cnum_list
	def get_label_encode(self,npy_file: str):
		"""Returns list of integers that represent the confounds of a given input file
	
		Args:
			npy_file (str): Numpy file of the given record, which is converted 
				into a key
		
		Returns:
			cnum_list (list): A list of integers indicating the nth label of the datapoint
	
		"""
		
		assert(os.path.splitext(npy_file)[1] == ".npy")
		label_strs = [self.loc_val(npy_file,c) for c in self.labels]
		cnum_list = []
		for j,c in enumerate(self.labels):
			if self.uniques[c]["discrete"]:
				c_uniques = self.uniques[c]["unique"]
				if is_nan(label_strs[j]):
					cnum_list.append(-1)
				else:
					cnum_list.append(c_uniques.index(label_strs[j]))
			else:
				max_ = self.uniques[c]["max"]
				min_ = self.uniques[c]["min"]
				if is_nan(confound_str[j]):
					cnum_list.append(-1)
				else:
					unnl = self.uniques[c]["nonnan_list"]
					for kk in range(len(unnl)-1):
						if unnl[kk] <= label_strs[j] and \
							unnl[kk+1] >= label_strs[j]:
							cnum_list.append(kk)
							break
		assert(len(cnum_list) == len(self.labels))
		return cnum_list
	def _get_val(self,npy_file: str,potential_columns):
		assert(np.all([isinstance(_,str) for _ in potential_columns]))
		val = None
		for c in potential_columns:
			if c in self.columns:
				val = self.loc_val(npy_file,c)
				if not(is_nan(val) or isinstance(val,str)):
					print(val)
				assert(is_nan(val) or isinstance(val,str))
				if not is_nan(val):
					break
		return val
	
	def get_ID(self,npy_file: str) -> str:
		"""Returns the Patient ID, if present in the database. Attempts to 
		guess it using the keys 'PatientID' and 'Patient ID'
		
		Args:
			npy_file (str): Cached numpy file of the image
		
		Returns:
			id (str): ID of the patient in question
		"""
		
		id = self._get_val(npy_file,["PatientID","Patient ID"])
		return id
	
	def parse_date(self,d,date_format="%Y-%m-%d %H:%M:%S"):
		"""Parses the date string"""
		
		if is_nan(d,inc_null_str=True) or d == "":
			return datetime.datetime(year=1970,month=1,day=1)
		elif is_float(d):
			return dateutil.parser.parse(str(d))
		else:
			try:
				return datetime.datetime.strptime(
					d.replace("_"," ").split(".")[0],
					date_format)
			except ValueError:
				return dateutil.parser.parse(d.replace("_"," "))
	def get_exam_date(self,npy_file: str) -> datetime.date:
		d = self._get_val(npy_file,["ExamEndDTS","Acquisition Date"])
		return self.parse_date(d)
	
	def get_birth_date(self,npy_file: str) -> datetime.date:
		d = self._get_val(npy_file,["BirthDTS"])
		return self.parse_date(d)
	
	def loc_val(self,npy_file,c):
		fkey = self.key_to_filename(npy_file,reverse=True)
		try:	
			return self.database.loc[fkey,c]
		except KeyError:
			nifti_file = get_dim_str(fkey,X_dim=self.X_dim,outtype=".nii.gz")
			
			if not os.path.isfile(nifti_file):
				nifti_file = get_dim_str(fkey,X_dim=self.X_dim,outtype=".nii")
			if not os.path.isfile(nifti_file):
				raise Exception("Key error: %s" % nifti_file)
			json_file = date_sorter(os.path.dirname(nifti_file),".json")[0]
			if not os.path.isfile(json_file):
				raise Exception("Key error - no JSON: %s" % fkey)
			self.add_json(nifti_file = nifti_file,json_file=json_file)
			self.out_dataframe(fkey_ass=fkey)
			return self.database.loc[fkey,c]
	def add_json(self,nifti_file,json_file=None):
		"""Adds a JSON from a DICOM file to the internal jdict, which will later
		be compiled into the DataFrame
		
		Args:
			nifti_file (str): Nifti (.nii.gz) file that was output by the DICOM preprocessing software
			json_file (str): JSON file was output by the DICOM preprocessing software. If not present, searches for the JSON file in the same directory as the the input nifti file.
		"""
		
		if not not_temp(nifti_file): return
		if json_file is None:
			json_file = "%s.json" % os.path.splitext(
				os.path.splitext(nifti_file)[0])[0]
			if not (os.path.isfile(json_file)):
				json_file = os.path.join(
					os.path.dirname(nifti_file),
					"metadata.json")
				if not os.path.isfile(json_file):
					return
		npy_file = get_dim_str(nifti_file,self.X_dim)
		fkey = self.key_to_filename(npy_file,reverse=True)
		if fkey not in self.jdict and fkey not in self.database.index:
			with open(json_file,'r') as fileobj:
				json_dict = json.load(fileobj)
			json_dict["fkey"] = fkey
			json_dict["filename"] = fkey
			for item in json_dict:
				if isinstance(json_dict[item],list):
					json_dict[item] = "_".join(
							[str(_) for _ in sorted(json_dict[item])]
						)
			self.columns = self.columns.union(set(json_dict))
			self.jdict.append(json_dict)
			assert(len(self.jdict) > 0)
	def get_file_list(self):
		flist = list(filter(lambda k: self.in_val_ranges(k),
				self.database.index))
		return [self.key_to_filename(str(_)) for _ in flist]
	def out_dataframe(self,fkey_ass = None):
		"""Merges the jdict with the dataframe and outputs it."""
		
		if len(self.jdict) > 0:
			out = pd.DataFrame(self.jdict,columns = list(self.columns))
			out.set_index("fkey",inplace=True)
			if len(self.database) > 0:
				self.database = pd.concat([self.database,out],
						ignore_index=False,
						join='inner')
				if fkey_ass is not None: assert(fkey_ass in self.database.index)
			else:
				self.database = out
				if fkey_ass is not None: assert(fkey_ass in self.database.index)
			assert(len(self.database) > 0)
			if fkey_ass is not None: assert(fkey_ass in self.database.index)
			self.database.drop_duplicates(inplace=True)
			if fkey_ass is not None: assert(fkey_ass in self.database.index)
			self.database.to_pickle(self.filename)
			if fkey_ass is not None: assert(fkey_ass in self.database.index)
			self.jdict = []
	def stack_list_by_label(self,filename_list,label):
		"""Reorganizes an input filename list by the list of labels. So, a list
		of filenames of men and women with sex as the given input label will be 
		returned as two lists of men and women.
		
		Args:
			filename_list (list): List of filenames
			label (str): Label to be reorganized by. Must be one of the confounds or labels in the DataBaseWrapper.
		
		Returns:
			filename_list_stack (list[list]): List of separate filename lists
		"""
		
		if label in self.labels:
			sel_list = self.labels
		elif label in self.confounds:
			sel_list = self.confounds
		else:
			raise Exception("%s not in labels or confounds")
		lnum = sel_list.index(label)
		if label in self.labels:
			lencodes = [self.get_label_encode(f)[lnum] for f in filename_list]
		elif label in self.confounds:
			lencodes = [self.get_confound_encode(f)[lnum] for f in filename_list]
		else:
			raise Exception(f"Label {label} not in label or confounds")
		filename_list_stack = [[] for _ in range(max(lencodes) + 2)]
		try:
			for filename,l in zip(filename_list,lencodes):
				filename_list_stack[l].append(filename)
			for i in range(len(filename_list_stack)-1,-1,-1):
				if len(filename_list_stack[i]) == 0: del filename_list_stack[i]
			return filename_list_stack
		except:
			print(lencodes)
			print(filename)
			print(l)
			print(label)
			raise Exception("Out of bounds error")
