import pandas as pd
import os
import numpy as np
from .utils import *
from .Records import ImageRecord
import dateutil

class DataBaseWrapper():
	"""
	Wrapper for Pandas table to cache some common and repeated functions
	"""
	def __init__(self,
					all_vars = None,
					filename=None,
					labels=[],
					confounds=[],
					dim=None,
					cdim=None,
					key_to_filename = key_to_filename_default,
					val_ranges={}):
		self.key_to_filename = key_to_filename
		check_key_to_filename(self.key_to_filename)
		self.filename = filename
		self.dim = dim
		self.val_ranges = val_ranges
		self.labels = [] if labels is None else labels
		self.confounds = [] if confounds is None else confounds
		if all_vars is not None:
			self.all_vars = all_vars
			self.columns = set(self.all_vars.columns)
		elif os.path.isfile(filename):
			self.all_vars = pd.read_pickle(self.filename)
			self.columns = set(self.all_vars.columns)
		elif self.filename is not None:
			assert(self.dim is not None)
			self.all_vars = pd.DataFrame()
			self.columns = set()
		else:
			raise Exception("DatabaseWrapper cannot have no inputs")
		self.jdict = []
		
		if isinstance(self.all_vars,str) and os.path.isfile(self.all_vars)\
			 and os.path.splitext(self.all_vars)[1] == ".pkl":
			self.all_vars = pd.read_pickle(self.all_vars)
	def has_im(self,im: ImageRecord):
		fkey = self.key_to_filename(im.npy_file,reverse=True)
		return fkey in self.all_vars.index
	def in_val_ranges(self,fkey: str) -> bool:
		valid = True
		for column_name in self.val_ranges:
			val = self.all_vars.loc[fkey,column_name]
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
		for l in self.labels:
			if l not in self.all_vars.columns:
				print("%s not in all_vars.columns" % l)
				print(self.all_vars.columns)
			assert(l in self.all_vars.columns)
		for c in self.confounds:
			assert(c in self.all_vars.columns)
		assert(len(set(self.labels).intersection(set(self.confounds)))==0)
		self.uniques = {}
		self.n_buckets=10
		for c in self.confounds + self.labels:
			self.uniques[c] = {}
			lis = list(self.all_vars.loc[:,c])
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
	def get_confound_encode(self,npy_file: str):
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
		#fkey = get_dim_str(filename,dim=self.dim)
		for c in potential_columns:
			if c in self.columns:
				val = self.loc_val(npy_file,c)
				if not(is_nan(val) or isinstance(val,str)):
					print(val)
				assert(is_nan(val) or isinstance(val,str))
				if not is_nan(val):
					break
		
		return val
	def get_ID(self,npy_file):
		id = self._get_val(npy_file,["PatientID","Patient ID"])
		return id
	def parse_date(self,d,date_format="%Y-%m-%d %H:%M:%S"):
		if is_nan(d,inc_null_str=True):
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
			#if fkey not in self.all_vars.index:
			#	print("Error - %s not present in index. Adding..." % fkey)
			#assert(fkey in self.all_vars.index)
			
			return self.all_vars.loc[fkey,c]
		except KeyError:
			nifti_file = get_dim_str(fkey,dim=self.dim,outtype=".nii.gz")
			
			if not os.path.isfile(nifti_file):
				nifti_file = get_dim_str(fkey,dim=self.dim,outtype=".nii")
			if not os.path.isfile(nifti_file):
				raise Exception("Key error: %s" % nifti_file)
			json_file = date_sorter(os.path.dirname(nifti_file),".json")[0]
			if not os.path.isfile(json_file):
				raise Exception("Key error - no JSON: %s" % fkey)
			self.add_json(nifti_file = nifti_file,json_file=json_file)
			self.out_dataframe(fkey_ass=fkey)
			return self.all_vars.loc[fkey,c]
	def add_json(self,nifti_file,json_file=None):
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
		npy_file = get_dim_str(nifti_file,self.dim)
		fkey = self.key_to_filename(npy_file,reverse=True)
		if fkey not in self.jdict and fkey not in self.all_vars.index:
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
				self.all_vars.index))
		return [self.key_to_filename(str(_)) for _ in flist]
	def out_dataframe(self,fkey_ass = None):
		
		if len(self.jdict) > 0:
			out = pd.DataFrame(self.jdict,columns = list(self.columns))
			out.set_index("fkey",inplace=True)
			if len(self.all_vars) > 0:
				self.all_vars = pd.concat([self.all_vars,out],
						ignore_index=False,
						join='inner')
				if fkey_ass is not None: assert(fkey_ass in self.all_vars.index)
			else:
				self.all_vars = out
				if fkey_ass is not None: assert(fkey_ass in self.all_vars.index)
			assert(len(self.all_vars) > 0)
			if fkey_ass is not None: assert(fkey_ass in self.all_vars.index)
			self.all_vars.drop_duplicates(inplace=True)
			if fkey_ass is not None: assert(fkey_ass in self.all_vars.index)
			self.all_vars.to_pickle(self.filename)
			if fkey_ass is not None: assert(fkey_ass in self.all_vars.index)
			self.jdict = []
	def stack_list_by_label(self,filename_list,label):
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
		filename_list_stack = [[] for _ in range(len(np.unique(lencodes)))]
		for filename,l in zip(filename_list,lencodes):
			filename_list_stack[l].append(filename)
		return filename_list_stack
