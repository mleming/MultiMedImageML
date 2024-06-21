import os,sys,json,argparse,glob,shutil,warnings,re,torch,random,requests,gdown
import datetime,datefinder,platform,dicom2nifti
import numpy as np
import pandas as pd
from copy import deepcopy as copy
import nibabel as nb
from sklearn.preprocessing import MultiLabelBinarizer
import pydicom as dicom
from scipy import ndimage,stats
from scipy.stats import mannwhitneyu 
from sklearn.metrics import roc_curve, auc
from dateutil import relativedelta,parser
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()
from platformdirs import user_cache_dir
import urllib.request
from typing import Callable
platform_system = platform.system()

# Used to get a training set with equal distributions of input covariates
# Can also be used to only have certain ranges of continuous covariates,
# or certain labels of discrete covariates.

if shutil.which('dcm2niix') is None:
	dcm2niix_installed = False
	warnings.warn(
	"""Install dcm2niix for best results
	https://github.com/rordenlab/dcm2niix""")
else:
	dcm2niix_installed = True

def compile_dicom_py(dicom_folder: str):
	dicom2nifti.convert_directory(dicom_folder,"conv.nii.gz")
	dicom_files = (glob.glob(os.path.join("*.dcm")))

def key_to_filename_default(filename: str,reverse: bool =False) -> str:
	"""Default function for converting a pandas key to a filename
	
	This function can be replaced by a more elaborate one that is able to
	convert the location of a .npy file on a filesystem to a lookup key
	in a pandas dataframe. By default, the file path is the key.
	"""
	
	return filename

def check_key_to_filename(key_to_filename: Callable[[str,bool], str]):
	"""Verifies that the key to file name conversion method is working properly
	
	This method is called to verify that a user-defined key-to-filename function
	is properly implemented, such that the function is able to convert an input
	path to a key forwards and backwards.
	"""
	
	if key_to_filename(
			key_to_filename(os.path.realpath(__file__),reverse=False),
		reverse=True) != os.path.realpath(__file__):
		raise Exception("""
			key_to_filename must be a function that translates
			a filename to an index key and back with the
			'reverse' option input""")

def compile_dicom_folder(dicom_folder: str,db_builder=None):
	"""Converts a folder of dicoms to a .nii.gz, with .json metadata
	
	Uses dcm2niix, since that's had the best results overall when converting
	dicom to nifti, even though it's a system command. Uses pydicom as a backup.
	The resulting files are stored in the folder. Also takes a DatabaseWrapper
	object for building the database in real time.
	
	Args:
		dicom_folder (str): Folder of interest
		db_builder (multi_med_image_ml.DataBaseWrapper.DateBaseWrapper): Optional input for building up the database
	"""
	dicom_folder = os.path.realpath(dicom_folder)
	
	if dcm2niix_installed:
		cwd = os.getcwd()
		os.chdir(dicom_folder)
		if platform_system == "Windows":
			os.system('dcm2niix *.dcm 2> nul')
		else:
			os.system('dcm2niix *.dcm >/dev/null 2>&1')
		os.chdir(cwd)
	else:
		dicom2nifti.convert_directory(dicom_folder,dicom_folder)
	
	json_file = date_sorter(dicom_folder,'.json')
	if len(json_file) == 0:
		raise FileNotFoundError("No json file in %s" % dicom_folder)
	json_file = json_file[-1]
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		dcm = dicom.dcmread(glob.glob(os.path.join(dicom_folder,'*.dcm'))[0])
		with open(json_file,'r') as fileobj:
			json_dict = json.load(fileobj)
	
		for element in dcm:
			if isinstance(element.value,str) or isinstance(element.value,int)\
				or isinstance(element.value,float):
				n = element.name.replace("[","").replace("]","")
				if n not in json_dict:
					json_dict[n] = element.value
	with open(os.path.join(dicom_folder,json_file),'w') as fileobj:
		json.dump(json_dict,fileobj,indent=4)
	nii_file = date_sorter(dicom_folder,'.nii*')
	if len(nii_file) > 0:
		nii_file = nii_file[-1]
	elif len(date_sorter(dicom_folder,'.nii.gz')) > 0:
		nii_file = date_sorter(dicom_folder,'.nii.gz')[-1]
		if db_builder is not None:
			db_builder.add_json(nifti_file=nii_file,json_file=json_file)
		return nii_file,json_file
	else:
		raise FileNotFoundError("No nii file in %s" % dicom_folder)
	return nii_file,json_file

def get_dim_str(filename: str = None,
		X_dim: tuple = None,
		outtype: str = ".npy") -> str:
	"""Converts an input filename to the filename of the cached .npy file
	
	Given an input filename (e.g. /path/to/myfile.nii.gz) with a given dimension
	(e.g. (96,48,48)), converts the filepath to the cached version (e.g.
	/path/to/myfile_resized_96_48_48.npy). Perfect cube dimensions are annotated
	with a single number rather than three. If no filename is input, the
	string itself is returned (resized_96_48_48.npy).
	
	Args:
		filename (str): Name of the file to be converted (Default None)
		X_dim (tuple): Size that the image is going to be resized to (Default None)
		outtype (str): 
	
	Returns:
		String of the cached image file, or a string that can be added to a filename
	
	"""
	
	assert(X_dim is not None)
	if max(X_dim) == min(X_dim):
		dim_str = str(X_dim[0])
	else:
		dim_str = "_".join([str(_) for _ in X_dim])
	if filename is not None:
		base,ext1 = os.path.splitext(filename)
		base,ext2 = os.path.splitext(base)
		if outtype == ".npy":
			if filename.endswith(f"resized_{dim_str}.npy"):
				return filename
			elif ext1.lower() == ".npy":
				foo = re.sub("resized_[0-9].*.npy$",
						f"resized_{dim_str}.npy",filename)
				return foo
			return "%s_resized_%s.npy" % (base,dim_str)
		elif outtype == "dicom":
			return os.path.dirname(filename)
		else:
			assert(outtype[0] == ".")
			if filename.endswith(f"_resized_{dim_str}.npy"):
				return filename.replace(f"_resized_{dim_str}.npy",outtype)
			else:
				return filename.replace(ext2+ext1,outtype)
	else:
		return dim_str

def download_file_from_google_drive(file_id: str, destination: str):
	"""Downloads files from Google drive
	
	Downloads files from Google drive and saves them to a destination.
	
	Args:
		file_id (str): ID in the Google Drive URL
		destination (str): Place to save the file to
	"""
	
	URL = "https://docs.google.com/uc?export=download&confirm=t"

	session = requests.Session()

	response = session.get(URL, params={"id": file_id}, stream=True)
	token = get_confirm_token(response)
	print("token")
	print(token)
	print("response")
	print(response)
	token = "t"
	if token:
		params = {"id": file_id, "confirm": token}
		response = session.get(URL, params=params, stream=True)
	print(destination)
	save_response_content(response, destination)


def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith("download_warning"):
			return value
	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768
	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk:  # filter out keep-alive new chunks
				f.write(chunk)

def download_weights(weights: str):
	"""Downloads and caches pretrained model weights
	
	Downloads model weights from Google drive and stores them in the user's
	cache for future use.
	
	Args:
		weights (str): String indicating which weights can be used.
	"""
	
	drive_url = "https://drive.google.com/uc?export=download&confirm=pbef&id="
	weights_dir = os.path.join(user_cache_dir(),"MultiMedImageML","weights")
	print(weights_dir)
	os.makedirs(weights_dir,exist_ok=True)
	weights_file = os.path.join(weights_dir,f"{weights}.pt")
	if os.path.isfile(weights_file):
		return weights_file
	weights_lib_json = os.path.join(weights_dir,"weights.json")
	if not os.path.isfile(weights_lib_json):
		file_id = "1Scl5iib7V5pWRULKnc6-k0edN2YfGhc7"
		gdown.download(
			f"{drive_url}{file_id}",
   	 		weights_lib_json
		)
	with open(weights_lib_json,'r') as fileobj:
		weights_lib = json.load(fileobj)
	if weights not in weights_lib:
		raise Exception(
			"No such model: %s. Available options are %s" % \
			(weights," ".join(list(weights_lib)))
			)
	else:
		file_id = weights_lib[weights]
		gdown.download(
			f"{drive_url}{file_id}",
   	 		weights_file
		)
	assert(os.path.isfile(weights_file))
	return weights_file

def is_image_file(filename: str) -> bool:
	"""Determines if input file is medical image

	Determines if the input is an applicable image file. Excludes temporary
	files.
	
	Args:
		filename (str): Path to file
	
	Returns:
		bool
	"""
	
	basename,ext = os.path.splitext(filename)
	_,ext2 = os.path.splitext(basename)
	ext = ext2 + ext
	basename = os.path.basename(basename)
	if not not_temp(basename): return False
	return ext.lower() in [".nii",".nii.gz"]

def is_dicom(filename):
	"""Determines if file is dicom
	
	"""
	
	basename,ext = os.path.splitext(filename)
	return ext.lower() == ".dcm"

def get_file_list(obj,allow_list_of_list : str =True,db_builder=None):
	"""Searches a folder tree for all applicable images.
	
	Uses the os.walk method to search a folder tree and returns a list of image
	files. Relies on get_file_list_from_str and get_file_list_from_list to do 
	so. Takes in a DataBaseWrapper (db_builder) to build up a pandas dataframe
	during the search.
	
	Args:
		obj (list or str): List of string of interest
		allow_list_of_list (str): Allows lists of lists to be parsed
		db_builder (multi_med_image_ml.DataBaseWrapper.DateBaseWrapper): Optional input to allow database to be build up
	"""

	if isinstance(obj,str):
		obj = get_file_list_from_str(obj,db_builder=db_builder)
	elif isinstance(obj,list):
		obj = get_file_list_from_list(obj,db_builder=db_builder)
	else:
		raise Exception("Invalid path input")
	assert(isinstance(obj,list))
	assert(np.all([isinstance(_,str) for _ in obj]))
	if np.all([len(_) == 0 for _ in obj]):
		raise Exception("No valid files found")
	elif np.any([len(_) == 0 for _ in obj]):
		raise Exception("One without valid files")
	return obj


def get_file_list_from_str(obj,db_builder=None):
	assert(isinstance(obj, str))
	if os.path.isfile(obj):
		if is_image_file(obj):
			return [obj]
		else:
			return []
	elif os.path.isdir(obj):
		all_filename_list = []
		for root, dirs, files in os.walk(obj, topdown=False,followlinks=True):
			n_ims = 0
			n_dics = 0
			for name in files:
				filename = os.path.realpath(os.path.join(root, name))
				if is_image_file(filename):
					all_filename_list.append(filename)
					if db_builder is not None:
						db_builder.add_json(nifti_file=filename)
					n_ims += 1 
				elif is_dicom(filename):
					n_dics += 1
			if n_dics > 0 and n_ims == 0:
				all_filename_list.append(os.path.realpath(root))
		return all_filename_list
	else:
		raise Exception("Invalid string input: %s" % obj)

def get_file_list_from_list(obj,allow_list_of_list=True,db_builder=None):
	assert(isinstance(obj,list))
	if np.all([isinstance(_,list) for _ in obj]):
		if not allow_list_of_list:
			raise Exception("Cannot have nested lists")
		list_of_list = []
		for l in obj:
			list_of_list = list_of_list + get_file_list(l,
				allow_list_of_list = False,db_builder=db_builder)
		return list_of_list
	elif np.all([isinstance(_,str) for _ in obj]):
		list_of_str = []
		for l in obj:
			list_of_str = list_of_str + get_file_list_from_str(l,
				db_builder=db_builder)
		return list_of_str
	else:
		raise Exception("""Inputs must be strings, lists of lists,
			or lists of strings""")

# Ad-hoc function that determines whether given keys are equal
def equal_terms(term):
	trans_dict = {'NOT_HISPANIC':'NO_NON_HISPANIC',
				'HISPANIC':'YES_HISPANIC',
				'PREFER_NOT_TO_SAY/DECLINE':'UNAVAILABLE',
				'DECLINED': 'UNAVAILABLE',
				'NULL':'UNAVAILABLE'}
	if term in trans_dict: return trans_dict[term]
	else: return term

def not_temp(filename):
	basename = os.path.basename(filename)
	basename = os.path.splitext(basename)[0]
	basename = os.path.splitext(basename)[0]
	return basename.lower() != "temp"

def date_sorter(folder,ext):
	filelist = glob.glob(os.path.join(folder,"*" + ext))
	filelist = list(filter(lambda k: not_temp(k),filelist))
	filelist = sorted(filelist,key=os.path.getmtime)
	return filelist

def compile_dicom(dicom_folder: str,cache=True,db_builder=None,verbose=False):
	"""Compiles a folder of DICOMs into a .nii and .json file
	
	Takes a folder of dicom files and turns it into a .nii.gz file, with
	metadata stored in a .json file. Relies on dcm2niix.
	
	Args:
		dicom_folder (str): The folder with DICOM files
		cache (bool): Whether to cache .npy files in the DICOM folder
		db_builder (multi_med_image_ml.DataBaseWrapper.DateBaseWrapper): Object that may optionally be input for building up the database

	"""

	assert(os.path.isdir(dicom_folder))
	json_file = date_sorter(dicom_folder,'.json')
	nii_file = date_sorter(dicom_folder,'.nii*')
	if (len(json_file) == 0 or len(nii_file) == 0) or (not cache):
		nii_file,json_file = compile_dicom_folder(dicom_folder,
			db_builder=db_builder)
	else:
		nii_file,json_file = nii_file[-1],json_file[-1]
	_,ext1 = os.path.splitext(nii_file)
	_,ext2 = os.path.splitext(_)
	tempfile = os.path.join(dicom_folder,"temp%s%s"%(ext2,ext1))
	if not os.path.isfile(nii_file):
		print("nii_file %s not found" % nii_file)

	cwd = os.getcwd()
	os.chdir(os.path.dirname(nii_file))
	if verbose:
		print("Reorienting %s" % nii_file)
		print("tempfile is %s" % tempfile)
	os.system("fslreorient2std '%s' '%s' >/dev/null 2>&1" % (os.path.basename(nii_file),os.path.basename(tempfile)))
	if not os.path.isfile(tempfile):
		if os.path.isfile(tempfile + ".gz"):
			tempfile = tempfile + ".gz"
			if os.path.splitext(nii_file)[1] == ".nii":
				os.remove(nii_file)
				nii_file = nii_file + ".gz"
		else:
			print("Tempfile %s not found" % tempfile)
	if (os.path.isfile(tempfile)):
		shutil.move(tempfile,nii_file)
	
	if os.path.splitext(nii_file)[1].lower() == ".nii":
		os.system("gzip '%s' >/dev/null 2>&1" % os.path.basename(nii_file))
		nii_file = nii_file + ".gz"
		assert(os.path.isfile(nii_file))
	os.chdir(cwd)
	if db_builder is not None:
		db_builder.add_json(nifti_file=nii_file,json_file=json_file)
	return nii_file,json_file

# Resizes and standardizes 3d numpy arrays to normalized dimensions
def resize_np(nifti_data,dim):
	if nifti_data.min() < 0 or nifti_data.max() > 1:
		nifti_data -= nifti_data.min()
		m = nifti_data.max()
		nifti_data = nifti_data / m
	if nifti_data.dtype != np.float32:
		nifti_data = nifti_data.astype(np.float32)
	if len(nifti_data.shape) != len(dim):
		nifti_data = np.squeeze(nifti_data)
		if len(nifti_data.shape) != len(dim):
			nifti_data = np.squeeze(np.mean(nifti_data,axis=-1))
		assert(len(nifti_data.shape) == len(dim))
	if nifti_data.shape != tuple(dim):
		zp = [dim[i]/nifti_data.shape[i] for i in range(len(dim))]
		nifti_data = ndimage.zoom(nifti_data,zp)
	return nifti_data

# Prime number functions used in the data matching schemes
def prime(i, primes):
	for prime in primes:
		if not (i == prime or i % prime):
			return False
	primes.append(i)
	return i

def get_first_n_primes(n):
	primes = []
	i, p = 2, 0
	while True:
		if prime(i, primes):
			p += 1
			if p == n:
				return primes
		i += 1

def discretize_value(v,buckets):
	if isinstance(v,str):
		for i in range(len(buckets)):
			if buckets[i] == v:
				return i
	else:
		return np.searchsorted(buckets,v)
	assert(False)

# This method uses prime numbers to speed up datapoint matching. Each bucket
# gets a prime number, and each datapoint is assigned a product of these primes.
# These are then matched with one another.
def get_prime_form(confounds,n_buckets,sorted_confounds = None):
	if sorted_confounds is None:
		sorted_confounds = np.sort(confounds,axis=0)
	n_primes = get_first_n_primes(np.sum(n_buckets) + 1)
	discretized_confounds = np.zeros(confounds.shape)
	for i in range(confounds.shape[0]):
		if isinstance(confounds[i,0],str):
			buckets = np.unique(confounds[i,:])
		else:
			buckets_s = []
			lim = int(np.ceil(sorted_confounds.shape[1]/float(n_buckets[i])))
			for kk in range(0,sorted_confounds.shape[1],lim):
				buckets_s.append(sorted_confounds[i,kk])
			buckets_s.append(sorted_confounds[i,-1])
			buckets_s = np.array(buckets_s)
			min_conf = sorted_confounds[i,0]
			max_conf = sorted_confounds[i,-1]
			buckets_v = (np.array(range(n_buckets[i] + 1))/float(n_buckets[i]))\
				 * (max_conf - min_conf) + min_conf
			sv_ratio = 1.0
			buckets = (sv_ratio) * buckets_s + (1.0 - sv_ratio) * buckets_v
		for j in range(confounds.shape[1]):
			d = discretize_value(confounds[i,j],buckets)
			d = n_primes[int(np.sum(n_buckets[:i])) + d]
			discretized_confounds[i,j] = d
	return discretized_confounds

# Given buckets, selects values that fall into each one
def get_class_selection(classes,primed,unique_classes=None):
	assert(len(classes) == len(primed))
	if unique_classes is None:
		num_classes = len(np.unique(classes))
	else:
		num_classes = len(unique_classes)
	selection = np.zeros(classes.shape,dtype=bool)
	hasher = {}
	rr = list(range(len(classes)))
	random.shuffle(rr)

	for i in rr:
		if True:
			p = primed[i]
			if p not in hasher:
				hasher[p] = [[] for x in range(num_classes)]
			hasher[p][classes[i]].append(i)
		else:
			print("Hasher screw up")
			exit()
	for key in hasher:
		value = hasher[key]
		admitted_values = min(map(lambda k:len(k),value))
		for arr in value:
			for i in range(admitted_values):
				selection[arr[i]] = True
	return selection

def multi_mannwhitneyu(arr):
	max_p = -np.Inf
	min_p = np.Inf
	for i in range(len(arr)):
		for j in range(i+1,len(arr)):
			try:
				s,p = stats.ttest_ind(arr[i],arr[j])
			except:
				p = 1
			if p > max_p:
				max_p = p
			if p < min_p:
				min_p = p
	return min_p,max_p

def test_all(classes,confounds):
	unique_classes = np.unique(classes)
	all_min_p = np.Inf
	for i in range(confounds.shape[0]):
		if not isinstance(confounds[i,0],str):
			ts = [confounds[i,classes == j] for j in unique_classes]
			min_p,max_p = multi_mannwhitneyu(ts)
			if min_p < all_min_p:
				all_min_p = min_p
	return all_min_p

def integrate_arrs(S1,S2):
	assert(len(S1) >= len(S2))
	assert(np.sum(~S1) == len(S2))
	if len(S1) == len(S2):
		return S2
	i = 0
	i2 = 0
	output = np.zeros(S1.shape,dtype=bool)
	while i < len(S1):
		if ~S1[i]:
			output[i] = S2[i2]
			i2 += 1
		i += 1
	assert(np.sum(output) == np.sum(S2))
	return output

def integrate_arrs_none(S1,S2):
	assert(len(S1) >= len(S2))
	assert(np.sum(S1) == len(S2))
	i = 0
	i2 = 0
	output = np.zeros(S1.shape,dtype=bool)
	while i < len(S1):
		if S1[i]:
			output[i] = S2[i2]
			i2 += 1
		i += 1
	return output

# Returns a boolean array that is true if either classes or confounds has a None
# or NaN value anywhere at the given index
def get_none_array(classes=None,confounds=None):
	if classes is not None and confounds is not None:
		assert(confounds.shape[1] == classes.shape[0])
	elif classes is None:
		classes = np.ones((confounds.shape[1],))
	elif confounds is None:
		confounds = np.ones((1,classes.shape[0]))
	else:
		raise Exception("Cannot have two null arrays input to get_none_array")
	has_none = np.zeros(classes.shape,dtype=bool)
	for i in range(confounds.shape[1]):
		if not has_none[i]: has_none[i] = is_nan(classes[i])
		for j in range(confounds.shape[0]):
			if not has_none[i]: has_none[i] = is_nan(confounds[j,i])
	return has_none


# Main function. Takes as input classes (as integers starting from 0 in a 1D
# numpy array) and confounds (as floats and strings, or just objects, in a
# 2D numpy array). plim is the max p-value, in a nonparametric statistical test,
# at which discretization stops and enough buckets have been reached. If recurse
# is set to True, this method calls itself recursively on excluded data, though
# this doesn't guarantee that the final p values for continuous covariates will
# be up to snuff.
# Method returns an array of logicals that selects a subset of the given data,
# also forcing equal ratios between each class.

def class_balance(classes,confounds,plim = 0.05,recurse=True,exclude_none=True,unique_classes = None):
	classes = np.array(classes)
	confounds = np.array(confounds)
	if len(confounds) == 0:
		confounds = np.ones((1,len(classes)),dtype=object)
	ff = {}
	if exclude_none:
		has_none = get_none_array(classes,confounds)
		confounds = confounds[:,~has_none]
		classes = classes[~has_none]
	else:
		has_none = get_none_array(classes=None,confounds=confounds)
		confounds[:,has_none] = "none"
		has_none = get_none_array(classes=classes,confounds=None)
		classes[has_none] = "none"
	assert(np.all([not is_nan(_) for _ in classes]))
	classes = np.array(classes)
	if unique_classes is None:
		try:
			unique_classes = np.unique(classes)
		except:
			print(classes)
			exit()
	elif isinstance(unique_classes,list):
		unique_classes = np.unique(unique_classes)
	if not np.all(sorted(unique_classes) == list(range(len(unique_classes)))):
		for i in range(len(classes)):
			for j in range(len(unique_classes)):
				if classes[i] == unique_classes[j]:
					classes[i] = j
					break
	n_buckets = [1 for x in range(confounds.shape[0])]
	# Used for bucketing purposes
	sorted_confounds = np.sort(confounds,axis=1)
	# Automatically marks strings as discrete, giving each its own bucket
	string_mapper = {}
	unique_strs = []
	for i in range(confounds.shape[0]):
		if len(confounds.shape) > 1 and confounds.shape[1] > 0 and isinstance(confounds[i,0],str):
			u = np.unique(confounds[i,:])
			unique_strs.append(u)
			n_buckets[i] = len(u)
	p_vals = [0 for x in range(confounds.shape[0])]
	selection = np.ones(classes.shape,dtype=bool)
	while min(p_vals) < plim and np.sum(selection) > 0:
		primed = get_prime_form(confounds,n_buckets, sorted_confounds)
		primed = np.prod(primed,axis=0,dtype=int)
		selection = get_class_selection(classes,
										primed,
										unique_classes=unique_classes)
		rr = list(range(confounds.shape[0]))
		random.shuffle(rr)
		for i in rr:
			if not isinstance(confounds[i,0],str):
				ts = [confounds[i,np.logical_and(selection, classes == j)] \
					for j in range(len(unique_classes))]
				# Makes sure there are at least five instances of 
				# each class remaining
				if np.any(list(map(lambda k: len(k) < 5, ts))):
					selection = np.zeros(classes.shape,dtype=bool)
					break				
				min_p,max_p = multi_mannwhitneyu(ts)
				p_vals[i] = min_p
				if p_vals[i] < plim:
					n_buckets[i] += 1
					break
			else:
				p_vals[i] = 1
	if np.sum(selection) > 40 and confounds[:,~selection].shape[1] > 0:
		recurse_selection = integrate_arrs(selection, class_balance(classes[~selection],confounds[:,~selection],plim = plim,exclude_none=False,unique_classes=unique_classes))
		selection = np.logical_or(selection , recurse_selection)
	if exclude_none:
		selection = integrate_arrs_none(~has_none,selection)
		assert(len(selection) == len(has_none))
		assert(np.sum(~has_none) == len(classes))
	return selection

def separate_set(selections,set_divisions = [0.5,0.5],IDs=None):
	assert(isinstance(set_divisions,list))
	set_divisions = [i/np.sum(set_divisions) for i in set_divisions]
	rr = list(range(len(selections)))
	random.shuffle(rr)
	if IDs is None:
		IDs = np.array(list(range(len(selections))))
	selections_ids = np.zeros(selections.shape,dtype=int)
	totals = list(range(len(set_divisions)))
	prime_hasher = {}
	for i in rr:
		if not selections[i]:
			continue
		is_none = IDs[i] == None or IDs[i] == "NULL"
		if not is_none and IDs[i] in prime_hasher:
			selections_ids[i] = prime_hasher[IDs[i]]
			totals[selections_ids[i] - 1] += 1
			continue
		for j in range(len(set_divisions)):
			if np.sum(totals) == 0 or \
				totals[j] / np.sum(totals) < set_divisions[j]:
				break
		selections_ids[i] = j+1
		totals[j] += 1
		if not is_none and IDs[i] not in prime_hasher:
			prime_hasher[IDs[i]] = j + 1
	return selections_ids

def nifti_to_np(nifti_filepath,X_dim):
	nifti_file = nb.load(nifti_filepath)
	nifti_data = nifti_file.get_fdata()
	nifti_data -= nifti_data.min()
	m = nifti_data.max()
	nifti_data = nifti_data / m
	nifti_data = nifti_data.astype(np.float32)
	zp = [X_dim[i]/nifti_data.shape[i] \
		for i in range(len(X_dim))]
	nifti_data_zoomed = ndimage.zoom(nifti_data,zp)
	return nifti_data_zoomed

def str_to_list(s,nospace=False):
	if s is None or s == "": return []
	if s[0] == "[" and  s[-1] == "]":
		s = s[1:-1]
		s = s.replace("'","").replace("_","").replace("-","")
		if nospace:s=s.replace(" ","")
		s = s.split(",")
		if nospace and "" in s: s.remove("")
		return s
	else:
		return [s]

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def parsedate(d,date_format="%Y-%m-%d %H:%M:%S",verbose=True):
	try:
		return datetime.datetime.strptime(
			d.replace("_"," ").split(".")[0],
			date_format)
	except ValueError:
		return parser.parse(d.replace("_"," "))

def is_float(N):
	try:
		float(N)
		return True
	except:
		return False

def is_list_str(s):
	if is_nan(s): return False
	return (s[0] == "[" and s[-1] == "]")

def list_to_str(val):
	val = str(sorted(val))
	val = val.upper()
	val = val.replace(" ","_")
	val = val.replace("-","_")
	return val

def is_nan(k,inc_null_str=False):
	if k is None:
		return True
	if inc_null_str and isinstance(k,str):
		if k.lower() == "null" or k.lower() == "unknown":
			return True
	try:
		if np.isnan(k):
			return True
		else:
			return False
	except:
		if k == np.nan:
			return True
		else:
			return False

def get_multilabel_acc(y_pred,Y):
	y_pred = np.squeeze(y_pred.cpu().detach().numpy())
	Y = np.squeeze(Y.cpu().detach().numpy())
	#acc = np.argmax(np.squeeze(y_pred),axis=1) == np.argmax(Y,axis=1),axis=1)
	return (y_pred > 0.5) == (Y > 0.5)
	#return acc

def bucketize(arr,n_buckets):
	non_arr_list = []
	max_ = -np.Inf
	min_ = np.Inf
	for i in range(len(arr)):
		if not is_nan(arr[i]):
			if isinstance(arr[i],str): return arr
			non_arr_list.append(arr[i])
			if arr[i] > max_: max_ = arr[i]
			if arr[i] < min_: min_ = arr[i]
	bucketized_list = np.array(["NaN" for i in range(len(arr))],
			dtype=np.dtype(object))
	non_arr_list = sorted(non_arr_list)
	skips = int(len(non_arr_list)/float(n_buckets)) + 1
	buckets = np.array(non_arr_list[::skips])
	range_dist=((np.arange(n_buckets)/float(n_buckets-1))*(max_-min_))+min_
	while len(buckets) < n_buckets:
		print(buckets)
		buckets = np.array([buckets[0]] + list(buckets))
	buckets = (range_dist + buckets) / 2
	for i in range(len(arr)):
		if not is_nan(arr[i]):
			for j in range(len(buckets)-1):
				if arr[i] > buckets[j] and \
						arr[i] <= buckets[j+1]:
					bucketized_list[i] = str(j)
					break
	return bucketized_list

# Method to be implemented for determining whether or not a partition of a given
# distance array is random
import networkx as nx

def label_to_community(labels):
	npu = np.unique(labels)
	set_arr = {}
	for i,l in enumerate(labels):
		if l not in set_arr:
			set_arr[l] = set()
		set_arr[l].add(i)
	return [set_arr[_] for _ in set_arr]

def determine_random_partition2(arr2d,labels):
	assert(isinstance(arr2d,np.ndarray))
	assert(isinstance(labels,np.ndarray))
	G = nx.from_numpy_array(arr2d)
	true_modularity = nx.community.modularity(G,label_to_community(labels))
	return true_modularity
	random_modularities = []
	n_labels = np.sum(labels)
	rand_labels = copy(labels)
	for i in range(200):
		random.shuffle(rand_labels)
		random_modularities.append(
			nx.community.modularity(G,label_to_community(rand_labels))
		)
	random_modularities = np.sort(random_modularities)
	index = np.searchsorted(random_modularities,true_modularity)
	r = float(index) / (len(random_modularities) + 1)
	print(r)
	return r

def mod_meas(arr2d,labels):
	inner = 0
	inner_c = 0
	outer = 0
	outer_c = 0
	for i in range(arr2d.shape[0]):
		for j in range(i+1,arr2d.shape[1]):
			if labels[i] and labels[j]:
				inner += arr2d[j,i]
				inner_c += 1
			elif (labels[i] and not labels[j]) or (labels[j] and not labels[i]):
				outer += arr2d[j,i]
				outer_c += 1
	if inner_c == 0 or outer_c == 0: return 0
	inner = float(inner) / inner_c
	outer = float(outer) / outer_c
	return (outer - inner) #** 2

	
def determine_random_partition(arr2d,labels):
	assert(isinstance(arr2d,np.ndarray))
	assert(isinstance(labels,np.ndarray))
	true_modularity = mod_meas(arr2d,labels)
	random_modularities = []
	n_labels = np.sum(labels)
	rand_labels = np.zeros((labels.shape),dtype=bool)
	rand_labels[:n_labels] = True
	for i in range(200):
		random.shuffle(rand_labels)
		random_modularities.append(
			mod_meas(arr2d,rand_labels)
		)
	random_modularities = np.sort(random_modularities)
	index = np.searchsorted(random_modularities,true_modularity)
	r = float(index) / (len(random_modularities) + 1)
	return r



# Given the filenames (or, rather, filestubs), returns encoded input and output
# labels, as well as encoded confounds, if specified, as either a set of strings
# or binary arrays
def get_data_from_filenames(filename_list,test_variable=None,confounds=None,
		return_as_strs = False,unique_test_vals = None,database=None,
		return_choice_arr = False,dict_obj=None,return_as_dict=False,
		key_to_filename = None,X_encoder=None,vae_encoder=False,uniques=None,
		density_confound_sort=True,n_buckets=3):
	if dict_obj is not None:
		if "uniques" in dict_obj:
			uniques = dict_obj["uniques"]
	if database is None and test_variable is not None:
		database = pd.read_pickle(pandas_output)
	if key_to_filename is not None:
		X_filenames_list = [key_to_filename(_) for _ in filename_list]
	else:
		X_filenames_list = filename_list
	selection = np.array([os.path.isfile(_) for _ in X_filenames_list],
			dtype=bool)
	if confounds is not None:
		confound_strs = [[None for _ in confounds] \
					for __ in filename_list]
	if isinstance(test_variable,str):
		test_variable = [test_variable]
	if test_variable is not None:
		Y_strs = [[None for _ in filename_list] for __ in test_variable]
	if X_encoder is None:
		X = np.zeros((len(filename_list),
			image_dim[0],image_dim[1],image_dim[2]))
	else:
		X = None
	for i in range(len(filename_list)):
		if selection[i] == 0: continue
		f = X_filenames_list[i]
		f_key = filename_list[i]
		assert(os.path.isfile(f))
		try:
			X_single = np.load(f)
			if X_encoder is not None:
				X_single = torch.tensor(X_single)
				X_single = torch.unsqueeze(X_single,0)
				X_single = torch.unsqueeze(X_single,0).cuda(0)
				if vae_encoder:
					y, z_mean, z_log_sigma = X_encoder(X_single)
					X_single = z_mean + z_log_sigma.exp()*X_encoder.epsilon
				else:
					X_single = X_encoder(X_single)
				X_single = X_single.cpu().detach().numpy()
				X_single = np.squeeze(X_single)
		except:
			print("Error")
			selection[i] = 0
			continue
		if X is None:
			X = np.zeros(
				(len(filename_list),
				X_single.shape[1],
				X_single.shape[2]))
		X[i,...] = X_single
		if test_variable is not None:
			for j,t in enumerate(test_variable):
				Y_strs[j][i] = str_to_list(database.loc[f_key,t],nospace=True)
			if confounds is not None:
				for j,c in enumerate(confounds):
					confound_strs[i][j] = database.loc[f_key,c]
	filename_list = list(np.array(filename_list)[selection])
	X_filenames_list = list(np.array(X_filenames_list)[selection])
	X = X[selection,...]
	if test_variable is None:
		return X
	Y_strs = list(np.array(Y_strs)[:,selection])
	if return_as_strs:
		if confounds is not None:
			return X_filenames_list,Y_strs,confound_strs
		else:
			return X_filenames_list,Y_strs
	
	Y = []
	for j,t in enumerate(test_variable):
		mlb = MultiLabelBinarizer()
		if unique_test_vals is not None:
			mlb.fit([unique_test_vals])
		else:
			Y_strs_all = [[] for _ in test_variable]
			for s in database.loc[:,t]:
				if not is_nan(s):
					Y_strs_all[j].append(str_to_list(s,nospace=True))
				else:
					Y_strs_all[j].append(['None'])
			mlb.fit(Y_strs_all[j])
		Y.append(mlb.transform(Y_strs[j]))
	max_dim = np.max([_.shape[1] for _ in Y])
	Y = np.array(
			[np.concatenate(
					(_,np.zeros((_.shape[0],max_dim - _.shape[1]))),
				axis=1) for _ in Y]
		)
	Y = np.swapaxes(Y,0,1)
	if confounds is not None:
		if uniques is None or np.any([c not in uniques for c in confounds]):
			uniques = {}
			for c in confounds:
				uniques[c] = {}
				lis = list(database.loc[:,c])
				if np.any([isinstance(_,str) for _ in lis]):
					uniques[c]["discrete"] = True
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
					if density_confound_sort:
						u = sorted(
									[(_,u[_]) for _ in u],
									key=lambda k: k[1],
									reverse=True)
						u = [_[0] for _ in u]
					else:
						u = sorted(list(u))
					uniques[c]["unique"] = u
					n_buckets = max(n_buckets,len(u))
				else:
					uniques[c]["discrete"] = False
					max_ = -np.inf
					min_ = np.inf
					nonnan_list = []
					for l in lis:
						if not is_nan(l):
							max_ = max(max_,l)
							min_ = min(min_,l)
							nonnan_list.append(l)
					uniques[c]["max"] = max_
					uniques[c]["min"] = min_
					uniques[c]["nonnan_list"] = sorted(nonnan_list)
			for c in confounds:
				if not uniques[c]["discrete"]:
					n_buckets_cont = min(n_buckets,10)
					skips = int(len(uniques[c]["nonnan_list"])/\
						float(n_buckets_cont)) + 1
					uniques[c]["nonnan_list"] = \
						uniques[c]["nonnan_list"][::skips]
					# Get mean between density and range dists
					if True:
						max_ = uniques[c]["max"]
						min_ = uniques[c]["min"]
						rd = np.arange(n_buckets_cont)
						rd = rd / float(n_buckets_cont-1)
						rd = list((rd * (max_ - min_)) + min_)
						uniques[c]["nonnan_list"] = \
							[(rd[i] + \
							uniques[c]["nonnan_list"][i])/2 \
							for i in range(n_buckets_cont)]
						uniques[c]["nonnan_list"][-1] = max_
						uniques[c]["nonnan_list"][0] = min_
					assert(len(uniques[c]["nonnan_list"]) == \
						n_buckets_cont)
		confound_encode = np.zeros((len(filename_list),len(confounds),
				n_buckets + 1))
		if return_choice_arr:
			choice_arr = np.zeros((1,len(confounds),n_buckets + 1))
			for i in range(choice_arr.shape[1]):
				choice_arr[:,i,-1] = 1
				c = confounds[i]
				if uniques[c]["discrete"]:
					c_uniques = uniques[c]["unique"]
					for j in range(len(c_uniques)):
							choice_arr[:,i,j] = 1
				else:
					choice_arr[:,i,:] = 1
		for j in range(len(confounds)):
			c = confounds[j]
			if uniques[c]["discrete"]:
				c_uniques = uniques[c]["unique"]
				for i in range(len(filename_list)):
					if is_nan(confound_strs[i][j]):
						confound_encode[i,j,-1] = 1
					else:
						confound_encode[i,j,
							c_uniques.index(confound_strs[i][j])]=1
			else:
				max_ = uniques[c]["max"]
				min_ = uniques[c]["min"]
				for i in range(len(filename_list)):
					if is_nan(confound_strs[i][j]):
						confound_encode[i,j,-1] = 1
					else:
						unnl = uniques[c]["nonnan_list"]
						for kk in range(len(unnl)-1):
							if unnl[kk] <= confound_strs[i][j] and \
								unnl[kk+1] >= confound_strs[i][j]:
								confound_encode[i,j,kk]=1
								break
		try:
			assert(np.all(np.sum(confound_encode,axis=2) == 1))
		except:
			print(np.sum(confound_encode,axis=2))
			print("Assertion failed")
			print(confound_encode)
			exit()
		if return_as_dict:
			obj = {}
			obj["X"] = X
			obj["Y"] = Y
			obj["confound_encode"] = confound_encode
			if "choice_arr" in locals():
				obj["choice_arr"] = choice_arr
			obj["classes"] = list(mlb.classes_)
			obj["uniques"] = uniques
			return obj
		elif return_choice_arr:
			return X,Y,confound_encode,choice_arr,list(mlb.classes_),uniques
		else:
			return X,Y,confound_encode,uniques
	else:
		return X,Y

from copy import deepcopy as copy

def recompute_selection_ratios(selection_ratios,selection_limits,N):
	new_selection_ratios = copy(selection_ratios)
	assert(np.any(np.isinf(selection_limits)))
	variable = [True for i in range(len(selection_ratios))]

	for i in range(len(selection_ratios)):
		if selection_ratios[i] * N > selection_limits[i]:
			new_selection_ratios[i] = selection_limits[i] / N
			variable[i] = False
		else:
			new_selection_ratios[i] = selection_ratios[i]
	vsum = 0.0
	nvsum = 0.0
	for i in range(len(selection_ratios)):
		if variable[i]: vsum += new_selection_ratios[i]
		else: nvsum += new_selection_ratios[i]
	assert(nvsum < 1)
	for i in range(len(selection_ratios)):
		if variable[i]:
			new_selection_ratios[i] = \
				(new_selection_ratios[i] / vsum) * (1 - nvsum)
	return new_selection_ratios

def get_balanced_filename_list(test_variable,confounds_array,
		selection_ratios = [0.66,0.16,0.16],
		selection_limits = [np.Inf,np.Inf,np.Inf],value_ranges = [],
		output_selection_savepath = None,test_value_ranges=None,
		get_all_test_set=False,total_size_limit=None,
		verbose=False,non_confound_value_ranges = {},database = None,
		n_buckets = 10,patient_id_key=None):
	if len(value_ranges) == 0:
		value_ranges = [None for _ in confounds_array]
	assert(len(value_ranges) == len(confounds_array))
	
	covars_df = database
	if verbose: print("len(covars): %d" % len(covars_df))
	value_selection = np.ones((len(covars_df),),dtype=bool)
	for ncv in non_confound_value_ranges:
		if ncv in confounds_array:
			print("confounds_array: %s" % str(confounds_array))
			print("non_confound_value_ranges: %s" % \
				str(non_confound_value_ranges))
			print("ncv: %s" % str(ncv))
		assert(ncv not in confounds_array)
		confounds_array.append(ncv)
		value_ranges.append(non_confound_value_ranges[ncv])
	confounds_array.append(test_variable)
	value_ranges.append(test_value_ranges)
	if verbose: print("confounds_array: %s" % str(confounds_array))
	if verbose: print("value_ranges: %s" % str(value_ranges))
	for i in range(len(confounds_array)):
		temp_value_selection = np.zeros((len(covars_df),),dtype=bool)
		c = covars_df[confounds_array[i]]
		value_range = value_ranges[i]
		if value_range is None:
			continue
		if isinstance(value_range,tuple):
			for j in range(len(c)):
				if c[j] is None:
					continue
				if c[j] >= value_range[0] and\
						 c[j] <= value_range[1]:
					temp_value_selection[j] = True
		elif callable(value_range):
			for j in range(len(c)):
				if c[j] is None:
					continue
				if value_range(c[j]):
					temp_value_selection[j] = True
		else:
			for j in range(len(c)):
				if c[j] is None:
					continue
				if c[j] in value_range:
					temp_value_selection[j] = True	
		value_selection = np.logical_and(value_selection,
					temp_value_selection)
	del confounds_array[-1]
	del value_ranges[-1]
	for ncv in non_confound_value_ranges:
		del confounds_array[-1]
		del value_ranges[-1]
	if verbose:
		print("value_selection.sum(): %s"%str(value_selection.sum()))
	if verbose:
		print("value_selection.shape: %s"%str(value_selection.shape))
	covars_df = covars_df[value_selection]
	covars_df = covars_df.sample(frac=1)
	test_vars = covars_df[test_variable].to_numpy(dtype=np.dtype(object))
	# If it's a string array, it just returns strings
	test_vars = bucketize(test_vars,n_buckets)
	ccc = {}
	if output_selection_savepath is not None and \
			os.path.isfile(output_selection_savepath):
		selection = np.load(output_selection_savepath)
	else:
		
		if len(confounds_array) == 0:
			if verbose: print(test_value_ranges)
			selection = class_balance(test_vars,[],
				unique_classes=test_value_ranges,plim=0.1)
		else:
			selection = class_balance(test_vars,
				covars_df[confounds_array].to_numpy(\
					dtype=np.dtype(object)).T,
				unique_classes=test_value_ranges,plim=0.1)
		selection_ratios = recompute_selection_ratios(selection_ratios,
			selection_limits,np.sum(selection))
		if total_size_limit is not None:
			select_sum = selection.sum()
			rr = list(range(len(selection)))
			for i in rr:
				if select_sum <= total_size_limit:
					break
				if selection[i]:
					selection[i] = 0
					select_sum -= 1
		if patient_id_key is None:
			selection = separate_set(selection,selection_ratios)
		else:
			selection = separate_set(selection,selection_ratios,
				covars_df[patient_id_key].to_numpy(dtype=\
				np.dtype(object)).T)
		if output_selection_savepath is not None:
			np.save(output_selection_savepath,selection)
	all_files = (covars_df.index.values)
	if get_all_test_set:
		selection[selection == 0] = 2
	X_files = [all_files[selection == i] \
			for i in range(1,len(selection_ratios) + 1)]
	Y_files = [test_vars[selection == i] \
			for i in range(1,len(selection_ratios) + 1)]
	if verbose: print(np.sum([len(x) for x in X_files]))
	for i in range(len(X_files)):
		rr = list(range(len(X_files[i])))
		random.shuffle(rr)
		X_files[i] = X_files[i][rr]
		Y_files[i] = Y_files[i][rr]
	return X_files,Y_files

def YC_conv(Y,C,y_weight):
	Y = np.reshape(Y,(Y.shape[0],1,Y.shape[1]))
	Y_ = Y
	for j in range(y_weight-1):
		Y_ = np.concatenate((Y_,Y),axis=1)
	Y = Y_
	Y = np.concatenate((Y,np.zeros((Y.shape[0],
		Y.shape[1],C.shape[2]-Y.shape[2]))),axis=2)
	YC = np.concatenate((Y,C),axis=1)
	C_dud = np.zeros(C.shape)
	C_dud[:,:,0] = 1
	YC_dud = np.concatenate((Y,C_dud),axis=1)
	return YC,YC_dud

# Legacy code from the previous version of this script. Will likely use this in
# the future to save models.

def parsedate(d,date_format="%Y-%m-%d %H:%M:%S"):
	for match in datefinder.find_dates(d.replace("_"," ")): return match
	return datetime.datetime.strptime(d.split(".")[0],date_format)

def validate_database(database,args):
	for c in args.confounds:
		if c not in database.columns:
			raise Exception("Confound %s not in columns of %s"\
				%(c,args.var_file))
	
	if args.label not in database.columns:
		raise Exception("Label %s not in columns of %s"\
			%(args.label,args.var_file))
	
	for index in database.index:
		if os.path.splitext(index)[1] != ".npy":
			raise Exception(("Indices of %s must all be .npy files: "+\
				"exception at index %s") % (args.var_file,index))

def hidden_batch_predictions(
			X,
			model,
			group_vars,
			last_icd,
			last_hidden_var,
			ensemble=False,
			device=None
		):
	assert(X.size()[0] == len(group_vars))
	batch_size = X.size()[0]
	hidden_size = last_hidden_var.size()[1]
	if ensemble:
		ensemble_size = last_hidden_var.size()[2]
	if ensemble:
		if device is None:
			hidden_batch = torch.zeros((batch_size,1,hidden_size,ensemble_size))
		else:
			hidden_batch = torch.zeros(
					(batch_size,1,hidden_size,ensemble_size)
				).cuda(device)
	else:
		if device is None:
			hidden_batch = torch.zeros((batch_size,1,hidden_size))
		else:
			hidden_batch = torch.zeros((batch_size,1,hidden_size)).cuda(device)
	preds = None
	for i in range(batch_size):
		hidden_batch[i,:] = last_hidden_var
		if (i == 0 and group_vars[i] == last_icd) or \
				(i > 0 and group_vars[i-1] != group_vars[i]):
			if ensemble:
				if device is None:
					last_hidden_var = torch.zeros(
							(1,1,hidden_size,ensemble_size)
						)
				else: last_hidden_var = torch.zeros((1,1,hidden_size,ensemble_size)).cuda(device)
			else:
				if device is None:
					last_hidden_var = torch.zeros((1,1,hidden_size))
				else: last_hidden_var = torch.zeros(
										(1,1,hidden_size)).cuda(device)
		#print(last_hidden_var.size())
		if (ensemble and len(last_hidden_var.size()) == 3) or\
			 ((not ensemble) and len(last_hidden_var.size()) == 2):
			last_hidden_var = torch.unsqueeze(last_hidden_var,0)
		with torch.no_grad():
			out,last_hidden_var = model(torch.unsqueeze(X[i,...],0),
				last_hidden_var)
		if preds is None:
			if ensemble: preds = torch.zeros((batch_size,
									out.size()[1],
									out.size()[2],
									out.size()[3]))
			else: preds = torch.zeros((batch_size,
									out.size()[1],
									out.size()[2]))
		preds[i,...] = out
	return preds,hidden_batch

## Outputs the test set evaluations

def output_test(
		args,
		test_val_ranges,
		output_results,
		test_predictions_file,
		mucran,
		database,
		X_files = None,
		return_Xfiles = False):
	pred   = None
	c_pred = None
	Y	  = None
	C	  = None
	cert   = None
	results = {}
	b = args.label
	batch_size = args.batch_size
	confounds = args.confounds
	y_weight = args.y_weight
	np.random.seed(0)
	if X_files is None:
		[X_files],_ = get_balanced_filename_list(b,[],
			selection_ratios=[1],
			total_size_limit=np.inf,
			non_confound_value_ranges = test_val_ranges,
			database=database)
	temp = X_files
	while len(X_files) > 0:
		X_,Y_,C_,choice_arr,output_labels = get_data_from_filenames(
					X_files[:batch_size],
					b,confounds=confounds,
					return_as_strs = False,
					unique_test_vals = None,
					return_choice_arr=True,
					database=database)
		YC_pred = mucran.predict(X_)
		pred_ = np.mean(YC_pred[:,:y_weight,:],axis=1)
		c_pred_ = YC_pred[:,y_weight:,:]
		cert_ = None
		####
		if Y is None:
			X = X_
			Y = Y_
			C = C_
			pred = pred_
			c_pred = c_pred_
			cert = cert_
		else:
			pred   = np.concatenate((pred,pred_),	 axis=0)
			c_pred = np.concatenate((c_pred,c_pred_), axis=0)
			Y	  = np.concatenate((Y,Y_),		   axis=0)
			C	  = np.concatenate((C,C_),		   axis=0)
		X_files = X_files[batch_size:]
	X_files = temp
	save_dict = {}
	for i in range(Y.shape[0]):
		X_file = X_files[i]
		save_dict[X_file] = [[float(_) for _ in pred[i,:]],
			[float(_) for _ in Y[i,:]]]
	json.dump(save_dict,open(test_predictions_file,'w'),indent=4)
	pred_bin = np.zeros(Y.shape)
	Y_bin = np.zeros(Y.shape)
	for i in range(pred_bin.shape[0]):
		pred_bin[i,np.argmax(pred[i,:Y.shape[1]])] = 1
		Y_bin[i,np.argmax(Y[i,:])] = 1
	roc_aucs = []
	print("Y AUROCS")
	results[b] = {}
	print("Y.shape: %s" % str(Y.shape))
	print("pred_bin.shape: %s" % str(pred_bin.shape))
	results[b]["Y_acc"] = \
		float(np.mean(np.all(Y == pred_bin,axis=1)))
	for i in range(Y.shape[1]):
		fpr, tpr, threshold = roc_curve(Y[:,i],pred[:,i])
		roc_auc = auc(fpr, tpr)
		roc_aucs.append(roc_auc)
		print("%s AUROC: %s" % (output_labels[i],str(roc_auc)))
		results[b][output_labels[i]] = float(roc_auc)
	results[b]["Mean AUROC"] = float(np.mean(roc_aucs))
	
	print("Mean AUROC: % s" % str(np.mean(roc_aucs)))
	print("Y acc: %f" % results[b]["Y_acc"])
	print("+++")
	print("MAX CONFOUND AUROCS")
	for i in range(len(confounds)):
		confound = confounds[i]
		roc_aucs = []
		roc_aucs_counts = []
		for j in range(C.shape[2]):
			if np.any(C[:,i,j] == 1):
				fpr, tpr, threshold = roc_curve(C[:,i,j],c_pred[:,i,j])
				roc_auc = auc(fpr, tpr)
				if not is_nan(roc_auc):
					roc_aucs.append(roc_auc)
					roc_aucs_counts.append(int(np.sum(C[:,i,j])))
		weighted_mean = np.sum(
				[c1*c2 for c1,c2 in zip(roc_aucs,roc_aucs_counts)]) /\
				 np.sum(roc_aucs_counts)
		try:
			results[confound] = {}
			if len(roc_aucs) == 0:
				print("No AUCs for %s" % confound)
			else:
				mroc = int(np.argmax(roc_aucs))
				meanroc = np.mean(roc_aucs)
				print(("%s: %f (max); %d (num in max) ;"+\
					" %f (mean); %f (weighted mean)") \
					% (confound,roc_aucs_counts[mroc],roc_aucs[mroc],meanroc,
					weighted_mean))
				results[confound]["MAX AUROC"] = float(roc_aucs[mroc])
				results[confound]["NUM IN MAX"] = float(roc_aucs_counts[mroc])
				results[confound]["MEAN AUROC"] = float(meanroc)
				results[confound]["WEIGHTED MEAN"] = float(weighted_mean)
		except:
			print("Error in outputting %s" % confound)

	json.dump(results,open(output_results,'w'),indent=4)
	if return_Xfiles: return X_files


def tensor2im(input_image, imtype=np.uint8):
	""""Converts a Tensor array into a numpy image array.

	Parameters:
		input_image (tensor) --  the input image tensor array
		imtype (type)		--  the desired type of the converted numpy array
	"""
	if not isinstance(input_image, np.ndarray):
		if isinstance(input_image, torch.Tensor): # get the data from a variable
			image_tensor = input_image.data
		else:
			return input_image
		# convert it into a numpy array
		image_numpy = image_tensor[0].cpu().float().numpy()
		if image_numpy.shape[0] == 1:  # grayscale to RGB
			image_numpy = np.tile(image_numpy, (3, 1, 1))
		# post-processing: tranpose and scaling
		image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	else:
		# if it is a numpy array, do nothing
		image_numpy = input_image
	return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
	"""Calculate and print the mean of average absolute(gradients)

	Parameters:
		net (torch network) -- Torch network
		name (str) -- the name of the network
	"""
	mean = 0.0
	count = 0
	for param in net.parameters():
		if param.grad is not None:
			mean += torch.mean(torch.abs(param.grad.data))
			count += 1
	if count > 0:
		mean = mean / count
	print(name)
	print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
	"""Save a numpy image to the disk

	Args:
		image_numpy (numpy array) -- input numpy array
		image_path (str)		  -- the path of the image
	"""

	image_pil = Image.fromarray(image_numpy)
	h, w, _ = image_numpy.shape

	if aspect_ratio > 1.0:
		image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
	if aspect_ratio < 1.0:
		image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
	image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
	"""Print the mean, min, max, median, std, and size of a numpy array

	Args:
		val (bool) -- if print the values of the numpy array
		shp (bool) -- if print the shape of the numpy array
	"""
	x = x.astype(np.float64)
	if shp:
		print('shape,', x.shape)
	if val:
		x = x.flatten()
		print(('mean = %3.3f, min = %3.3f, "+\
			"max = %3.3f, median = %3.3f, std=%3.3f') % (
			np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

# https://stackoverflow.com/questions/26685067/represent-a-hash-string-to-binary-in-python
# Two functions to encode strings as binary arrays
def text_to_bin(text, n_bin=32,d=512):
	"""Encodes strings as binary arrays"""
	if text is None: text=""
	text=text.lower()
	word_ord = '{}'.format(
			bin(int(hashlib.md5(text.encode('utf-8')).hexdigest(), n_bin))
		)
	word_ord = word_ord[2:]
	arr = []
	for i in range(d):
		a = word_ord[i % len(word_ord)]
		if a == "1":
			arr.append(1.0)
		elif a == "0":
			arr.append(0.0)
		else:
			raise Exception("%s is bad"% str(a))
	return arr

def encode_static_inputs(static_input,d=512):
	"""
	"""
	arr = np.zeros((len(static_input),d))
	for i in range(len(static_input)):
		arr[i,:] = text_to_bin(static_input[i],d=d)
	return arr
