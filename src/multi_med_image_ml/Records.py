import functools,os
import numpy as np
from monai.transforms import *
from .utils import *
import gc
import psutil

generate_transforms = Compose([
		RandAffine(prob=0.5, translate_range=10), 
		RandRotate(prob=0.5, range_x=10.0),
		RandGaussianNoise(prob=0.5),
		RandBiasField(prob=0.5)]
)
#,
#		RandSmoothDeform(spatial_size=(96,96,96),rand_size=(96,96,96))
#])

@functools.total_ordering
class ImageRecord:
	"""A class used to represent an abstraction of an image for MedImageLoader.
	
	ImageRecord is used to keep and organize a given image in main memory.
	The same image may be represented on the file system as a nifti, dicom,
	or an npy file, which caches the file at a particular size. This reads
	in the file without creating duplicates. The image may also be cleared
	or read in in real time to avoid having the images take up too much 
	space in main memory.
	
	Attributes:
		filename (str): Filename of the image
		database (str): Object used to quickly look up metadata associated with the image (default None)
		dtype (str): Type of output (either "torch" or "numpy") (default "torch")
		extra_info_list (list): 
		dim (tuple): Standard dimension that the image will be resized to upon returning it (default (96,96,96))
		Y_dim (tuple): A tuple indicating the dimension of the image's label. The first number is the number of labels associated with the image and the second is the number of choices that has. Extra choices will not affect the model but fewer will throw an error — thus, if Y_dim is (1,2) and the label has three classes, it will crash. But (1,4) will just result in an output that is always zero. This should match the Y_dim parameter in the associated MultiInputModule (default (1,32))
		C_dim (tuple): A tuple indicating the dimension of the image's confounds. This effectively operates the same way as Y_dim, except the default number of confounds is higher (default (16,32))
		image (Numpy array): Variable containing the actual image, at size dim. It may be None, to save memory (default None)
		Y (Numpy array): Variable containing the encoding of the image label(s), at size Y_dim
		C (Numpy array): Variable containing the encoding of the image confound(s), at size C_dim
		y_on_c (bool): If true, replicates the Y array on the bottom of all C arrays. Used for regression training. C_dim must to large enough to accommodate the extra Y array or it will crash. (default True)
		times_called (int): Counter to count the number of times get_image is called (default 0)
		static_inputs (list): A list of values that may be called to put into the model as text
		cache (bool): If true, caches the image file as a .npy array. Takes up extra space but it's recommended. (default True)
		cached_record (bool): Path of the cached record
		npy_file (str): Path of the cached .npy record
		exam_date (datetime): Date that the image was taken, if it can be read in from the database/dicom records (default None)
		bdate (datetime): Birth date of the patient, if it can be read in from the database/dicom records (default None)
		json_file (str): File name of the json that results from a DICOM being converted to nifti (default None)
	"""
	
	def __init__(self,
		filename: str,
		database = None,
		dim : tuple = (96,96,96),
		dtype : str ="torch",
		extra_info_list : list = None,
		y_on_c : bool = True,
		cache : bool = True,
		Y_dim : tuple = (1,32),
		C_dim : tuple = (16,32),
		y_nums : list = None,
		c_nums : list = None,
		static_inputs=[]):
		
		self.dim=dim
		self.filename = filename
		self.npy_file = get_dim_str(self.filename,self.dim)
		self.image_type = None
		self.dtype = dtype
		self.database = database
		self.Y_dim = Y_dim
		self.C_dim = C_dim
		self.y_on_c = y_on_c # Adds the Y value to the C array as well
		
		self.is_loaded = False			
		self.group_by = None
		self.bdate = None
		self.exam_date = datetime.datetime(year=1970,month=1,day=1)
		self.extra_info_list = None
		self.json_file = None
		
		self.image = None
		self.Y = None
		self.C = None
		self.y_nums = y_nums
		self.c_nums = c_nums
		
		self.static_inputs = []
		self.static_input_res = None
		
		self.loaded = False
		if self.database is not None:
			if self.npy_file in self.database.database:
				self.load_extra_info()
		
		self.times_called = 0
		self.cache = cache
		self.cached_record = None
	def get_exam_date(self):
		if not self.loaded:
			self.load_extra_info()
		return self.exam_date
	def get_birth_date(self):
		if not self.loaded:
			self.load_extra_info()
		return self.bdate
	def load_extra_info(self):
		if self.loaded: return
		if self.database is None:
			return
		if not self.database.has_im(self):
			return
		self.bdate = self.database.get_birth_date(self.npy_file)
		self.exam_date = self.database.get_exam_date(self.npy_file)
		self.group_by = self.database.get_ID(self.npy_file)
		self.loaded = True
	def get_static_inputs(self):
		if self.static_input_res is None:
			self.static_input_res = []
			for key in self.static_inputs:
				static_inputs.append(
					self.database.loc_val(self.npy_file,key)
				)
		return self.static_input_res
	def _is_valid_operand(self, other):
		return hasattr(other, "exam_date") and hasattr(other,"group_by")
	def __eq__(self, other):
		if not self._is_valid_operand(other):
			return NotImplemented
		return (self.exam_date == other.exam_date)
	def __lt__(self, other):
		if not self._is_valid_operand(other):
			return NotImplemented
		return self.exam_date < other.exam_date
	def get_image_type(self):
		if self.image_type is None:
			if os.path.isdir(self.filename):
				self.image_type = "dicom_folder"
			elif os.path.isfile(self.filename):
				name,ext = os.path.splitext(self.filename)
				name = name.lower()
				ext = ext.lower()
				if ext == ".npy":
					self.image_type = "npy"
				elif ext  == ".nii":
					self.image_type = "nifti"
				elif ext == ".gz" and os.path.splitext(name)[1] == ".nii":
					self.image_type = "nifti"
				elif ext == ".dcm":
					self.image_type = "dicom"
				else:
					raise Exception(
					"Not implemented for extension %s" % ext)
			elif os.path.isfile(
				get_dim_str(self.filename,
						dim=self.dim,
						outtype='.nii.gz')):
				self.filename = get_dim_str(self.filename,
							dim=self.dim,
							outtype='.nii.gz')
				return self.get_image_type()
			elif os.path.isfile(
				get_dim_str(self.filename,
						dim=self.dim,
						outtype='.nii')):
				self.filename = get_dim_str(self.filename,
							dim=self.dim,
							outtype='.nii')
				return self.get_image_type()
			elif os.path.isdir(get_dim_str(self.filename,
							dim=self.dim,
							outtype='dicom')):
				self.filename = get_dim_str(self.filename,
							dim=self.dim,
							outtype='dicom')
				return self.get_image_type()
		return self.image_type
	def get_mem(self):
		if self.image is None:
			return 0
		#else:
		#	return np.prod(self.image.shape) * \
		#		np.dtype(self.image.dtype).itemsize
		elif self.dtype == "torch":
			return self.image.element_size() * self.image.nelement()
		elif self.dtype == "numpy":
			return np.prod(self.image.shape) * \
				np.dtype(self.image.dtype).itemsize
		#else:
		#	raise Exception("Unimplemented dtype: %s" % self.dtype)
	def clear_image(self):
		del self.image
		del self.C
		del self.Y
		self.image = None
		self.Y = None
		self.C = None
	def read_image(self):
		if self.get_image_type() == "dicom_folder":
			self.filename,self.json_file = compile_dicom(self.filename,
				db_builder=self.database)
			self.image_type = None
		self.cached_record = get_dim_str(self.filename,self.dim)
		if self.cache and os.path.isfile(self.cached_record):
			self.image = np.load(self.cached_record)
		elif self.get_image_type() == "nifti":
			self.image = nb.load(self.filename).get_fdata()
		elif self.get_image_type() == "npy":
			self.image = np.load(self.filename)
		elif self.get_image_type() == "dicom":
			self.image = dicom.dcmread(self.filename).pixel_array
		else:
			print("Error in %s" % self.filename)
			print("Error in %s" % self.npy_file)
			raise Exception("Unsupported image type: %s"%self.get_image_type())
		self.image = resize_np(self.image,self.dim)
		if self.cache and not os.path.isfile(self.cached_record):
			np.save(self.cached_record,self.image)
		if self.database is not None:
			self.database.add_json(nifti_file=self.filename)
		if self.dtype == "torch":
			self.image = torch.tensor(self.image)
	def get_image(self,augment=False):
		"""Reads in and returns the image, with the option to augment"""
		
		if self.image is None:
			self.read_image()
		self.load_extra_info()
		self.times_called += 1
		if augment and self.dtype == "torch":
			return generate_transforms(self.image)
		else: return self.image
	def _get_Y(self):
		"""Returns label"""
		
		if self.y_nums is not None:
			return self.y_nums
		self.y_nums = self.database.get_label_encode(self.npy_file)
		return self.y_nums
	def _get_C(self):
		"""Returns confound array"""
		
		if self.c_nums is not None:
			return self.c_nums
		self.c_nums = self.database.get_confound_encode(self.npy_file)
		return self.c_nums
	def get_Y(self):
		if self.Y is not None:
			return self.Y
		y_nums = self._get_Y()
		if self.dtype == "numpy":
			self.Y = np.zeros(self.Y_dim)
		elif self.dtype == "torch":
			self.Y = torch.zeros(self.Y_dim)
		for i,j in enumerate(y_nums):
			self.Y[i,j] = 1
		return self.Y
	def get_C(self):
		if self.C is not None:
			return self.C
		c_nums = self._get_C()
		if self.dtype == "numpy":
			self.C = np.zeros(self.C_dim)
		elif self.dtype == "torch":
			self.C = torch.zeros(self.C_dim)
		for i,j in enumerate(c_nums):
			self.C[i,j] = 1
		if self.y_on_c:
			y_nums = self._get_Y()
			for i,j in enumerate(y_nums):
				self.C[i+len(self.database.confounds),j] = 1
		return self.C
	def get_C_dud(self):
		"""Returns an array of duds with the same dimensionality as C
		
		Returns an array of duds with the same dimensionality as C but with all
		values set to the first choice. Used in training the regressor. If
		y_on_c is set to True, this replicates the Y array on the bottom rows of
		the array."""
		
		if self.dtype == "numpy":
			C_dud = np.zeros(self.C_dim)
			C_dud[:len(self.database.confounds),0] = 1
		elif self.dtype == "torch":
			C_dud = torch.zeros(self.C_dim)
			C_dud[:len(self.database.confounds),0] = 1
		if self.y_on_c:
			y_nums = self._get_Y()
			for i,j in enumerate(y_nums):
				C_dud[i+len(self.database.confounds),j] = 1
		return C_dud
		
class BatchRecord():
	def __init__(self,
			image_records,
			dtype="torch",
			device=None,
			sort=True,
			batch_by_pid=False,
			channels_first = True,
			gpu_ids="",
			batch_size = 14):
		self.image_records = image_records
		assert(
			np.all(
				[isinstance(image_record,ImageRecord) \
					for image_record in image_records
				]
				)
			)
		self.gpu_ids = gpu_ids
		self.channels_first = channels_first
		self.batch_by_pid=batch_by_pid
		if sort: self.image_records = sorted(image_records)
		#assert(
		#	np.all(
		#		[_.group_by == self.image_records[0].group_by \
		#			for _ in self.image_records]
		#	)
		#)
		self.device=device
		self.dtype=dtype
		self.batch_size = batch_size
	#def process_to_cuda(self,device):
	#	self.Y = torch.tensor(self.Y).float().cuda(device)
	#	self.Y = torch.unsqueeze(self.Y,1)
	#	self.Y = self.Y[-1,...]
	#	
	#	self.X = torch.tensor(self.X)
	#	self.X = torch.unsqueeze(self.X,1).float().cuda(device)
	#	
	#	self.C = torch.tensor(self.C).float().cuda(device)
	#	self.C_dud = torch.tensor(self.C_dud).float().cuda(device)
	
	def name(self):
		return 'BatchRecord'
	def _get(self,callback,augment=False):
		Xs = []
		no_arr = False
		
		for i,im in enumerate([self.image_records[-1]] if (callback == "Y" and self.batch_by_pid) \
			else self.image_records):
			if i >= self.batch_size: break
			if callback == "X":
				X = im.get_image(augment=augment)
				if self.dtype == "torch":
					assert(len(X.size()) == 3)
					if self.channels_first:
						X = torch.unsqueeze(X,0)
					else:
						X = torch.unsqueeze(X,-1)
				elif self.dtype == "numpy":
					assert(len(X.shape) == 3)
					if self.channels_first:
						X = np.expand_dims(X,axis=0)
					else:
						X = np.expand_dims(X,axis=-1)
				else:
					raise Exception("Invalid dtype: %s" % self.dtype)
			elif callback == "Y":
				X = im.get_Y()
			elif callback == "C":
				X = im.get_C()
			elif callback == "C_dud":
				X = im.get_C_dud()
			elif callback == "birth_dates":
				X = im.get_birth_date()
				no_arr = True
			elif callback == "exam_dates":
				X = im.get_exam_date()
				no_arr = True
			elif callback == "static_inputs":
				X = im.get_static_inputs()
				no_arr = True
			else:
				raise Exception("Invalid callback")
			if no_arr:
				Xs.append(X)
			elif self.dtype == "torch":
				Xs.append(torch.unsqueeze(X,0))
			elif self.dtype == "numpy":
				Xs.append(np.expand_dims(X,axis=0))
			else:
				raise Exception("Invalid dtype")
		if no_arr:
			return Xs
		elif self.dtype == "torch":
			Xs = torch.concatenate(Xs,0)
			if self.gpu_ids == "":
				return Xs.float()
			else:
				return Xs.float().cuda(self.gpu_ids[0])
		elif self.dtype == "numpy":
			Xs = np.concatenate(Xs,axis=0)
			return Xs #.astype(np.float32)
		else:
			raise Exception("Invalid dtype: %s" % self.dtype)
	def get_image(self,augment=False):
		return self._get("X",augment=augment)
	def get_Y(self):
		return self._get("Y")
	def get_C(self):
		return self._get("C")
	def get_C_dud(self):
		return self._get("C_dud")
	def get_exam_dates(self):
		return self._get("exam_dates")
	def get_birth_dates(self):
		return self._get("birth_dates")
	def get_static_inputs(self):
		return self._get("static_inputs")

# Used for memory management purposes
class AllRecords():
	def __init__(self,dim):
		self.image_dict = {}
		self.mem_limit = psutil.virtual_memory().available * 0.2
		self.cur_mem = 0
		temp = torch.zeros(dim).float()
		self.obj_size = temp.element_size() * temp.nelement()
		del temp
		gc.collect()
	def add(self,filename: str, im: ImageRecord):
		self.image_dict[filename] = im
		if self.obj_size is None and im.image is not None:
			self.obj_size = im.get_mem()
	def has(self,filename: str):
		return filename in self.image_dict
	def get(self,filename: str):
		return self.image_dict[filename]
	def clear_images(self):
		for filename in self.image_dict:
			self.image_dict[filename].clear_image()
		gc.collect()
	def get_mem(self):
		if self.obj_size is None: return 0
		n_images = 0
		for filename in self.image_dict:
			if self.image_dict[filename] is not None:
				n_images += 1
		return n_images * self.obj_size
	def check_mem(self):
		if True or self.get_mem() < self.mem_limit:
			self.clear_images()
