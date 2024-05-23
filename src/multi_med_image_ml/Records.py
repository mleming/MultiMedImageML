import functools,os
import numpy as np
from monai.transforms import *
from utils import *

@functools.total_ordering
class ImageRecord():
	def __init__(self,
		filename,
		all_vars = None,
		dim=(96,96,96),
		dtype="torch",
		ID = None,
		extra_info_list = None,
		y_on_c = True,
		cache = True,
		Y_dim = (32,32),
		C_dim = (32,32),
		y_nums = None,
		c_nums = None,
		static_inputs=[]):
		
		self.dim=dim
		self.filename = filename
		self.fkey = get_dim_str(self.filename,self.dim)
		self.image_type = None
		self.dtype = dtype
		self.all_vars = all_vars
		self.Y_dim = Y_dim
		self.C_dim = C_dim
		self.y_on_c = y_on_c # Adds the Y value to the C array as well
		
		self.is_loaded = False			
		self.group_by = None
		self.bdate = None
		self.exam_date = datetime.datetime(year=1970,month=1,day=1)
		self.ID = None
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
		if self.all_vars is not None:
			if self.fkey in self.all_vars.all_vars:
				self.load_extra_info()
		
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
		if self.all_vars is None:
			return
		if self.fkey not in self.all_vars.all_vars.index:
			return
		self.bdate = self.all_vars.get_birth_date(self.fkey)
		self.exam_date = self.all_vars.get_exam_date(self.fkey)
		self.group_by = self.all_vars.get_ID(self.fkey)
		self.loaded = True
	def get_static_inputs(self):
		if self.static_input_res is None:
			self.static_input_res = []
			for key in self.static_inputs:
				static_inputs.append(self.all_vars.all_vars.loc[self.fkey,key])
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
					raise Exception("Not implemented for extension %s" % ext)
		return self.image_type
	def get_mem(self):
		if self.image is None:
			return 0
		#else:
		#	return np.prod(self.image.shape) * \
		#		np.dtype(self.image.dtype).itemsize
		elif self.dtype == "torch":
			return self.image.element_size()
		elif self.dtype == "numpy":
			return np.prod(self.image.shape) * \
				np.dtype(self.image.dtype).itemsize
		#else:
		#	raise Exception("Unimplemented dtype: %s" % self.dtype)
	def clear_image(self):
		self.image = None
		self.X = None
		self.Y = None
		self.C = None
		#self.loaded = False
	def read_image(self):
		if self.get_image_type() == "dicom_folder":
			self.filename,self.json_file = compile_dicom(self.filename,
				db_builder=self.all_vars)
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
			raise Exception("Unsupported image type: %s"%self.get_image_type())
		self.image = resize_np(self.image,self.dim)
		if self.cache and not os.path.isfile(self.cached_record):
			np.save(self.cached_record,self.image)
		if self.all_vars is not None:
			self.all_vars.add_json(nifti_file=self.filename)
		if self.dtype == "torch":
			self.image = torch.tensor(self.image)
	def get_image(self):
		if self.image is None:
			self.read_image()
		self.load_extra_info()
		return self.image
	def _get_Y(self):
		"""Returns label"""
		if self.y_nums is not None:
			return self.y_nums
		self.y_nums = self.all_vars.get_label_encode(self.fkey)
		return self.y_nums
	def _get_C(self):
		"""Returns confound array"""
		if self.c_nums is not None:
			return self.c_nums
		self.c_nums = self.all_vars.get_confound_encode(self.fkey)
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
				self.C[i+len(self.all_vars.confounds),j] = 1
		return self.C
	def get_C_dud(self):
		if self.dtype == "numpy":
			C_dud = np.zeros(self.C_dim)
			C_dud[0,:len(self.all_vars.confounds)] = 1
		elif self.dtype == "torch":
			C_dud = torch.zeros(self.C_dim)
			C_dud[0,:len(self.all_vars.confounds)] = 1
		if self.y_on_c:
			y_nums = self._get_Y()
			for i,j in enumerate(y_nums):
				C_dud[i+len(self.all_vars.confounds),j] = 1
		return C_dud
		
class BatchRecord():
	def __init__(self,
			image_records,
			dtype="torch",
			device=None,
			sort=True,
			batch_by_pid=False,
			channels_first = True):
		self.image_records = image_records
		assert(
			np.all(
				[isinstance(image_record,ImageRecord) \
					for image_record in image_records
				]
				)
			)
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
	def process_to_cuda(self,device):
		self.Y = torch.tensor(self.Y).float().cuda(device)
		self.Y = torch.unsqueeze(self.Y,1)
		self.Y = self.Y[-1,...]
		
		self.X = torch.tensor(self.X)
		self.X = torch.unsqueeze(self.X,1).float().cuda(device)
		
		self.C = torch.tensor(self.C).float().cuda(device)
		self.C_dud = torch.tensor(self.C_dud).float().cuda(device)
	
	def name(self):
		return 'BatchRecord'
	def _get(self,callback):
		Xs = []
		no_arr = False
		for im in (self.image_records if (callback != "Y" and not self.batch_by_pid)\
			else [self.image_records[-1]]):
			if callback == "X":
				X = im.get_image()
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
			return Xs.float()
		elif self.dtype == "numpy":
			Xs = np.concatenate(Xs,axis=0)
			return Xs #.astype(np.float32)
		else:
			raise Exception("Invalid dtype: %s" % self.dtype)
	def get_image(self):
		return self._get("X")
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

generate_transforms = Compose([
		RandAffine(prob=0.5, translate_range=10), 
		RandRotate(prob=0.5, range_x=10.0),
		RandGaussianNoise(prob=0.5),
		RandBiasField(prob=0.5),
		RandSmoothDeform(spatial_size=(96,96,96),rand_size=(96,96,96))
])
