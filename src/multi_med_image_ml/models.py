import torch.nn as nn
from torch.autograd import Variable
import torch
import math
import random
import itertools
import numpy as np
from datetime import datetime
import os,sys
from copy import deepcopy as copy
from .Records import ImageRecord,BatchRecord
from .utils import download_weights,text_to_bin,encode_static_inputs

# Three functions that are used to get the age encoding functions
def time_index(i,pos,d=512,c=10000):
	if i % 2 == 0:
		v = math.sin(pos/(c**(2*i/d)))
	else:
		v = math.cos(pos/(c**(2*i/d)))
	return v

def get_age_arr(age,max_age=120.0,d=512):
	arr = np.zeros((d,))
	pos = int(age / max_age * 1000)
	for i in range(d):
		arr[i] = time_index(i,pos,d)
	return arr

def get_age_encoding(date,birthdate,d=512):
	if date is None or birthdate is None:
		return np.zeros((d,0))
	age = date.year - birthdate.year
	return get_age_arr(age,d=d)

class Reshape(nn.Module):
	'''
		Used in a nn.Sequential pipeline to reshape on the fly.
	'''
	def __init__(self, *target_shape):
		super().__init__()
		self.target_shape = target_shape
	
	def forward(self, x):
		return x.view(*self.target_shape)

class Encoder(nn.Module):
	def __init__(self,latent_dim=512):
		super(Encoder,self).__init__()
		nchan=1
		base_feat = 64
		self.encoder = nn.Sequential(
			nn.Conv3d(in_channels = nchan, out_channels = base_feat, stride=2,
				kernel_size=5, padding = 2), #1*96*96*96 -> 64*48*48*48
			nn.LeakyReLU(),
			#nn.InstanceNorm3d(base_feat),
			nn.Conv3d(in_channels = base_feat, out_channels = base_feat*2,
				stride=2, kernel_size = 5,
				padding=2), #64*48*48*48 -> 128*24*24*24
			nn.LeakyReLU(),
			nn.InstanceNorm3d(base_feat*2),
			nn.Conv3d(in_channels = base_feat*2, out_channels = base_feat*4,
				stride=2,kernel_size = 3,
				padding=1), #128*24*24*24 -> 256*12*12*12
			nn.LeakyReLU(),
			#nn.InstanceNorm3d(base_feat*4),
			nn.Conv3d(in_channels = base_feat*4, out_channels = base_feat*4,
				stride=4,kernel_size = 5,padding=2), #256*12*12*12 -> 256*3*3*3
			nn.LeakyReLU(),
			nn.InstanceNorm3d(base_feat*4),
			nn.Conv3d(in_channels = base_feat*4, out_channels = base_feat*32,
				stride=1,kernel_size = 3,padding=0), #256*3*3*3 -> 2048*1*1*1
			nn.LeakyReLU(),
			Reshape([-1,base_feat*32]),
			nn.Linear(in_features = base_feat*32, out_features = base_feat*16),
			nn.LeakyReLU(),
			nn.Linear(in_features = base_feat*16, out_features = latent_dim),
		)
	def parameters(self):
		return self.encoder.parameters()
	def forward(self, x):
		x = self.encoder(x)
		return x


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		nchan=1
		base_feat = 64
		latent_dim = 512	
		self.decoder = nn.Sequential(
			nn.Linear(in_features = latent_dim, out_features = base_feat*16),
			nn.ReLU(),
			nn.BatchNorm1d(base_feat*16),
			nn.Linear(in_features = base_feat*16, out_features = base_feat*32),
			nn.ReLU(),
			Reshape([-1,base_feat*32,1,1,1]),
			nn.ConvTranspose3d(in_channels = base_feat*32,
				out_channels = base_feat*16, kernel_size = 3,stride=1,
				padding=0), #2048*1*1*1 -> 1024*3*3*3
			nn.ReLU(),
			nn.BatchNorm3d(base_feat*16),
			nn.ConvTranspose3d(in_channels = base_feat*16,
				out_channels = base_feat*4, kernel_size = 4,stride=2, padding=1,
				bias=False), #256*3*3*3 -> 256*6*6*6
			nn.ReLU(),
			nn.BatchNorm3d(base_feat*4),
			nn.ConvTranspose3d(in_channels = base_feat*4,
				out_channels = base_feat*4, kernel_size = 4,stride=2, padding=1,
				bias=False), #256*6*6*6 -> 256*12*12*12
			nn.ReLU(),
			nn.BatchNorm3d(base_feat*4),
			nn.ConvTranspose3d(in_channels = base_feat*4,
				out_channels = base_feat*2, kernel_size = 4,stride=2, padding=1,
				bias=False), #256*12*12*12 -> 128*24*24*24
			nn.ReLU(),
			nn.BatchNorm3d(base_feat*2),
			nn.ConvTranspose3d(in_channels = base_feat*2,
				out_channels = base_feat, kernel_size = 4, stride=2, padding=1,
				bias=False), #128*24*24*24 -> 64*48*48*48
			nn.ReLU(),
			nn.BatchNorm3d(base_feat),
			nn.ConvTranspose3d(in_channels = base_feat, out_channels = 1,
				kernel_size = 4,stride=2,padding=1,
				bias=False), #64*48*48*48 -> 1*96*96*96
			nn.Sigmoid()
		)
	def forward(self, x):
		x = self.decoder(x)
		return x

class Regressor(nn.Module):
	def __init__(self,latent_dim,n_confounds,n_choices,device='cpu'):
		super(Regressor,self).__init__()
		base_feat = 64
		n = 4
		self.regressor_set = []
		for _ in range(n_choices):
			self.regressor_set.append(
				nn.Sequential(
					nn.Linear(latent_dim,base_feat*n),
					nn.LeakyReLU(),
					Reshape([-1,n,base_feat]),
					nn.InstanceNorm1d(n,affine=True),
					Reshape([-1,n*base_feat]),
					
					nn.Linear(base_feat*n,base_feat*n),
					nn.LeakyReLU(),
					Reshape([-1,n,base_feat]),
					nn.InstanceNorm1d(n,affine=True),
					Reshape([-1,n*base_feat]),
						
					nn.Linear(base_feat*n,base_feat*n),
					nn.LeakyReLU(),
					Reshape([-1,n,base_feat]),
					nn.InstanceNorm1d(n,affine=True),
					Reshape([-1,n*base_feat]),
			
					nn.Linear(n*base_feat,n_confounds),
					#nn.ReLU(),
					Reshape([-1,n_confounds,1]),
					nn.Sigmoid()
				)
			)
	def parameters(self):
		return itertools.chain(*[r.parameters() \
			for r in self.regressor_set])
	def cuda(self,device):
		for r in self.regressor_set:
			r.cuda(device)
	def cpu(self):
		for r in self.regressor_set:
			r.cpu()
	def state_dict(self,*args,**kwargs):
		state_dict1 = {}
		for i,r in enumerate(self.regressor_set):
			state_dict1["regressor.%d" % i] = r.state_dict()
		return state_dict1
	def load_state_dict(self,state_dict,*args,**kwargs):
		for i,r in enumerate(self.regressor_set):
			r.load_state_dict(state_dict["regressor.%d" % i])
		return
	def forward(self,x):
		return torch.cat([r(x) for r in self.regressor_set],2)
		
class Classifier(nn.Module):
	def __init__(self,latent_dim,n_inputs,base_feat,n_out,n_labels):
		super(Classifier,self).__init__()
		
		self.classifier = nn.Sequential(
			nn.Linear(latent_dim*n_inputs,n_inputs*base_feat*4),
			nn.LeakyReLU(),
			Reshape([-1,n_inputs,base_feat*4]),
			nn.InstanceNorm1d(n_inputs,affine=True),
			Reshape([-1,1,n_inputs*base_feat*4]),

			nn.Linear(in_features = n_inputs*base_feat*4,
				out_features = n_inputs*base_feat*2),
			nn.LeakyReLU(),
			Reshape([-1,n_inputs,base_feat*2]),
			nn.InstanceNorm1d(n_inputs,affine=True),
			Reshape([-1,1,n_inputs*base_feat*2]),

			nn.Linear(in_features = n_inputs*base_feat*2,
				out_features = n_out*n_labels),
			Reshape([-1,n_out,n_labels]),
			nn.Sigmoid(),	
		)
	def parameters(self):
		return self.classifier.parameters()
	def forward(self,x):
		return self.classifier(x)

class MultiInputModule(nn.Module):
	"""Takes variable imaging and non-imaging data and outputs a prediction
	
	Can take multiple, variable-sized images and static text inputs as input 
	and output a label prediction, while also regressing confounds.
	
	Attributes:
		encoder (nn.Module): Encodes input images to latent array
		classifier (nn.Module): Takes multiple images encoded by the encoder and combines them into a single predictive value
		regressor (nn.Module): Optional network that regresses confounds from the encoder's latent representation using adversarial regression
		Y_dim (tuple): A tuple indicating the dimension of the image's label. The first number is the number of labels associated with the image and the second is the number of choices that has. Extra choices will not affect the model but fewer will throw an error — thus, if Y_dim is (1,2) and the label has three classes, it will crash. But (1,4) will just result in an output that is always zero. This should match the Y_dim parameter in the associated Records class (default (1,32))
		C_dim (tuple): A tuple indicating the dimension of the image's confounds. This effectively operates the same way as Y_dim, except the default number of confounds is higher (default (16,32))
		n_dyn_inputs (int): The maximum number of images that can be passed in (default 14)
		n_stat_inputs (int): The maximum number of text-based static inputs that can be input into the model (default 2)
		encode_age (bool): Encode the age of the patient on individual images prior to being input into the classifier (default False)
		device (torch.device): GPU/CPU that the module is on (default: torch.device('cpu'))
		weights (str): Pretrained weight indicator. Weights automatically download if this is set. Default options must be in place or results are unpredictable. (default None)
		latent_dim (int): Size of the intermediary representation that the encoder outputs and inputs into the classifier (default 128)
		variational (bool): Turns the encoding into a variational setup, a la a variational autoencoder, in which the encoding is sampled from a Gaussian distribution rather than a set array of numbers (default False)

		remove_uncertain (bool): UNIMPLEMENTED/UNTESTED. Experimental subroutine designed to remove from consideration encoded images that are a certain "distance" from the training set (default False)
		use_attn (bool): UNIMPLEMENTED/UNTESTED. Adds an attention mechanism to the classifier (default False)
		num_training_samples (int): Number of training samples to sample for the uncertainty removal mechanism (default 300)
		static_record (set): Set of static keys put into the model during training, to prevent unrecognized keys from being input during testing
	"""
	
	def __init__(self,
				Y_dim: tuple = (1,32), # Number of labels, Number of choices
				C_dim: tuple = (16,32), # Number of labels, Number of choices
				n_dyn_inputs: int = 14,
				n_stat_inputs: int = 2,
				use_attn: bool = False,
				encode_age: bool = False,
				variational: bool = False, # Makes it a variational encoder
				zero_input: bool = False, # Repeats input into classifier or makes it zeros
				remove_uncertain: bool = False,
				device = torch.device('cpu'),
				latent_dim: int = 128,
				weights: str = None,
				grad_layer: int = 7):
		super(MultiInputModule,self).__init__()
		
		# Model Parameters
		self.latent_dim = latent_dim
		
		# Number of non-image, patient-specific demographic inputs. Sex and 
		# ethnicity are two that can be applied.
		self.n_stat_inputs = n_stat_inputs
		
		# Max number of images that can be passed in
		self.n_dyn_inputs = n_dyn_inputs
		
		# Total number of inputs, which is used for the classifier model
		self.n_inputs = self.n_stat_inputs + self.n_dyn_inputs
		
		if (not isinstance(Y_dim,tuple)) or (len(Y_dim) != 2) or\
			(not isinstance(C_dim,tuple)) or (len(C_dim) != 2):
			raise Exception("""
				Y_dim and C_dim must be tuples.
				(<# labels>,<max # choices>)
				One label with two choices: (1,2)
			""")
		self.Y_dim = Y_dim
		self.C_dim = C_dim
		self.zero_input = zero_input
		self.remove_uncertain = remove_uncertain
		self.grad_layer = grad_layer

		# Under development
		if self.remove_uncertain:
			self.record_training_sample = False
			self.num_training_samples = 300
			self.training_sample = torch.zeros(
				(
					self.latent_dim,
					self.num_training_samples
				),
				device=device)

		# Training options
		self.use_attn = use_attn
		self.encode_age = encode_age
		self.static_dropout = True # Randomly mask static inputs in training
		self.variational = variational
		
		# A record that prevents unrecognized keys from being applied during
		# the test phase
		self.static_record = [set() for _ in range(self.n_stat_inputs)]
		
		# Sets the multiplier for the number of features in each model component
		base_feat = 64
		# Modules
		
		# Makes the encoder output a variational latent space, so it's a
		# Gaussian distribution.
		if self.variational:
			self.encoder = Encoder(latent_dim=self.latent_dim)
			self.z_mean = nn.Sequential(
				nn.Linear(self.latent_dim,self.latent_dim)
			)
			self.z_log_sigma = nn.Sequential(
				nn.Linear(self.latent_dim,self.latent_dim)
			)
			self.epsilon = torch.distributions.Normal(0, 1)
			self.epsilon.loc = self.epsilon.loc
			self.epsilon.scale = self.epsilon.scale
		else:
			self.encoder = Encoder(latent_dim=self.latent_dim)
		# The output of the classifier and regressor, and encoder are kept
		# consistent, to 16 max outputs and 32 possible choices. This makes
		# cross-training easier, though it's less efficient.
		self.classifier = Classifier(latent_dim = self.latent_dim,
										n_inputs = self.n_inputs,
										base_feat = base_feat,
										n_out = self.Y_dim[0],#16,
										n_labels = self.Y_dim[1])#32)
		if self.C_dim is not None:
			n_confounds,n_choices = self.C_dim
			self.regressor = Regressor(self.latent_dim,
				n_confounds=self.C_dim[0],
				n_choices=self.C_dim[1],
				device=device)
		else: self.regressor = None
		
		if weights is not None:
			if self.C_dim != (16,32) or self.Y_dim != (1,32):
				warnings.warn(
					"Pretrained models may not function if defaults are altered"
					)
			if os.path.isfile(weights) and \
				os.path.splitext(weights)[1] == ".pt":
				self.load_state_dict(torch.load(weights))
			else:
				self.load_state_dict(torch.load(download_weights(weights)))
		
		# Placeholder for gradients for Grad-CAM analysis
		self.gradients = None
		
	# hook for the gradients of the activations
	def activations_hook(self, grad):
		self.gradients = grad
	
	# method for the gradient extraction
	def get_activations_gradient(self):
		return self.gradients
	
	# method for the activation extraction
	def get_activations(self, x):
		return self.encoder.encoder[:self.grad_layer](x)
	
	def load_state_dict(self,state_dict,*args, **kwargs):
		if self.regressor is not None:
			self.regressor.load_state_dict(state_dict['regressor'])
		super().load_state_dict(state_dict,*args,**kwargs)
		return
		
	def state_dict(self,*args,**kwargs):
		state_dict1 = super().state_dict(*args, **kwargs)
		if self.regressor is not None:
			state_dict1.update({'regressor':self.regressor.state_dict()})
		return state_dict1
		
	def forward_ensemble(self,kwargs,n_ens=10):
		x = []
		for i in range(n_ens):
			x.append(self(**kwargs))
		return x

	def cuda(self,device):
		self.device=device
		if self.variational:
			self.epsilon.loc = self.epsilon.loc.cuda(device)
			self.epsilon.scale = self.epsilon.scale.cuda(device)
		self.regressor.cuda(device)
		return super().cuda(device)
		
	def cpu(self):
		self.device = torch.device('cpu')
		if self.variational:
			self.epsilon.loc = self.epsilon.loc.cpu()
			self.epsilon.scale = self.epsilon.scale.cpu()
		self.regressor.cpu()
		return super().cpu()
	
	def regressor_freeze(self):
		for param in self.classifier_parameters():
			param.requires_grad = True
		for param in self.regressor.parameters():
			param.requires_grad = False
			
	def classifier_freeze(self):
		for param in self.classifier_parameters():
			param.requires_grad = False
		for param in self.regressor.parameters():
			param.requires_grad = True

	def classifier_parameters(self):
		if self.variational:
			return itertools.chain(self.encoder.parameters(),
				self.classifier.parameters(),
				self.z_log_sigma.parameters(),
				self.z_mean.parameters())
		else:
			return itertools.chain(self.encoder.parameters(),
				self.classifier.parameters())

	def forward(self,
				x,
				static_input=None,
				dates=None,
				bdate=None,
				return_regress = False,
				return_encoded = False,
				encoded_input = False,
				grad_eval = False,
				record_encoding = False):
		"""Puts image or BatchRecord through model and predicts a value.
		
		Args:
			x (torch.Tensor or BatchRecord): Image or BatchRecord that contains data to be predicted
			static_input (list): List of text to be input into model
			dates (list[datetime.datetime]): List of dates input in the model, when a BatchRecord is not input
			bdate (datetime.datetime): Patient birthdate, when a BatchRecord is not input
			return_regress (bool): If True, returns the confound prediction array as a second value (default False)
			return encoded (bool): If True, returns the encoded values of the images (default False)
			encoded_input (bool): Indicator that X is input that's already been encoded and can be put straight into the classifier (default False)
			record_encoding (bool): If set, saves the most recent encoding as a numpy file in the variable saved_encoding
		"""
		use_regression = hasattr(self,'regressor') and \
			(self.regressor is not None) and \
			(self.training or return_regress)
		if not encoded_input:				
				#assert(self.gradients is not None)
			if isinstance(x,BatchRecord):
				assert(x.dtype == "torch")
				if len(x.get_static_inputs()[0]) > 0:
					static_inputs = [_[0] for _ in x.get_static_inputs()]
					if self.n_stat_inputs != len(static_inputs):
						raise Exception(
					"Number of static inputs not equal to input: %d != %d"\
						 % (len(static_inputs),self.n_stat_inputs))
				dates = x.get_exam_dates()
				bdates = x.get_birth_dates()
				bdate = None
				for i,b in enumerate(bdates):
					if b is not None:
						bdate = b
				x = x.get_X(augment=self.training)
				assert(torch.is_tensor(x))
				
			if (self.encode_age and (dates is None or bdate is None)):
				raise Exception("Need dates as input to encode age")
			if x.size(0) > self.n_dyn_inputs:
				raise Exception(
					"Max dynamic inputs is %d. Received %d. Reduce batch size." % \
						(
							self.n_dyn_inputs,
							int(x.size(0))
						)
					)
			if static_input is not None and len(static_input) > self.n_stat_inputs:
				raise Exception(
					"Max dynamic inputs is %d. Received %d. Reduce batch size." % \
						(
							self.n_stat_inputs,
							len(static_input)
						)
					)
			# Encode everything - separate batches
			if self.variational:
				if grad_eval:
					x = self.encoder.encoder[:self.grad_layer](x)
					h = x.register_hook(self.activations_hook)
					x = self.encoder.encoder[self.grad_layer:](x)
				else:
					x = self.encoder(x)
				z_mean = self.z_mean(x)
				z_log_sigma = self.z_log_sigma(x)
				x = z_mean + (z_log_sigma.exp()*self.epsilon.sample(z_mean.shape))
				self.kl = (z_mean**2 + z_log_sigma.exp()**2 - z_log_sigma-0.5).mean()
			else:
				if grad_eval:
					x = self.encoder.encoder[:self.grad_layer](x)
					h = x.register_hook(self.activations_hook)
					x = self.encoder.encoder[self.grad_layer:](x)
				else:
					x = self.encoder(x)
			if record_encoding:
				self.saved_encoding = x.cpu().detach().numpy()
			if hasattr(self,'remove_uncertain'):
				if self.remove_uncertain:
					if self.record_training_sample:
						self.training_sample[:,
							self.training_i:min(self.training_i + x.shape[0],
							self.num_training_samples)] = x
						self.training_i += x.shape[0]
						if self.training_i >= self.num_training_samples:
							self.record_training_sample = False
				
			if return_encoded:
				return x
		if use_regression:
			reg = self.regressor(x)
			# Encode dynamic inputs with dates using positional encoding
		if (self.encode_age) or \
			(self.training and self.static_dropout and \
				random.choice([True,False])):
			age_encodings = []
			if dates is not None:
				for i,date in enumerate(dates):
					age_encoding = get_age_encoding(
									date,
									bdate,
									d=self.latent_dim)
					age_encodings.append(age_encoding)
				age_encodings = np.array(age_encodings)
				age_encodings =  torch.tensor(
									age_encodings,
									device=x.device
									).float()
				x = torch.add(x,age_encodings)

		x = torch.unsqueeze(x,0)
		# Pad encodings with zeros, depending on input size
		e_size = list(x.size())
		e_size[1] = self.n_inputs - e_size[1] #[1-16]*512 -> 16*512
		if (not hasattr(self,'zero_input')) or self.zero_input:
			x = torch.cat((x,torch.zeros(e_size,device=x.device)),axis=1)
		else:
			x_ = torch.clone(x)
			while x_.size()[1] < e_size[1]:
				x_ = torch.cat((x_,torch.clone(x_)),axis=1)
			x_ = x_[:,:e_size[1],:]
			x = torch.cat((x,x_),axis=1)
			#x = x.repeat(1,(e_size[1]*2 // x.size()[1]),1)[:,:e_size[1],:]
		
		# Place static inputs near end
		if static_input is not None:
			if len(static_input) != self.n_stat_inputs:
				raise Exception(
						"Received %d static inputs, but it is set at %d" % \
						(len(static_input),self.n_stat_inputs)
					)
			if self.training:
				for i,e in enumerate(static_input):
					self.static_record[i].add(e)
			else:
				for i,e in enumerate(self.static_record):
					if static_input[i] not in e:
						raise Exception("Input %s not a previous demographic input (previous inputs were %s)" % (static_input[i],str(e)))
			x_ = encode_static_inputs(static_input,d=self.latent_dim)
			x_ = torch.tensor(x_,device = x.device)
			x_ = torch.unsqueeze(x_,0)
			for i in range(x_.shape[0]):
				if ((not self.static_dropout) or random.choice([True,False]) and self.training) or\
					(not self.static_dropout):
					x[:,(-(self.n_stat_inputs) + i):,:] = x_[i,:]
		
		# Randomly order dynamic encodings
		r = list(range(self.n_inputs))
		r_ = r[:self.n_stat_inputs]
		random.shuffle(r_)
		r[:self.n_stat_inputs] = r_
		x = x[:,r,...]
		
		# Apply attention mask
		if self.use_attn:
			m = torch.cat(
					(
						torch.zeros(
							list(x.size())[:],
							device=x.device,
							dtype=torch.bool),
						torch.ones(
							e_size[:],
							device=x.device,
							dtype=torch.bool)
					),
				axis=1)
			m = m[:,r,...]
			x,_ = self.multihead_attn(x,x,x,need_weights=False)#,attn_mask=m)

		# Switch batch channel with layer channel prior to running classifier
		x = torch.unsqueeze(x,-1)
		x = x.contiguous().view([-1,1,self.latent_dim*self.n_inputs]) # 16*512 -> 1*[16*512]
		x = self.classifier(x)
		if use_regression: return x,reg
		else: return x

class EnsembleModel(nn.Module):
	def __init__(self,model_list):
		super(EnsembleModel,self).__init__()
		self.models = [torch.load(_) for _ in model_list]
		for model in self.models: model.eval()
	def forward(self,input,hidden):
		new_output = []
		new_hidden = []
		for i in range(hidden.size()[3]):
			model = self.models[i]
			x = model.Encoder(input)
			x = torch.unsqueeze(x,1).float()
			output,h = model.RNN(x,hidden[...,i])
			new_output.append(torch.unsqueeze(output,3))
			new_hidden.append(torch.unsqueeze(h,3))
		new_output = torch.cat(new_output,dim=3)
		new_hidden = torch.cat(new_hidden,dim=3)
		return new_output,new_hidden

class Encoder1D(nn.Module):
	def __init__(self,input_dim,output_dim,conv=False):
		super(Encoder1D,self).__init__()
		base_feat = 1024
		if conv:
			self.encoder = nn.Sequential(
				Reshape([-1,1,1,input_dim]),
				nn.Conv2d(1,base_feat,kernel_size=(1,input_dim)),
				nn.LeakyReLU(),
				Reshape([-1,1,base_feat]),
				nn.BatchNorm1d(1,affine=True),
				Reshape([-1,base_feat]),
				nn.Linear(base_feat,output_dim),
				
			)
		else:
			self.encoder = nn.Sequential(
				nn.Linear(input_dim,base_feat),
				nn.LeakyReLU(),
				nn.BatchNorm1d(1,affine=True),

				nn.Linear(base_feat,base_feat),
				nn.LeakyReLU(),
				nn.BatchNorm1d(1,affine=True),

				nn.Linear(base_feat,base_feat),
				nn.LeakyReLU(),
				nn.BatchNorm1d(1,affine=True),

				nn.Linear(base_feat,input_dim//2),
				nn.LeakyReLU(),
				nn.BatchNorm1d(1,affine=True),

				nn.Linear(in_features = input_dim//2, out_features = input_dim//4),
				nn.LeakyReLU(),
				nn.BatchNorm1d(1,affine=True),
				
				nn.Linear(in_features = input_dim//4, out_features = input_dim//8),
				nn.LeakyReLU(),
				
				nn.Linear(in_features = input_dim//8, out_features = output_dim),
			)
	def forward(self,x):
		x = self.encoder(x)
		return x


class Decoder1D(nn.Module):
	def __init__(self,input_dim,output_dim,conv=False):
		super(Decoder1D, self).__init__()
		base_feat = 1024
		if conv:
			self.decoder = nn.Sequential(
				nn.Linear(output_dim,base_feat),
				nn.LeakyReLU(),
				Reshape([-1,1,base_feat]),
				nn.BatchNorm1d(1,affine=True),
				Reshape([-1,base_feat,1]),
				nn.ConvTranspose1d(base_feat,1,kernel_size=input_dim)
			)
		else:
			self.decoder = nn.Sequential(
				nn.Linear(in_features = output_dim,out_features = input_dim//8),

				nn.LeakyReLU(),
				nn.Linear(in_features = input_dim//8,out_features = input_dim//4),
				nn.LeakyReLU(),

				nn.BatchNorm1d(1,affine=True),
				nn.Linear(in_features = input_dim//4,out_features = base_feat),
				nn.LeakyReLU(),
					
				nn.Linear(base_feat,base_feat),
				nn.LeakyReLU(),
				nn.BatchNorm1d(1,affine=True),

				nn.Linear(base_feat,base_feat),
				nn.LeakyReLU(),
				nn.BatchNorm1d(1,affine=True),

				nn.Linear(in_features = base_feat,out_features = input_dim),
			)
	def forward(self,x):
		x = self.decoder(x)
		return x
		
class VAE(nn.Module):
	def __init__(self, input_dim,latent_dim=2,device=torch.device('cpu')):
		super(VAE, self).__init__()
		self.device = device
		self.latent_dim = latent_dim
		self.z_mean = nn.Linear(64, latent_dim)
		self.z_log_sigma = nn.Linear(64, latent_dim)
		#self.epsilon = torch.normal(size=(1, latent_dim), mean=0, std=1.0,
		#	device=self.device)
		self.epsilon = torch.distributions.Normal(0, 1)
		self.epsilon.loc = self.epsilon.loc.cuda(device)
		self.epsilon.scale = self.epsilon.scale.cuda(device)
		self.encoder = Encoder1D(input_dim,64)
		self.decoder = Decoder1D(input_dim,latent_dim)
		
	#	self.reset_parameters()
	  
	def reset_parameters(self):
		for weight in self.parameters():
			stdv = 1.0 / math.sqrt(weight.size(0))
			torch.nn.init.uniform_(weight, -stdv, stdv)

	def forward(self, x):
		x = self.encoder(x)
		x = torch.flatten(x, start_dim=1)
		z_mean = self.z_mean(x)
		z_log_sigma = self.z_log_sigma(x)
		z = z_mean + (z_log_sigma.exp()*self.epsilon.sample(z_mean.shape))
		#z = nn.functional.sigmoid(z)
		y = self.decoder(z)
		self.kl = (z_mean**2 + z_log_sigma.exp()**2 - z_log_sigma - 0.5).sum()
		return y,z, z_mean, z_log_sigma

class AutoEncoder1D(nn.Module):
	def __init__(self,input_dim,latent_dim=2,device='cpu'):
		super(AutoEncoder1D,self).__init__()
		
		self.encoder = Encoder1D(input_dim,latent_dim)#.cuda(device)
		self.decoder = Decoder1D(input_dim,latent_dim)#.cuda(device)
		
	def forward(self,x):
		latent = self.encoder(x)
		x = self.decoder(latent)
		return latent,x
