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
import warnings
from bisect import bisect

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES


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

def combine_covar(C1,C2,M1,M2,N1,N2):
	M_com = (N1 * M1 + M2 * N2) / (N1 + N2)
	N_com = N1 + N2
	C_com = (C1 * N1 + N1 * (M1 - M_com)**2 + \
			C2 * N2 + N2 * (M2 - M_com)** 2) / N_com
	return C_com,M_com,N_com


class PatchEmbeddings(nn.Module):
	"""
	Convert the image into patches and then project them into a vector space.
	"""

	def __init__(self,hidden_size=64):
		super().__init__()
		self.image_size = 96
		self.patch_size = 16
		self.num_channels = 1
		self.hidden_size = hidden_size 
		# Calculate the number of patches from the image size and patch size
		self.num_patches = (self.image_size // self.patch_size) ** 3
		# Create a projection layer to convert the image into patches
		# The layer projects each patch into a vector of size hidden_size
		self.projection = nn.Conv3d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

	def forward(self, x):
		# (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
		x = self.projection(x)
		x = x.flatten(start_dim=2,end_dim=-1).transpose(1, 2)
		
		return x

class Embeddings(nn.Module):
	"""
	Combine the patch embeddings with the class token and position embeddings.
	"""
		
	def __init__(self,hidden_size=64):
		super().__init__()
		self.hidden_size=hidden_size
		self.hidden_dropout_prob = 0.25
		self.patch_embeddings = PatchEmbeddings(hidden_size=self.hidden_size)
		# Create a learnable [CLS] token
		# Similar to BERT, the [CLS] token is added to the beginning of the input sequence
		# and is used to classify the entire sequence
		self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
		# Create position embeddings for the [CLS] token and the patch embeddings
		# Add 1 to the sequence length for the [CLS] token
		self.position_embeddings = \
			nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, self.hidden_size))
		self.dropout = nn.Dropout(self.hidden_dropout_prob)

	def forward(self, x):
		x = self.patch_embeddings(x)
		batch_size, _, _ = x.size()
		# Expand the [CLS] token to the batch size
		# (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
		cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		# Concatenate the [CLS] token to the beginning of the input sequence
		# This results in a sequence length of (num_patches + 1)
		x = torch.cat((cls_tokens, x), dim=1)
		x = x + self.position_embeddings
		x = self.dropout(x)
		return x

class AttentionHead(nn.Module):
	"""
	A single attention head.
	This module is used in the MultiHeadAttention module.
	"""
	def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
		super().__init__()
		self.hidden_size = hidden_size
		self.attention_head_size = attention_head_size
		# Create the query, key, and value projection layers
		self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
		self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
		self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		# Project the input into query, key, and value
		# The same input is used to generate the query, key, and value,
		# so it's usually called self-attention.
		# (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
		query = self.query(x)
		key = self.key(x)
		value = self.value(x)
		# Calculate the attention scores
		# softmax(Q*K.T/sqrt(head_size))*V
		attention_scores = torch.matmul(query, key.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		attention_probs = nn.functional.softmax(attention_scores, dim=-1)
		attention_probs = self.dropout(attention_probs)
		# Calculate the attention output
		attention_output = torch.matmul(attention_probs, value)
		return (attention_output, attention_probs)

class MultiHeadAttention(nn.Module):
	"""
	Multi-head attention module.
	This module is used in the TransformerViTEncoder module.
	"""

	def __init__(self,
			hidden_size=64,
			num_attention_heads=12,
			hidden_dropout_prob=0.0,
			attention_heads_dropout_prob=0.25):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_attention_heads = num_attention_heads #config["num_attention_heads"]
		self.hidden_dropout_prob = hidden_dropout_prob
		self.attention_heads_dropout_prob = attention_heads_dropout_prob
		# The attention head size is the hidden size divided by the number of attention heads
		self.attention_head_size = self.hidden_size // self.num_attention_heads
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		# Whether or not to use bias in the query, key, and value projection layers
		self.qkv_bias = True # config["qkv_bias"]
		# Create a list of attention heads
		self.heads = nn.ModuleList([])
		for _ in range(self.num_attention_heads):
			head = AttentionHead(
				self.hidden_size,
				self.attention_head_size,
				self.attention_heads_dropout_prob,
				self.qkv_bias
			)
			self.heads.append(head)
		# Create a linear layer to project the attention output back to the hidden size
		# In most cases, all_head_size and hidden_size are the same
		self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
		self.output_dropout = nn.Dropout(self.hidden_dropout_prob)

	def forward(self, x, output_attentions=False):
		# Calculate the attention output for each attention head
		attention_outputs = [head(x) for head in self.heads]
		# Concatenate the attention outputs from each attention head
		attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
		# Project the concatenated attention output back to the hidden size
		attention_output = self.output_projection(attention_output)
		attention_output = self.output_dropout(attention_output)
		# Return the attention output and the attention probabilities (optional)
		if not output_attentions:
			return (attention_output, None)
		else:
			attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
			return (attention_output, attention_probs)

class MLP(nn.Module):
	"""
	A multi-layer perceptron module.
	"""

	def __init__(self, hidden_size = 64,intermediate_size=1024,hidden_dropout_prob=0.25):
		super().__init__()
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.hidden_dropout_prob = hidden_dropout_prob
		self.dense_1 = nn.Linear(self.hidden_size, self.intermediate_size)
		#self.activation =  nn.functional.gelu() # NewGELUActivation()
		self.dense_2 = nn.Linear(self.intermediate_size, self.hidden_size)
		self.dropout = nn.Dropout(self.hidden_dropout_prob)

	def forward(self, x):
		x = self.dense_1(x)
		#x = self.activation(x)
		x = nn.functional.gelu(x)
		x = self.dense_2(x)
		x = self.dropout(x)
		return x

class Block(nn.Module):
	"""
	A single transformer block.
	"""

	def __init__(self,hidden_size=64):
		super().__init__()
		self.hidden_size=hidden_size
		self.attention = MultiHeadAttention(self.hidden_size)
		self.layernorm_1 = nn.LayerNorm(self.hidden_size)
		self.mlp = MLP(hidden_size=self.hidden_size)
		self.layernorm_2 = nn.LayerNorm(self.hidden_size)

	def forward(self, x, output_attentions=False):
		# Self-attention
		attention_output, attention_probs = \
			self.attention(self.layernorm_1(x), output_attentions=output_attentions)
		# Skip connection
		x = x + attention_output
		# Feed-forward network
		mlp_output = self.mlp(self.layernorm_2(x))
		# Skip connection
		x = x + mlp_output
		# Return the transformer block's output and the attention probabilities (optional)
		if not output_attentions:
			return (x, None)
		else:
			return (x, attention_probs)

class ViTEncoder(nn.Module):
	"""
	The transformer encoder module.
	"""

	def __init__(self, num_hidden_layers = 12, hidden_size=64):
		super().__init__()
		# Create a list of transformer blocks
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.blocks = nn.ModuleList([])
		for _ in range(self.num_hidden_layers):
			block = Block(hidden_size=self.hidden_size)
			self.blocks.append(block)

	def forward(self, x, output_attentions=False):
		# Calculate the transformer block's output for each block
		all_attentions = []
		for block in self.blocks:
			x, attention_probs = block(x, output_attentions=output_attentions)
			if output_attentions:
				all_attentions.append(attention_probs)
		# Return the encoder's output and the attention probabilities (optional)
		if not output_attentions:
			return (x, None)
		else:
			return (x, all_attentions)

class ViTForClassification(nn.Module):
	"""
	The ViT model for classification.
	"""

	def __init__(self,image_size=(96,96,96),hidden_size=1024,num_classes=128):
		super().__init__()
		self.image_size = image_size
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		# Create the embedding module
		self.embedding = Embeddings(hidden_size=self.hidden_size)
		# Create the transformer encoder module
		self.encoder = ViTEncoder(hidden_size=self.hidden_size)
		# Create a linear layer to project the encoder's output to the number of classes
		self.classifier = nn.Linear(self.hidden_size, self.num_classes)
		# Initialize the weights
		#self.apply(self._init_weights)

	def forward(self, x, output_attentions=False):
		# Calculate the embedding output
		embedding_output = self.embedding(x)
		# Calculate the encoder's output
		encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
		# Calculate the logits, take the [CLS] token's output as features for classification
		logits = self.classifier(encoder_output[:, 0])
		# Return the logits and the attention probabilities (optional)
		if not output_attentions:
			return logits #(logits, None)
		else:
			return (logits, all_attentions)



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
		self.regressor = nn.Sequential(
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
			
					nn.Linear(n*base_feat,n_choices),
					#nn.ReLU(),
					Reshape([-1,1,n_choices])
					#nn.Sigmoid()
				)
	def parameters(self):
		return self.regressor.parameters()
	def cuda(self,device):
		self.regressor.cuda(device)
	def cpu(self):
		self.regressor.cpu()
	def forward(self,x):
		x = self.regressor(x)
		#x = torch.nn.functional.softmax(x,dim=2)
		return x


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
			Reshape([-1,n_out,n_labels])
			#nn.Sigmoid(),	
		)
	def parameters(self):
		return self.classifier.parameters()
	def forward(self,x):
		x = self.classifier(x)
		#x = torch.nn.functional.softmax(x,dim=2)
		return x

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
		static_dropout (bool): If true, randomly masks static input and age encoding during training to make models that can accept either input (default False)
	"""
	
	def __init__(self,label=["Folder"],confounds=[],
				Y_dim: tuple = (1,32), # Number of labels, Number of choices
				C_dim: tuple = (1,32), # Number of labels, Number of choices
				Y_labels: list = [],
				C_confounds: list = [],
				n_dyn_inputs: int = 14,
				n_stat_inputs: int = 2,
				use_attn: bool = False,
				encode_age: bool = True,
				use_static_input: bool = True,
				variational: bool = False, # Makes it a variational encoder
				zero_input: bool = False, # Repeats input into classifier or makes it zeros
				remove_uncertain: bool = False,
				device = torch.device('cpu'),
				latent_dim: int = 128,
				weights: str = None,
				grad_layer: int = 7,
				verbose : bool = False,
				static_dropout: bool = True, # Randomly drops static inputs and age encoding when training
				vision_transformer: bool = False):
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
		
		# Use a vision transformer instead of CNN
		self.vision_transformer = vision_transformer
		
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

		self.verbose = verbose

		# Training options
		self.use_attn = use_attn
		self.encode_age = encode_age
		self.use_static_input = use_static_input
		self.static_dropout = static_dropout # Randomly mask static inputs in training
		self.variational = variational
		self.device=device
		# A record that prevents unrecognized keys from being applied during
		# the test phase
		self.static_record = [set() for _ in range(self.n_stat_inputs)]
		
		# Sets the multiplier for the number of features in each model component
		self.base_feat = 64
		# Modules
		

		# Makes the encoder output a variational latent space, so it's a
		# Gaussian distribution.
		if self.vision_transformer:
			self.encoder = ViTForClassification(hidden_size=1024,
				num_classes=self.latent_dim)
		else:
			self.encoder = Encoder(latent_dim=self.latent_dim)
		if self.variational:
			self.z_mean = nn.Sequential(
				nn.Linear(self.latent_dim,self.latent_dim)
			)
			self.z_log_sigma = nn.Sequential(
				nn.Linear(self.latent_dim,self.latent_dim)
			)
			self.epsilon = torch.distributions.Normal(0, 1)
			self.epsilon.loc = self.epsilon.loc
			self.epsilon.scale = self.epsilon.scale
		# The output of the classifier and regressor, and encoder are kept
		# consistent, to 16 max outputs and 32 possible choices. This makes
		# cross-training easier, though it's less efficient.
		self.classifiers = {}
		for l in label:
			self.classifiers[l] = Classifier(latent_dim = self.latent_dim,
										n_inputs = self.n_inputs,
										base_feat = self.base_feat,
										n_out = self.Y_dim[0],#16,
										n_labels = self.Y_dim[1])#32)
		if self.C_dim is not None:
			n_confounds,n_choices = self.C_dim
			self.regressors = {}
			for c in confounds:
				self.regressors[c] = Regressor(self.latent_dim,
					n_confounds=self.C_dim[0],
					n_choices=self.C_dim[1],
					device=self.device)
		else: self.regressors = None
		
		if weights is not None:
			if self.C_dim != (1,32) or self.Y_dim != (1,32):
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
		
		self.reset_encoded()
		#self.encoded_record = {}
		#for l in self.classifiers: self.encoded_record[l] = {}
		#try:
		#	for r in self.regressors: self.encoded_record[r] = {}
		#except: pass
		self.max_encoded = 8*self.latent_dim
	def full_encoded(self,out_fig="temp.png",label = None,labels_only=True,verbose=True,verbose2=False):
		if verbose2:
			cc = 0
			ccc = 0
			f = plt.figure()
			xticks = []
			alldist = []
			for l in list(self.classifiers) + list(self.regressors):
				for c in self.encoded_record[l]:
					if c is None or c == 'None': continue
					alldist.append([])
					#print("%s-%s: %d" %(l,c,len(self.encoded_record[l][c])))
					cc += len(self.encoded_record[l][c])
					ccc += self.max_encoded
					d_mean = np.squeeze(np.mean(self.encoded_record[l][c],axis=0))
					if len(d_mean) != self.latent_dim:
						print(d_mean.shape)
					assert(len(d_mean) == self.latent_dim)
					xticks.append(f"{l} - {c}")
					for l2 in list(self.classifiers) + list(self.regressors):
						for c2 in self.encoded_record[l2]:
							if c2 is None or c2 == 'None': continue
							d2_mean = np.squeeze(np.mean(self.encoded_record[l2][c2],axis=0))
							assert(len(d2_mean) == self.latent_dim)
							if False and c == c2:
								dist = np.mean(np.var(self.encoded_record[l2][c2],axis=0))
							else:
								dist = np.mean((d_mean - d2_mean) ** 2)
							alldist[-1].append(dist)
			alldist = np.array(alldist)
			plt.matshow(alldist)
			plt.xticks(list(range(len(xticks))),xticks, rotation=90,fontsize=4)
			plt.yticks(list(range(len(xticks))),xticks,fontsize=4)
			cb = plt.colorbar()
			cb.ax.tick_params(labelsize=14)
			plt.savefig(out_fig,dpi=500)
			plt.clf()
			plt.cla()
			plt.close(f)
			plt.close('all')
		if verbose:
			for l in self.encoded_record:
				for c in self.encoded_record[l]:
					if hasattr(self,'pop_key_lists') and len(self.pop_key_lists[l][c]) > 0:
						pop_key_avg = 0# np.mean(self.pop_key_lists[l][c])
						pop_key_max = 0 #np.max(self.pop_key_lists[l][c])
					else:
						pop_key_avg = 0
						pop_key_max = 0
					l_enc = len(self.encoded_record[l][c])
					if l in self.mean_cache and c in self.mean_cache[l] and len(self.mean_cache[l][c]):
						l_enc += self.mean_cache[l][c][2]
					print("%s - %s : %d (Mean: %.4f, Max: %.4f)" % (l,c,l_enc,pop_key_avg,pop_key_max))
		if label is None:
			for l in self.classifiers:
				if len(self.encoded_record[l]) == 0: return False
				for c in self.encoded_record[l]:
					if len(self.encoded_record[l][c]) < self.max_encoded:
						return False
				#	elif hasattr(self,'pop_key_lists') and np.mean(self.pop_key_lists[l][c]) < 7:
				#		return False
		else:
			if len(self.encoded_record[label]) == 0: return False
			for c in self.encoded_record[label]:
				if len(self.encoded_record[label][c]) < self.max_encoded:
					return False
				#elif hasattr(self,'pop_key_lists') and \
				#	label in self.pop_key_lists and \
				#	c in self.pop_key_lists[label] and \
				#	np.mean(self.pop_key_lists[label][c]) < 7:
				#	return False
		return True
				
	def reset_encoded(self):
		self.encoded_record = {}
		self.mean_cache = {}
		self.covar_cache = {}
		for l in self.classifiers:
			self.encoded_record[l] = {}
		for r in self.regressors:
			self.encoded_record[r] = {}
		self.mean_cache = {}
	def record_encoded(self,
			e,
			fname,
			e_is_encoded=True,
			on_eval=False,
			label=None,
			pop_key=None):
		if not hasattr(self,'pop_key_lists'):
			self.pop_key_lists = {}
		if not e_is_encoded:
			e = self.forward(e,return_encoded=True)
			e = e.clone().detach()
		label = list(self.encoded_record) if label is None else [label]
		for l in label:
			if l not in self.pop_key_lists: self.pop_key_lists[l] = {}
			c = self.dataloader.database.loc_val(fname,l)
			if c not in self.pop_key_lists[l]: self.pop_key_lists[l][c] = []
			if c not in self.encoded_record[l]:
				self.encoded_record[l][c] = []
			if self.training or on_eval:
				if False: # and pop_key is not None:
					if len(self.pop_key_lists[l][c]) == 0:
						ind = 0
					elif self.pop_key_lists[l][c][0] <= pop_key:
						ind = 0
					elif self.pop_key_lists[l][c][-1] >= pop_key:
						ind = len(self.pop_key_lists[l][c])
					else:
						for i in range(len(self.pop_key_lists[l][c])):
							assert(self.pop_key_lists[l][c][i] >= self.pop_key_lists[l][c][i+1])
							if self.pop_key_lists[l][c][i] >= pop_key and pop_key >= self.pop_key_lists[l][c][i+1]:
								ind = i+1
								break
					#ind = bisect(self.pop_key_lists[l][c],pop_key)
				else: ind = 0
				self.encoded_record[l][c].insert(ind,e)
				self.pop_key_lists[l][c].insert(ind,pop_key)
			else:
				raise Exception("Attempting to record encoded out of training")
				#self.encoded_record[l][c].insert(
				#	random.randint(0,len(self.encoded_record[l][c])),
				#e)
			while len(self.encoded_record[l][c]) > self.max_encoded:
				if self.training:
					self.encoded_record[l][c].pop()
					self.pop_key_lists[l][c].pop()
				else:
					self.combine_encoded(l,c)
					self.encoded_record[l][c] = []
					self.pop_key_lists[l][c] = []
				#self.encoded_record[l][c].pop()
				#self.pop_key_lists[l][c].pop()
	def triplet_all(self,e,fname,dataloader=None,record=True,n_mean=None,
			target_label=None,record_only=False):
		if not hasattr(self,'dataloader') and dataloader is not None:
			self.dataloader = dataloader
		elif not hasattr(self,'dataloader'):
			raise Exception("Must set dataloader in first call to mahalanobis")
		triplet_loss_l = torch.tensor(0.0,device = self.device)
		triplet_loss_c = torch.tensor(0.0,device = self.device)

		for i in range(e.shape[0]):
			triplet_l,triplet_c = \
				self._triplet_all(e[i,...],fname[i],record_only=record_only)
			
			triplet_loss_l += triplet_l
			triplet_loss_c += triplet_c
		
		return triplet_loss_l / e.shape[0] ,triplet_loss_c / e.shape[0]
	
	def valid(self,l,v):
		if l == 'AlzStage' and v == 'CONTROL': return True
		if 'ICD' in l:
			if 'EXCLUDE' or 'NOT' in v: return False
			else: return True
		return True

	def get_group_label(self,target_label):
		return ""
	def get_big_groupname(self,fname):
		groups = []
		for l in self.classifiers:
			v = self.dataloader.database.loc_val(fname,l)
			groups.append(v)
		if any([False if g is None else g.startswith('I') for g in groups]):
			#if 'CONTROL' in groups:
			#	raise Exception("Flawed labeling: %s" % fname)
			for g in groups:
				if g is not None and g.startswith('I'): return g,g
		elif 'AD' in groups:
			return 'AlzStage','AD'
		elif 'CONTROL' in groups:
			return 'AlzStage','CONTROL'
		else:
			raise Exception('Incomplete labeling: %s' % fname)
		
	def _triplet_all(self,e_inp,fname,m=0.2,record_only=False,new_cluster=False):
		triplet_loss_l = torch.tensor(0.0,device = self.device)
		triplet_loss_c = torch.tensor(0.0,device = self.device)
		tl_count,tc_count=0,0
		l_inp,v_inp = self.get_big_groupname(fname)
		if new_cluster:
			amount_initial_centers = 2
			sample = self.encoded_record
			self.initial_centers = kmeans_plusplus_initializer(sample).initialize()
			xmeans_instance = xmeans(sample, initial_centers, 20)
			xmeans_instance.process()
			self.clusters = xmeans_instance.get_clusters()
			self.centers = xmeans_instance.get_centers()
			visualizer = cluster_visualizer()
			visualizer.append_clusters(self.clusters, sample)
			visualizer.append_cluster(self.centers, None, marker='*', markersize=10)
			visualizer.show()

		if v_inp not in self.encoded_record[l_inp]: record_only = True
		if not record_only:
			for v_rec1 in self.encoded_record[l_inp]:
				queue = []
				cc = 0
				random.shuffle(self.encoded_record[l_inp][v_rec1])
				for e_rec1 in self.encoded_record[l_inp][v_rec1]:
					cc2 = 0
					cc += 1
					if cc > 32: break
					for v_rec2 in self.encoded_record[l_inp]:
						for e_rec2 in self.encoded_record[l_inp][v_rec2]:
							if v_rec1 == v_rec2:
								if v_rec1 == v_inp: continue
								else:
									e_a,e_p,e_n = e_rec1,e_rec2,e_inp
							else:
								if v_rec1 == v_inp:
									e_a,e_p,e_n = e_rec1,e_inp,e_rec2
								elif v_rec2 == v_inp:
									e_a,e_p,e_n = e_rec2,e_inp,e_rec1
								else: continue
							pd = torch.pow(e_a - e_p,2).mean()
							nd = torch.pow(e_a - e_n,2).mean()
							dist = pd - nd
							
							if dist + m > 0.0:
								queue.append(dist)
							cc2 += 1
							if cc2 > 5: break
				queue = sorted(queue,reverse=True)
				for i in range(min(16,len(queue))):
					if queue[i] + m > 0.0:
						triplet_loss_l += queue[i] + m
						tl_count +=1
				#if cc > 16: break
		self.record_encoded(e_inp.clone().detach(),fname)
		if tl_count > 0:
			triplet_loss_l = triplet_loss_l / tl_count
		if tc_count > 0:
			triplet_loss_c = triplet_loss_c / tc_count
		return triplet_loss_l,triplet_loss_c
		
	def euclidean(self,e,fname,dataloader=None,record=True,n_mean=None,
			target_label=None):
		if not hasattr(self,'dataloader') and dataloader is not None:
			self.dataloader = dataloader
		elif not hasattr(self,'dataloader'):
			raise Exception("Must set dataloader in first call to mahalanobis")
		label_m_x_all,conf_m_x_all,label_m_o_all,conf_m_o_all = [],[],[],[]
		for i in range(e.shape[0]):
			label_m_x,conf_m_x,label_m_o,conf_m_o = \
				self._euclidean(e[i,...],fname[i],
					record=record,
					n_mean=n_mean,
					target_label=target_label)
			label_m_x_all.append(label_m_x)
			conf_m_x_all.append(conf_m_x)
			label_m_o_all.append(label_m_o)
			conf_m_o_all.append(conf_m_o)
		
		label_m_x_all = np.concatenate(label_m_x_all,axis=0)
		conf_m_x_all  = np.concatenate(conf_m_x_all,axis=0)
		label_m_o_all = np.concatenate(label_m_o_all,axis=0)
		conf_m_o_all  = np.concatenate(conf_m_o_all,axis=0)
		
		return label_m_x_all,conf_m_x_all,label_m_o_all,conf_m_o_all
	
	def _euclidean(self,e,fname,record=True,n_mean=None,target_label=None):
		e = np.expand_dims(e,axis=0)
		assert(len(e.shape) == 2)
		assert(e.shape[0] == 1)
		assert(e.shape[1] == self.latent_dim)
		label_m_x = np.zeros(e.shape)#,device=self.device)
		conf_m_x = np.zeros(e.shape)#,device=self.device)
		label_m_o = np.zeros(e.shape)#,device=e.device)
		conf_m_o = np.zeros(e.shape)#,device=e.device)
		lc_x = 0
		cc_x = 0
		lc_o = 0
		cc_o = 0
		assert((not (self.training or record)) or (self.training and record))
		assert(n_mean is not None or (not self.training))
		mlim = self.max_encoded if n_mean is None else n_mean
		for l in self.encoded_record:
			if target_label not in ["AlzStage",
									"Ages_Buckets",
									"SexDSC",
									"Modality",
									"MRModality",
									"Angle",
									"ScanningSequence"]:
				continue
			#if target_label is not None and l != target_label and l in self.classifiers: continue
			#if label is not None and l != label and l in self.model.classifiers: continue
			c = self.dataloader.database.loc_val(fname,l)
			if c is None or c == 'None': continue
			for c2 in self.encoded_record[l]:
				if c2 is None or c2 == 'None': continue
				if len(self.encoded_record[l][c2]) < mlim: continue
				mean_arr,_ = self.get_mean_covar(l,c2,n_mean=n_mean)
				mean_arr = mean_arr.cpu().detach().numpy()
				assert(mean_arr.shape[1] == self.latent_dim)
				assert(mean_arr.shape[0] == 1)
				assert(len(mean_arr.shape) == 2)
				if (c != c2):
					if e.shape != mean_arr.shape:
						print(e.shape)
						print(mean_arr.shape)
					assert(e.shape == mean_arr.shape)
					#delta = e - mean_arr
					#delta = delta.squeeze()
					#m = torch.dot(delta,torch.matmul(torch.inverse(covar),delta)) / 128
					if l in self.classifiers:
						label_m_o += mean_arr
						lc_o += 1
					elif l in self.regressors:
						conf_m_o += mean_arr
						cc_o += 1
					else:
						raise Exception("Invalid label: %s" % l)
				elif (c == c2):
					if l in self.classifiers:
						label_m_x += mean_arr
						lc_x += 1
					elif l in self.regressors:
						conf_m_x += mean_arr
						cc_x += 1
					else:
						raise Exception("Invalid label: %s" % l)
		
		if (self.training or record):
			assert(n_mean is not None)
			self.record_encoded(e,fname)
		
		if lc_o != 0: label_m_o = label_m_o / lc_o
		if cc_o != 0: conf_m_o  = conf_m_o  / cc_o
		if lc_x != 0: label_m_x = label_m_x / lc_x
		if cc_x != 0: conf_m_x  = conf_m_x  / cc_x
		return label_m_x,conf_m_x,label_m_o,conf_m_o

	
	def get_mahal_dist(self,e,label,value,n_mean=None):
		m = []
		#print("&")
		#print("get_mahal_dist")
		#print(f"label: {label}")
		#print(f"value: {value}")
		#print("e.size(): %s" % str(e.size()))
		for i in range(e.size()[0]):
			m.append(self._get_mahal_dist(e[i,...],label,value,n_mean=n_mean))
		return m
				
	def old_get_mahal_dist(self,e,label,value):
		mean_arr,inv_covar = self.get_mean_covar(label,value)
		mean_arr = mean_arr.cuda(self.device)
		#inv_covar = inv_covar.cuda(self.device)
		#e = e.cuda(self.device)
		delta = e - mean_arr
		delta = delta.squeeze()
		m = torch.dot(delta,torch.matmul(inv_covar,delta)) / len(e)
		m = torch.sqrt(m)
		return m
	
	def _get_mahal_dist(self,e,label,value,n_mean=None):
		mean_arr,covar = self.get_mean_covar(label,value,n_mean=n_mean)
		mean_arr = mean_arr.cuda(self.device)
		covar = covar.cuda(self.device)
		#inv_covar = inv_covar.cuda(self.device)
		#e = e.cuda(self.device)
		#print("mean_arr.size(): %s" % str(mean_arr.size()))
		delta = e - mean_arr
		delta = delta.squeeze()
		m = torch.dot(delta,torch.matmul(torch.inverse(covar),delta)) / len(e)
		m = torch.sqrt(m)
		return m

	def combine_encoded(self,l,c):
		if l not in self.mean_cache:
			self.mean_cache[l] = {}
		if c not in self.mean_cache[l]:
			self.mean_cache[l][c] = []
		tlist = self.encoded_record[l][c]
		list_of_enc = self.encoded_record[l][c]
		for i,e in enumerate(list_of_enc):
			if len(e.size()) == 1:
				e = torch.unsqueeze(e,0)
				list_of_enc[i] = e
			e.to(self.device)
		arr = torch.cat(list_of_enc,0)
		mean_arr = torch.mean(arr,0).unsqueeze(0)
		covar_arr = torch.cov(arr.T)
		n = len(list_of_enc)
		if len(self.mean_cache[l][c]) == 0:
			self.mean_cache[l][c] = [covar_arr,mean_arr,n]
		else:
			C,M,N = self.mean_cache[l][c]
			C_com,M_com,N_com = combine_covar(C,covar_arr,M,mean_arr,N,n)
			self.mean_cache[l][c] = [C_com,M_com,N_com]

	def get_mean_covar(self,label,value,mean_only=False,n_mean=None):
		#for label in self.encoded_record:
		#	print(label)
		#print("^")
		#print("get_mean_covar")
		#if self.mean_arr is not None:
		#	return self.mean_arr,self.covar_arr
		if self.training:
			if n_mean is not None:
				arr = [self.encoded_record[label][value][i] for i in range(n_mean)]
				arr = torch.cat([torch.tensor(_,device=self.device) for _ in arr],0)
				assert(arr.size()[0] == n_mean and arr.size()[1] == self.latent_dim)
			else:
				list_of_enc = self.encoded_record[label][value]
				for i,e in enumerate(list_of_enc):
					if len(e.size()) == 1:
						e = torch.unsqueeze(e,0)
						list_of_enc[i] = e
					e.to(self.device)
				arr = torch.cat(list_of_enc,0)
				#print("l")
				#arr = torch.cat([torch.cat([torch.cat([torch.tensor(_,device=self.device) for _ in self.encoded_record[l][v]],0) for v in self.encoded_record[l]],0) for l in self.encoded_record],0)
			#print("arr.size(): %s" % str(arr.size()))
			mean_arr = torch.mean(arr,0).unsqueeze(0)
			#print("mean_arr.size(): %s" % str(mean_arr.size()))
			if mean_only:
				covar_arr = -1
			else:
				covar_arr = torch.cov(arr.T)
			#print("covar_arr.size(): %s"%str(covar_arr.size()))
			#self.mean_arr,self.covar_arr = mean_arr,covar_arr
			mean_arr = mean_arr.to(self.device)
			covar_arr = covar_arr.to(self.device)
			if not self.training:
				self.mean_cache[label][value],self.covar_cache[label][value] = mean_arr,covar_arr
			return mean_arr,covar_arr
		else:
			if label not in self.mean_cache or \
					value not in self.mean_cache[label] or \
					len(self.mean_cache[label][value]) == 0:
				self.combine_encoded(label,value)
			return self.mean_cache[label][value][1],self.mean_cache[label][value][0]


		if not self.training and label in self.mean_cache \
				and value in self.mean_cache[label]:
			mean_arr = self.mean_cache[label][value]
			assert(label in self.inv_covar_cache)
			assert(value in self.inv_covar_cache[label])
			inv_covar = self.inv_covar_cache[label][value]
		else:
			arr = torch.cat(self.encoded_record[label][value],0)
			#print("arr")
			#print(arr.size())
			mean_arr = torch.mean(arr,0)
			mean_arr = mean_arr.unsqueeze(0)
			if mean_only: covar=0
			else: covar = torch.cov(arr.T)
			inv_covar = torch.inverse(covar)
			if not self.training:
				if label not in self.mean_cache:
					self.mean_cache[label] = {}
					self.inv_covar_cache[label] = {}
				self.mean_cache[label][value] = mean_arr
				self.inv_covar_cache[label][value] = inv_covar
		return mean_arr,inv_covar
		
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
		self.regressors = {}
		self.classifiers = {}
		sl = []
		for s in state_dict:
			if s.startswith('classifier'):
				_,l = s.split(".",1)
				assert(_=="classifier")
				self.classifiers[l] = Classifier(latent_dim = self.latent_dim,
										n_inputs = self.n_inputs,
										base_feat = self.base_feat,
										n_out = self.Y_dim[0],#16,
										n_labels = self.Y_dim[1])#32)
				self.classifiers[l].load_state_dict(state_dict[s])
				sl.append(s)
			if s.startswith('regressor'):
				_,c = s.split(".",1)
				self.regressors[c] = Regressor(self.latent_dim,
					n_confounds=self.C_dim[0],
					n_choices=self.C_dim[1],
					device=self.device)
				self.regressors[c].load_state_dict(state_dict[s])
				
				sl.append(s)
			if s == "encoded_record":
				self.encoded_record = state_dict['encoded_record']
				sl.append(s)
			if s == "pop_key_lists":
				self.pop_key_lists = state_dict['pop_key_lists']
				sl.append(s)
			if s == "mean_cache":
				self.mean_cache = state_dict['mean_cache']
				sl.append(s)
		for s in sl: del state_dict[s]
		super().load_state_dict(state_dict,*args,**kwargs)
		self.cuda(self.device)
		return
		
	def state_dict(self,*args,**kwargs):
		state_dict1 = super().state_dict(*args, **kwargs)
		for l in self.classifiers:
			state_dict1.update(
				{'classifier.%s' % l : self.classifiers[l].state_dict()}
			)
		if self.regressors is not None:
			for c in self.regressors:
				state_dict1.update(
					{'regressor.%s' % c : self.regressors[c].state_dict()}
				)
		if self.encoded_record is not None:
			state_dict1.update(
				{'encoded_record': self.encoded_record}
			)
		if hasattr(self,'pop_key_lists'):
			state_dict1.update(
				{ 'pop_key_lists' : self.pop_key_lists }
			)
		if hasattr(self,'mean_cache'):
			state_dict1.update(
				{ 'mean_cache' : self.mean_cache }
			)
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
		for c in self.regressors:
			self.regressors[c].cuda(device)
		for l in self.classifiers:
			self.classifiers[l].cuda(device)
		#for l in self.encoded_record:
		#	for c in self.encoded_record[l]:
		#		for e in self.encoded_record[l][c]:
		#			e = e.cuda(device)
		for l in self.mean_cache:
			for v in self.mean_cache[l]:
				if len(self.mean_cache[l][v]) > 0:
					self.mean_cache[l][v][0] = self.mean_cache[l][v][0].cuda(device)
					self.mean_cache[l][v][1] = self.mean_cache[l][v][1].cuda(device)
		return super().cuda(device)
		
	def cpu(self):
		self.device = torch.device('cpu')
		if self.variational:
			self.epsilon.loc = self.epsilon.loc.cpu()
			self.epsilon.scale = self.epsilon.scale.cpu()
		for c in self.regressors:
			self.regressors[c].cpu()
		return super().cpu()
	
	def regressor_freeze(self):
		for param in self.classifier_parameters():
			param.requires_grad = True
		for param in self.regressor_parameters():
			param.requires_grad = False
			
	def classifier_freeze(self):
		for param in self.classifier_parameters():
			param.requires_grad = False
		for param in self.regressor_parameters():
			param.requires_grad = True
	
	def regressor_parameters(self):
		return itertools.chain(*[self.regressors[c].parameters() \
			for c in sorted(self.regressors)])

	def classifier_parameters(self):
		if self.variational:
			return itertools.chain(self.encoder.parameters(),
				*[self.classifiers[l].parameters() for l in sorted(self.classifiers)],
				self.z_log_sigma.parameters(),
				self.z_mean.parameters())
		else:
			return itertools.chain(self.encoder.parameters(),
				*[self.classifiers[l].parameters() for l in sorted(self.classifiers)])

	def forward(self,
				x,
				static_input=None,
				dates=None,
				bdate=None,
				return_regress = False,
				return_encoded = False,
				encoded_input = False,
				grad_eval = False,
				target_confound = None,
				target_label = None,
				return_confidence = False,
				dataloader=None,
				output_attentions=False):
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
			target_label (str) : Label to predict
			target_confound (str) : Confound to predict
			return_confidence (bool) : Returns a number indicating the confidence that a given input is valid
		"""
		use_regression = (len(self.regressors) > 0) and \
			(self.training or return_regress)
		#if record_encoding:
		#	assert(isinstance(x,BatchRecord))
		#	xf = [im.npy_file for im in x.image_records]
		if not encoded_input:
				#assert(self.gradients is not None)
			if isinstance(x,BatchRecord):
				assert(x.dtype == "torch")
				if len(x.get_static_inputs()) > 0:
					static_input = x.get_static_inputs() # [_[0] for _ in x.get_static_inputs()]
					if self.n_stat_inputs != len(static_input):
						raise Exception(
					"Number of static inputs not equal to input: %d != %d"\
						 % (len(static_input),self.n_stat_inputs))
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
				elif output_attentions:
					x,attention = self.encoder(x,output_attentions=True)
				else:
					x = self.encoder(x)
				self.z_mean_ = self.z_mean(x)
				self.z_log_sigma_ = self.z_log_sigma(x)
				x = self.z_mean_ + (self.z_log_sigma_.exp()*self.epsilon.sample(self.z_mean_.shape))
				#self.kl = (z_mean**2 + z_log_sigma.exp()**2 - z_log_sigma-0.5).mean()
				self.kl = 0.5 * (self.z_mean_**2 + self.z_log_sigma_.exp()**2 - self.z_log_sigma_ - 1).mean()
			else:
				if grad_eval:
					x = self.encoder.encoder[:self.grad_layer](x)
					h = x.register_hook(self.activations_hook)
					x = self.encoder.encoder[self.grad_layer:](x)
				elif output_attentions:
					x,attention = self.encoder(x,output_attentions=True)
				else:
					x = self.encoder(x)
			#if record_encoding:
			#	self.saved_encoding = x.cpu().detach().numpy()
			#	self.euclidean(x,xf,record=True,dataloader=dataloader)
			if hasattr(self,'remove_uncertain'):
				if self.remove_uncertain:
					if self.record_training_sample:
						self.training_sample[:,
							self.training_i:min(self.training_i + x.shape[0],
							self.num_training_samples)] = x
						self.training_i += x.shape[0]
						if self.training_i >= self.num_training_samples:
							self.record_training_sample = False
			if return_confidence:
				x_clone = x.clone().detach()
			if return_encoded:
				return x
		if use_regression:
			if target_confound is None:
				reg = [self.regressors[c](x) for c in self.regressors]				
				reg = torch.cat(reg,axis=1)
			else:
				reg = self.regressors[target_confound](x)
			# Encode dynamic inputs with dates using positional encoding
		if (self.encode_age and not self.training) or \
			(self.encode_age and self.training and not self.static_dropout) or \
			(self.training and self.static_dropout and \
				random.choice([True,False])):
			if self.verbose: print("Encoding age")
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
		else:
			if self.verbose: print("Not encoding age")
			#age_encodings = []
			#for i in range(x.size()[0]):
			#	age_encoding = get_age_encoding(
			#				datetime(year=2000,day=1,month=1),
			#				datetime(year=1920,day=1,month=1),
			#				d = self.latent_dim)
			#	age_encodings.append(age_encoding)
			#age_encodings = np.array(age_encodings)
			#age_encodings =  torch.tensor(
			#						age_encodings,
			#						device=x.device
			#						).float()
			#x = torch.add(x,age_encodings)
		
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
		if static_input is not None and len(static_input) > 0 and self.use_static_input:
			if len(static_input) != self.n_stat_inputs:
				print(static_input)
				print(self.n_stat_inputs)
				warnings.warn(
						"Received %d static inputs, but it is set at %d" % \
						(len(static_input),self.n_stat_inputs)
					)
			if self.training:
				for i,e in enumerate(static_input):
					self.static_record[i].add(e)
			elif False:
				for i,e in enumerate(self.static_record):
					if static_input[i] not in e:
						raise Exception("Input %s not a previous demographic input (previous inputs were %s)" % (static_input[i],str(e)))
			
			x_ = encode_static_inputs(static_input,d=self.latent_dim)
			x_ = torch.tensor(x_,device = x.device)
			x_ = torch.unsqueeze(x_,0)
			for i in range(x_.shape[0]):
				if (self.use_static_input and not self.training) or \
					(self.use_static_input and self.training and not self.static_dropout) or \
					(self.training and self.static_dropout and random.choice([True,False])):
					if self.verbose: print("Encoding static input")
					x[:,(-(self.n_stat_inputs) + i):,:] = x_[i,:]
				else:
					if self.verbose:
						print("Not encoding static input")
		else:
			if self.verbose: print("Not encoding static input")
		
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
		if target_label is None:
			x = [self.classifiers[l](x) for l in self.classifiers]
			x = torch.cat(x,axis=1)
		else: x = self.classifiers[target_label](x)
		if output_attentions:
			return x,attention
		elif return_confidence:
			assert(target_label is not None)
			c = int(torch.argmax(x,2).squeeze())
			#print("**")
			#print("x: %s" % str(x))
			#print("target_label: %s" % target_label)
			#print("int(torch.argmax(x,2).squeeze()): %d" % c)
			if True:
				c = min(c,len(self.encoded_record[target_label])-1)
				c = sorted(list(self.encoded_record[target_label]),key=lambda x: "zzzzzzzzz" if x is None else x)[c]
			else:
				print(list(self.encoded_record[target_label]))
				raise Exception("Index out of bounds: %s" % str(c))
			#print("sorted(list(self.encoded_record[target_label]),key=lambda x: 'zzzzzzzzz' if x is None else x)[c]: %s" % c)
			confidence = self.get_mahal_dist(x_clone,target_label,c)
			#print("x: %s" % str(x))
			#print("confidence: %s" % str(confidence))
			return x,confidence
		elif use_regression: return x,reg
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
