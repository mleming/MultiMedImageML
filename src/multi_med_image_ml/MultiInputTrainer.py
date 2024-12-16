import torch,os,json
from torch import nn
import numpy as np
from .Records import BatchRecord
import matplotlib.pyplot as plt
import matplotlib
from time import time
matplotlib.use('Agg')

class MultiInputTrainer:
	"""Used to train MultiInputModule.
	
	MultiInputModule requires an adversarial technique to train it, and the
	various data queueing techniques used get a bit complicated, so this 
	method is used to abstract all of that.
	
	Attributes:
		model (MultiInputModule): Input model to train
		lr (float): Learning rate
		loss_function: Pytorch loss function. MSE loss is used instead of class entropy because it is smoother and tends to work a bit better with the adversarial function, but this can be tested further (default nn.MSELoss)
		name (str): Name of the model, which is used for saving checkpoints and output graphs (default 'experiment_name')
		optimizer (torch.optim): Adam optimizer for the encoder/classifier. Incentivized to classify by the true label and set the regressor to the same values.
		optimizer_reg (torch.optim): Adam optimizer for the encoder/regressor. Incentivized to detect confounds from each individual image.
		out_record_folder (str): If set, outputs images of the loss function for the optimizers over time, for the classifier, the regressor, and the adversarial loss (default None)
		checkpoint_dir (str): If set, saves the model and optimizer state (default None)
		save_latest_freq (int): Number of iterations before it saves the loss image and the checkpoint (default 100)
		batch_size (int): Batch size of the training. Due to the optional-input nature, this cannot be set in the dataloader. Only one set of images can be passed through the loop at a given time. batch_size is how frequently the backpropagation algorithm is applied after graphs have accumulated (default 64)
		verbose (bool): Whether to print (default False)
		update_classifier (bool): Boolean to determine whether optimizer (True) or optimizer_reg (False) is applied
		index (int): Counts the number of iterations the trainer has gone through
	"""
	
	def __init__(self,
		model,
		dataloader,
		lr=1e-5,
		betas = (0.5,0.999),
		loss_function = "mse",
		batch_size = 64,
		regress = True,
		out_record_folder = None,
		checkpoint_dir = None,
		name = 'experiment_name',
		verbose = False,
		save_latest_freq = 100,
		return_lim = False,
		discriminator_optimizer='adam',
		classifier_optimizer='adam',
		forget_optimizer_state = False,
		use_triplet = False):
		
		# Core variables for training the self.model
		self.model = model
		self.name = name
		self.dataloader = dataloader
		self.classifier_optimizer = classifier_optimizer
		if self.classifier_optimizer == "adam":
			self.optimizer = torch.optim.Adam(
				self.model.classifier_parameters(),
				betas = betas,
				lr = lr
			)
		elif self.classifier_optimizer == "sgd":
			self.optimizer = torch.optim.SGD(
				self.model.classifier_parameters(),
				lr=lr
			)
		else:
			raise Exception("Invalid classifier optimizer: %s" % \
				self.classifier_optimizer)
		self.discriminator_optimizer = discriminator_optimizer
		if self.discriminator_optimizer == "adam":
			self.optimizer_reg = torch.optim.Adam(
				self.model.regressor_parameters(),
				betas = betas,
				lr = lr
			)
		elif self.discriminator_optimizer == "sgd":
			self.optimizer_reg = torch.optim.SGD(
				self.model.regressor_parameters(),
				lr = lr
			)
		else:
			raise Exception("Invalid value for discriminator optimizer: %s"\
				 % self.discriminator_optimizer)
		# Load saved or pretrained models
		
		self.checkpoint_dir = checkpoint_dir
			
		if self.checkpoint_dir is not None:
			os.makedirs(self.checkpoint_dir,exist_ok=True)
			self.model_file = os.path.join(
				self.checkpoint_dir,
				'%s.pt' % self.name)
			if os.path.isfile(self.model_file):
				state_dicts = torch.load(self.model_file)
				if "encoded_record" in state_dicts:
					state_dicts['model_state_dict']['encoded_record'] = state_dicts['encoded_record']
				self.model.load_state_dict(state_dicts['model_state_dict'])
				self.model.regressor_freeze()
				if not forget_optimizer_state:
					if self.classifier_optimizer == "adam":
						self.optimizer.load_state_dict(
							state_dicts['optimizer_state_dict']
						)
					if self.discriminator_optimizer == "adam":
						self.optimizer_reg.load_state_dict(
							state_dicts['optimizer_reg_state_dict']
						)
		if loss_function == "mse":
			self.loss_function = nn.MSELoss()
		elif loss_function == "cce":
			self._loss_function = nn.CrossEntropyLoss()
			def _loss(inp,target):
				target = torch.squeeze(target,1)
				inp = torch.squeeze(inp,1)
				return self._loss_function(inp,target)
			self.loss_function = _loss
		else:
			raise Exception("Invalid loss function: %s" % loss_function)
		self.batch_size = batch_size
		self.regress = regress
		self.update_classifier = True
		self.return_lim=return_lim
		# Outputs images of the loss function over time
		self.out_record_folder = out_record_folder
		self.index           = 0
		self.loss_Y          = 0
		self.loss_C_dud      = 0
		self.loss_classifier = 0
		self.loss_regressor  = 0
		if model.variational:
			self.loss_kl = 0
		if self.out_record_folder is not None:
			self.loss_image_dir = os.path.join(self.out_record_folder,'loss_ims')
		else: self.loss_image_dir = None
		if self.loss_image_dir is not None:
			self.pids_read = set()
			self.x_files_read = set()
			self.loss_tracker = LossTracker(self.name,self.loss_image_dir)
			self.loss_Y = self.loss_tracker.get_last_y_class_loss()
			self.loss_C_dud = self.loss_tracker.get_last_y_class_adv_loss()
			self.loss_classifier = self.loss_tracker.get_last_y_class_loss()
			self.loss_regressor  = self.loss_tracker.get_last_y_reg_loss()
			self.index = self.loss_tracker.get_last_xs()
			if model.variational:
				self.loss_kl = self.loss_tracker.get_last_y_kl_loss()
		self.verbose = verbose
		self.use_triplet = use_triplet
		self.save_latest_freq = save_latest_freq
		self.time = time()
		self.time_dict = {}


	def log_time(self,m):
		t = time() - self.time
		self.time = time()
		if m not in self.time_dict:
			self.time_dict[m] = (t,1)
		else:
			self.time_dict[m] = (self.time_dict[m][0] + t,
								self.time_dict[m][1]+1)
	def reset_time(self):
		self.time=time()
	def get_time_str(self):
		s = ""
		for m in sorted(self.time_dict):
			total_time,count = self.time_dict[m]
			mean_time = total_time / count
			s = s + ("%s %.3f; " % (m,mean_time))
		return s

	#def euclidean(self,x,f
	def loop(self,
			pr: BatchRecord):
		"""Loops a single BatchRecord through one iteration
		
		Loops a BatchRecord through one iteration. Also switches the queues of
		the MedImageLoader as it switches between optimizers.
		
		Args:
			pr (multi_med_image_ml.Records.BatchRecord): Record to be evaluated
		"""

		assert(isinstance(pr,BatchRecord))
		
		self.reset_time()
		x = self.model(pr,return_encoded=True)
		if self.model.variational:
			z = self.model.z_mean_
		else:
			z = x
		label_m_x_all,conf_m_x_all,label_m_o_all,conf_m_o_all = \
			self.model.euclidean(z.clone().detach().cpu().numpy(),
								pr.get_X_files(),
								dataloader = self.dataloader,
								n_mean=256,
								record=True,
								target_label=self.dataloader.tl())
		#print("---")
		#print(f"label_m_x_all: {label_m_x_all}")
		#print(f"conf_m_x_all: {conf_m_x_all}")
		#print(f"label_m_o_all: {label_m_o_all}")
		#print(f"conf_m_o_all: {conf_m_o_all}")
		label_m_x_all,conf_m_x_all,label_m_o_all,conf_m_o_all = \
			torch.tensor(label_m_x_all,device=self.model.device),\
			torch.tensor(conf_m_x_all,device=self.model.device),\
			torch.tensor(label_m_o_all,device=self.model.device),\
			torch.tensor(conf_m_o_all,device=self.model.device)
		
		mean_lx = (z - label_m_x_all).pow(2).mean(1)
		mean_lo = (z - label_m_o_all).pow(2).mean(1)
		mean_cx = (z - conf_m_x_all).pow(2).mean(1)
		mean_co = (z - conf_m_o_all).pow(2).mean(1)
		if self.use_triplet:
			triplet_loss_l,triplet_loss_c = self.model.triplet_all(x,pr.get_X_files(),
										dataloader = self.dataloader,
										record_only = not self.update_classifier)
		else:
			triplet_loss_l,triplet_loss_c = torch.tensor(0.0),torch.tensor(0.0)

		self.label_m =  (mean_lx - mean_lo).mean()
		self.conf_m =  (mean_co - mean_cx).mean()
		l_or_c = self.dataloader.stack
		tl_v = self.dataloader.tl()

		self.log_time("1. Encode run")
		if pr.get_text_records:
			x_text = encode_static_inputs(
						pr.get_text_records(),
						d=self.model.latent_dim
					)
			x = torch.concat(x,x_text,axis=0)
		self.log_time("2. Encode static")
		
		self.log_time("3. Run encoded")
		
		if self.update_classifier:
			if self.model.variational:
				self.loss_kl = self.model.kl * 0.05
			if self.dataloader.stack == "Labels":
				y_pred,y_reg = self.model(x,
						encoded_input=True,
						dates=pr.get_exam_dates(),
						bdate=pr.get_birth_dates()[0],
						static_input = pr.get_static_inputs(),
						target_label = self.dataloader.tl()
					)
				Y = pr.get_Y(label=self.dataloader.tl())
				self.log_time("4. Get Y")
				self.loss_Y = self.loss_function(y_pred,Y)
				self.loss_classifier = self.loss_Y
				#self.loss_classifier = self.loss_classifier + self.conf_m + self.label_m
			elif self.dataloader.stack == "Confounds":
				y_pred,y_reg = self.model(x,
						encoded_input=True,
						dates=pr.get_exam_dates(),
						bdate=pr.get_birth_dates()[0],
						static_input = pr.get_static_inputs(),
						target_confound = self.dataloader.tl(),
						return_regress = True
					)
				c_dud = pr.get_C_dud(
										return_lim=self.return_lim,
										confound=self.dataloader.tl())
				if c_dud.size() != y_reg.size():
					print(self.return_lim)
					print(c_dud.size()) # 1,32
					print(y_reg.size()) # 1,1
				assert(c_dud.size() == y_reg.size())
				#Y = pr.get_Y(label=self.dataloader.tl())
				self.loss_C_dud = self.loss_function(y_reg,c_dud)

				self.loss_classifier = self.loss_C_dud
				#self.loss_classifier = self.loss_classifier + self.conf_m + self.label_m
			else:
				raise Exception("Invalid dl mode: %s" % self.dataloader.stack)
			#self.loss_classifier = self.loss_Y + (self.loss_C_dud)
			#if self.model.variational:
			#	self.loss_kl = self.model.kl * 0.05
			#	print("self.loss_kl.size()")
			#	print(self.loss_kl.size())
			#	self.loss_classifier = self.loss_classifier + self.loss_kl
			#self.loss_classifier = self.loss_classifier + self.conf_m - self.label_m
			#if not self.has_var('x_grad'): self.x_grad = 0
			#self.x_grad = self.x_grad + x
			self.log_time("5. Loss additions")
			self.loss_classifier = self.loss_classifier + self.label_m + self.conf_m
			if self.model.variational:
				self.loss_classifier = self.loss_classifier + self.loss_kl
			self.loss_classifier.backward()
			self.log_time("6. Loss backward")
			self.dataloader.switch_stack()
		else:
			y_pred,y_reg = self.model(x,
					encoded_input=True,
					dates=pr.get_exam_dates(),
					bdate=pr.get_birth_dates()[0],
					static_input = pr.get_static_inputs(),
					target_confound = self.dataloader.tl()
				)
			if self.regress:
				self.loss_regressor = \
					self.loss_function(y_reg,pr.get_C(
						confound=self.dataloader.tl(),
						return_lim=self.return_lim))
			else:
				self.loss_regressor = \
					self.loss_function(y_reg,pr.get_C_dud(
						confound=self.dataloader.tl(),
						return_lim=self.return_lim))
				
			self.log_time("4. Regress loss")
			self.loss_regressor.backward()
			self.log_time("5. Regress loss backward")
		
		if self.loss_image_dir is not None:
			self.loss_tracker.update(
					y_class_loss = self.loss_Y,
					y_reg_loss = self.loss_regressor,
					xs = self.index,
					y_kl_loss = None if not self.model.variational \
									else float(torch.mean(self.loss_kl)),
					y_class_adv_loss = self.loss_C_dud,
					target = tl_v,
					label_or_confound = l_or_c,
					label_m = float(torch.mean(self.label_m)) if float(torch.mean(self.label_m)) != 0 else None,
					conf_m = float(torch.mean(self.conf_m)) if float(torch.mean(self.conf_m)) != 0 else None
			)
			self.log_time("7. Loss record append")

		self.dataloader.rotate_labels()
		
		# Apply updates to the optimizer
		if self.index % self.batch_size == 0 and self.index != 0:
			if self.update_classifier:
				self.optimizer.step()
				self.optimizer.zero_grad()
				self.model.classifier_freeze()
				if self.dataloader.stack == "Labels":
					self.dataloader.switch_stack()
				#if self.use_triplet:
				self.model.reset_encoded()
			else:
				self.optimizer_reg.step()
				self.optimizer_reg.zero_grad()
				self.model.regressor_freeze()
				if self.dataloader.stack == "Confounds":
					self.dataloader.switch_stack()
			self.log_time("8. One step")
			self.update_classifier = not self.update_classifier
		
		if self.index % self.save_latest_freq == 0 and self.index != 0:
			if self.checkpoint_dir is not None:
				torch.save({
					'model_state_dict' : self.model.state_dict(),
					'optimizer_state_dict' : self.optimizer.state_dict(),
					'optimizer_reg_state_dict' : self.optimizer_reg.state_dict()
					},
					self.model_file
				)
				self.log_time("9. Save state dict")
			if self.loss_image_dir is not None:
				self.loss_tracker.plot()
				self.loss_tracker.plot(smooth=True,log_yscale=False,temptitle="_smooth")
				self.loss_tracker.plot(smooth=False,log_yscale=True,temptitle="_log")
				self.loss_tracker.plot(smooth=True,log_yscale=True,temptitle="_slog")
				self.loss_tracker.save()
				#os.makedirs(os.path.join(self.loss_image_dir,"dist_ims"),exist_ok=True)
				#self.model.full_encoded(os.path.join(self.loss_image_dir,"dist_ims","dist_%d.png" % self.index))
				self.log_time("10. Plot fig")
		self.index += 1

	def test(self):
		return

class LossTracker:
	def __init__(self,name,loss_image_dir):
		self.loss_image_dir=loss_image_dir
		os.makedirs(self.loss_image_dir,exist_ok=True)
		self.name=name
		self.loss_image_file = os.path.join(self.loss_image_dir,
										f"{self.name}_loss.png")
		self.loss_vals_file  = os.path.join(self.loss_image_dir,
										f"{self.name}_vals.json")
		self.y_class_loss = {}
		self.xs = {}
		self.y_reg_loss = {}
		self.y_kl_loss = []
		self.y_class_adv_loss = {}
		self.label_m = []
		self.conf_m = []
		self.label_m_xs = []
		self.conf_m_xs = []
		if os.path.isfile(self.loss_vals_file):
			self.load()
			self.plot()
			self.plot(smooth=True,log_yscale=False,temptitle="_smooth")
			self.plot(smooth=False,log_yscale=True,temptitle="_log")

	def add_target(self,target,label_or_confound):
		if target not in self.xs: self.xs[target] = []
		if label_or_confound == "Labels":
			if target not in self.y_class_loss: self.y_class_loss[target]=[]
		elif label_or_confound == "Confounds":
			if target not in self.y_class_adv_loss:
				self.y_class_adv_loss[target] = []
			if target not in self.y_reg_loss:
				self.y_reg_loss[target] = []
		else:
			raise Exception(
				"Invalid input for label_or_confound: %s" % label_or_confound)
	def update(self,xs,
					y_class_loss,
					y_reg_loss,
					y_class_adv_loss,
					target,
					label_or_confound,
					y_kl_loss=None,
					label_m = None,
					conf_m = None):
		y_class_loss,y_reg_loss,y_class_adv_loss = \
			float(y_class_loss),float(y_reg_loss),float(y_class_adv_loss)
		if any([np.isnan(_) for _ in [y_class_loss,y_reg_loss,y_class_adv_loss]]):
			raise Exception("NaN loss")
		if y_kl_loss is not None:
			y_kl_loss = float(y_kl_loss)
		if label_m is not None:
			label_m = float(label_m)
		if conf_m is not None:
			conf_m = float(conf_m)
		self.add_target(target,label_or_confound)

		self.xs[target].append(xs)
		if label_or_confound == "Labels":
			self.y_class_loss[target].append(y_class_loss)
		elif label_or_confound == "Confounds":
			self.y_reg_loss[target].append(y_reg_loss)
			self.y_class_adv_loss[target].append(y_class_adv_loss)
		else:
			raise Exception("Invalid label_or_confound: %s" % label_or_confound)
		if y_kl_loss is not None:
			self.y_kl_loss.append(y_kl_loss)
		if label_m is not None:
			self.label_m.append(label_m)
			self.label_m_xs.append(xs)
		if conf_m is not None:
			self.conf_m.append(conf_m)
			self.conf_m_xs.append(xs)
	def _get_last_loss(self,d):
		if len(d) == 0: return 0
		if all([len(d[_]) == 0 for _ in d]):
			return 0
		mtarget = None
		for target in d:
			if mtarget is None: mtarget=target
			else:
				if self.xs[target][-1] > self.xs[mtarget][-1]:
					mtarget = target
		return d[mtarget][-1]
	
	def get_last_y_reg_loss(self):
		return self._get_last_loss(self.y_reg_loss)
	def get_last_y_kl_loss(self):
		if len(self.y_kl_loss) == 0: return None
		return self.y_kl_loss[-1]
	def get_last_y_class_adv_loss(self):
		return self._get_last_loss(self.y_class_adv_loss)
	def get_last_y_class_loss(self):
		return self._get_last_loss(self.y_class_loss)
	def get_last_xs(self):
		return self._get_last_loss(self.xs)
	def load(self):
		with open(self.loss_vals_file,'r') as fileobj:
			state_dict = json.load(fileobj)
		self.xs = state_dict['xs']
		self.y_class_adv_loss = state_dict['y_class_adv_loss']
		self.y_reg_loss = state_dict['y_reg_loss']
		self.y_kl_loss = state_dict['y_kl_loss']
		self.y_class_loss = state_dict['y_class_loss']
		if 'label_m' in state_dict:
			self.label_m = state_dict['label_m']
			self.label_m_xs = state_dict['label_m_xs']
			self.conf_m = state_dict['conf_m']
			self.conf_m_xs = state_dict['conf_m_xs']
		
	def title(self,target):
		lab = target
		if lab == "ScanningSequence": lab = "ScanSeq"
		if lab == "Ages_Buckets": lab = "Ages"
		if lab == "SexDSC": lab = "Sex"
		if lab == "MRModality": lab = "MRI type"
		if lab == "AlzStage": lab = "Dementia"
		if lab == "ICD_one_G35": lab = "MS"
		if lab == "ICD_one_G40": lab = "Epilepsy"
		if lab == "ICD_one_G20": lab = "Parkin."
		if "_one_" in lab: lab = lab.replace("_one_"," ")
		return lab
	def smooth(self,arr):
		if len(arr) < 200: return arr
		newarr = np.zeros((len(arr),))
		for i,v in enumerate(arr):
			min_i = max(0,i - 100)
			max_i = min(len(arr)-1,i + 100)
			newarr[i] = np.mean(arr[min_i:max_i])
		return newarr
	def plot(self,smooth=False,log_yscale=False,temptitle=None):
		plt.clf()
		fig = plt.figure()
		ax = plt.subplot()
		for target in self.y_class_loss:
			y = self.y_class_loss[target]
			if smooth:
				y = self.smooth(y)
			ax.plot(self.xs[target],y,
				label="%s (L)" % self.title(target)
			)
		for target in self.y_reg_loss:
			y = self.y_reg_loss[target]
			if smooth: y = self.smooth(y)
			p = ax.plot(self.xs[target],
					y,
					label = "%s (C)" % self.title(target)
			)
			if isinstance(p,list): p = p[0]
			y = self.y_class_adv_loss[target]
			if smooth: y = self.smooth(y)
			ax.plot(self.xs[target],
				y,
				c=p.get_color(),
				linestyle='dashed',
				label="%s (A)" % self.title(target)
			)
		if len(self.y_kl_loss) > 0:
			y = self.y_kl_loss
			if smooth: y = self.smooth(y)
			ax.plot(list(range(len(self.y_kl_loss))),
				y,
				label="KL")
		if len(self.conf_m) > 0:
			y = self.conf_m
			if smooth: y = self.smooth(y)
			ax.plot(self.conf_m_xs,[_  for _ in y],linestyle='dashdot',
				label="Confound M")
		if len(self.label_m) > 0:
			y = self.label_m
			if smooth: y = self.smooth(y)
			ax.plot(self.label_m_xs,[_ * -1  for _ in y],linestyle='dashdot',
				label="Label M")
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		
		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		
		plt.xlabel("Iterations")
		if log_yscale:
			plt.yscale("log")
			plt.ylabel("log(Loss)")
		else:
			plt.ylabel("Loss")
		if temptitle is not None:
			plt.savefig(self.loss_image_file.replace(".png",f"{temptitle}.png"),
				bbox_inches='tight')
		else:
			plt.savefig(self.loss_image_file,bbox_inches='tight')

		plt.clf()
		plt.cla()
		plt.close(fig)
		plt.close('all')

	def save(self):
		state_dict = {
			'xs':self.xs,
			'y_class_loss':self.y_class_loss,
			'y_reg_loss':self.y_reg_loss,
			'y_class_adv_loss':self.y_class_adv_loss,
			'y_kl_loss':self.y_kl_loss,
			'label_m':self.label_m,
			'label_m_xs':self.label_m_xs,
			'conf_m':self.conf_m,
			'conf_m_xs':self.conf_m_xs
		}
		with open(self.loss_vals_file,'w') as fileobj:
			json.dump(state_dict,fileobj)
