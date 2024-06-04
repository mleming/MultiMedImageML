import torch,os
from torch import nn
import numpy as np
from .Records import BatchRecord
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

"""
Used to apply training updates to the Multi Input Model. May also output graphs
of the training loss
"""

class MultiInputTrainer():
	def __init__(self,
		model,
		lr=1e-5,
		betas = (0.5,0.999),
		loss_function = nn.MSELoss(),
		batch_size = 64,
		regress = True,
		loss_image_dir = None,
		checkpoint_dir = None,
		name = 'experiment_name',
		verbose = False,
		save_latest_freq = 100):
		
		# Core variables for training the self.model
		self.model = model
		self.name = name

		self.optimizer = torch.optim.Adam(
			self.model.classifier_parameters(),
			betas = betas,
			lr= lr
		)
		self.optimizer_reg = torch.optim.Adam(
			self.model.regressor.parameters(),
			betas = betas,
			lr = lr
		)

		# Load saved or pretrained models
		self.checkpoint_dir = checkpoint_dir
		if self.checkpoint_dir is not None:
			os.makedirs(self.checkpoint_dir,exist_ok=True)
			self.model_file = os.path.join(
				self.checkpoint_dir,
				'%s.pt' % self.name)
			if os.path.isfile(self.model_file):
				state_dicts = torch.load(self.model_file)
				self.model.load_state_dict(state_dicts['model_state_dict'])
				self.model.regressor_freeze()
				self.optimizer.load_state_dict(
					state_dicts['optimizer_state_dict']
				)
				self.optimizer_reg.load_state_dict(
					state_dicts['optimizer_reg_state_dict']
				)
		self.loss_function = loss_function
		self.batch_size = batch_size
		self.regress = regress
		self.index = 0
		self.one_step = True
				
		# Outputs images of the loss function over time
		self.loss_image_dir = loss_image_dir
		if self.loss_image_dir is not None:
			os.makedirs(self.loss_image_dir,exist_ok=True)
			self.loss_image_file = os.path.join(self.loss_image_dir,
											f"{self.name}_loss.png")
			self.loss_vals_file  = os.path.join(self.loss_image_dir,
											f"{self.name}_vals.npy")
			self.pids_read = set()
			self.x_files_read = set()
			if os.path.isfile(self.loss_vals_file):
				_ = np.load(self.loss_vals_file)
				self.xs,self.ys,self.ys_2,self.ys_c_dud = \
						[list(_[i,:]) for i in range(4)]
				if self.model.variational:
					self.ys_kl = list(_[i,4])
	
				self.loss_Y          = self.ys[-1]
				self.loss_C_dud      = self.ys_c_dud[-1]
				self.loss_regressor  = self.ys_2[-1]
				if model.variational:
					self.loss_kl = self.ys_kl[-1]
				self.loss_classifier = self.loss_Y + self.loss_C_dud
			else:
				self.xs,self.ys,self.ys_2,self.ys_kl,self.ys_c_dud =\
							[[] for _ in range(5)]
		
				self.loss_Y          = 0
				self.loss_C_dud      = 0
				self.loss_classifier = 0
				self.loss_regressor  = 0
				if model.variational:
					self.loss_kl = 0
			
		self.verbose = verbose
		self.save_latest_freq = save_latest_freq
		
	def loop(self, pr: BatchRecord, dataloader = None):
		assert(isinstance(pr,BatchRecord))
		x = self.model(pr,return_encoded=True)
		y_pred,y_reg = self.model(x,
				encoded_input=True,
				dates=pr.get_exam_dates(),
				bdate=pr.get_birth_dates()[0])
		if self.one_step:
			Y = pr.get_Y()
			self.loss_Y = self.loss_function(y_pred,Y)
			self.loss_C_dud = self.loss_function(y_reg,pr.get_C_dud())
			self.loss_classifier = self.loss_Y + (self.loss_C_dud)
			if self.model.variational:
				self.loss_kl = self.model.kl * 0.0005
				self.loss_classifier = self.loss_classifier + loss_kl
			self.loss_classifier.backward()
		else:
			if self.regress:
				self.loss_regressor = \
					self.loss_function(y_reg,pr.get_C())
			else:
				self.loss_regressor = \
					self.loss_function(y_reg,pr.get_C_dud())
			self.loss_regressor.backward()
		
		if self.loss_image_dir is not None:
			
			self.ys.append(float(self.loss_Y))
			self.ys_2.append(float(self.loss_regressor))
			self.xs.append(self.index)
			if self.model.variational:
				self.ys_kl.append(float(self.loss_kl))
			self.ys_c_dud.append(float(self.loss_C_dud))

		if self.verbose:
			if self.model.variational:
				print(("%d: Class: %.6f, KL: %.6f, Dud: "+\
					"%.6f, Reg: %.6f (%d, %d) | %s") % \
					(i,float(loss_Y), float(loss_kl), float(loss_C_dud),
					float(loss_regressor),len(self.pids_read),
					len(self.x_files_read),self.name))
			else:
				print("%d: Class: %.6f, Dud: %.6f, Reg: %.6f (%d, %d) | %s" % \
					(i,float(loss_Y), float(loss_C_dud),
					float(loss_regressor),len(self.pids_read),
					len(self.x_files_read),self.name))

		if self.index % self.batch_size == 0 and self.index != 0:
			if self.one_step:
				self.optimizer.step()
				self.optimizer.zero_grad()
				self.model.classifier_freeze()
				if dataloader is not None:
					dataloader.switch_stack()
			else:
				self.optimizer_reg.step()
				self.optimizer_reg.zero_grad()
				self.model.regressor_freeze()
				if dataloader is not None:
					dataloader.switch_stack()
			self.one_step = not self.one_step
		if self.index % self.save_latest_freq == 0 and self.index != 0:
			if self.checkpoint_dir is not None:
				torch.save({
					'model_state_dict' : self.model.state_dict(),
					'optimizer_state_dict' : self.optimizer.state_dict(),
					'optimizer_reg_state_dict' : self.optimizer_reg.state_dict()
					},
					self.model_file
				)
			if self.loss_image_dir is not None:
				plt.plot(list(range(len(self.xs))),self.ys,
					label="Classifier loss - label")
				if self.regress:
					plt.plot(list(range(len(self.xs))),
						self.ys_2,
						label="Regressor loss")
				if self.model.variational:
					plt.plot(list(range(len(self.xs))),
						self.ys_kl,
						label="KL Loss")
				plt.plot(list(range(len(self.xs))),
					self.ys_c_dud,label="Classifier loss - adversarial")
				plt.legend(loc='upper right')
				plt.savefig(self.loss_image_file)
				plt.clf()
				
				if self.model.variational:
					np.save(self.loss_vals_file,np.array([self.xs,
													self.ys,
													self.ys_2,
													self.ys_c_dud,
													self.ys_kl]))
				else:
					np.save(self.loss_vals_file,np.array([self.xs,
													self.ys,
													self.ys_2,
													self.ys_c_dud]))
				
		self.index += 1

	def test(self):
		return
