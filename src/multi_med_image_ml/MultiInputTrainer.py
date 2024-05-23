import torch
from torch import nn
from Records import BatchRecord

class MultiInputTrainer():
	def __init__(self,
		model,
		lr=1e-5,
		betas = (0.5,0.999),
		loss_function = nn.MSELoss(),
		batch_size = 64,
		regress = True):
		self.model = model
		self.optimizer = torch.optim.Adam(
			model.classifier_parameters(),
			betas = betas,
			lr= lr
		)
		self.optimizer_reg = torch.optim.Adam(
			model.regressor.parameters(),
			betas = betas,
			lr = lr
		)
		self.loss_function = loss_function
		self.batch_size = batch_size
		self.regress = regress
		self.index = 0
		self.one_step = True
	def loop(self,pr: BatchRecord,dataloader=None ):
		assert(isinstance(pr,BatchRecord))
		x = self.model(pr,return_encoded=True)
		y_pred,y_reg = self.model(x,
				encoded_input=True,
				dates=pr.get_exam_dates(),
				bdate=pr.get_birth_dates()[0])
		if self.one_step:
			loss_Y = self.loss_function(y_pred,pr.get_Y())
			loss_C_dud = self.loss_function(y_reg,pr.get_C_dud())
			loss_classifier = loss_Y + (loss_C_dud)
			if self.model.variational:
				loss_kl = self.model.kl * 0.0005
				loss_classifier = loss_classifier + loss_kl
			loss_classifier.backward()
		else:
			if self.regress:
				loss_regressor = \
					self.loss_function(y_reg,pr.get_C())
			else:
				loss_regressor = \
					self.loss_function(y_reg,pr.get_C_dud())
			loss_regressor.backward()
		self.index += 1
		if self.index % self.batch_size == 0:
			if self.one_step:
				self.optimizer.step()
				self.optimizer.zero_grad()
				self.model.classifier_freeze()
				if dataloader is not None:
					dataloader.switch_stack()
					#assert(set(opt.confounds) == \
					#	set(dataloader.label))
			else:
				self.optimizer_reg.step()
				self.optimizer_reg.zero_step()
				self.model.regressor_freeze()
				if dataloader is not None:
					dataloader.switch_stack()
					#assert(set(opt.label) == \
					#	set(dataloader.label))
			

