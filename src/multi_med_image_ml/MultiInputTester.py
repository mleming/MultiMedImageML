import torch,os
from torch import nn
from pathlib import Path
import numpy as np
import pandas as pd
import json
import dateutil
from .Records import BatchRecord
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
#from pytorch_grad_cam import GradCAMPlusPlus,GradCAM
from sklearn.metrics import auc,roc_curve
import glob
import warnings
from .utils import resize_np,is_nan
from adjustText import adjust_text
from scipy.special import softmax

# Tests either the model directly or the output files
class MultiInputTester:
	"""Used for testing the outputs of MultiInputModule.
	
	MultiInputTester abstracts many of the functions for testing DL models, including grad cam and group AUC outputs.
	
	Attributes:
		database (DataBaseWrapper) : Associated database for testing
		model (MultiInputModule): Model to be tested
		out_record_folder (str): Folder to output results (default None)
		checkpoint_dir (str): Folder that has model checkpoints (default none)
		name (str): Name of the model to be tested (default 'experiment_name')
		test_name (str): The name of the experiment (default "")
		database_key (str): Variable used when grouping data together for AUROC analysis
		min_pids (int):  (default 1)
		top_not_mean (bool): Given multiple AUC output files, this will select one randomly instead of coming up with the mean prediction of all of them
		include_inds (list): (default [0,1])
		same_patients (bool): If true, only plots AUC/Accuracy for patients that are equally divided between groups (default False)
		x_axis_opts (str): Whether the X axis of the plot should be "images", "patients", or "images_per_patient" (default: "images")
	"""
	
	def __init__(self,
		database,
		model,
		out_record_folder : str = None,
		checkpoint_dir : str = None,
		verbose : bool = False,
		name : str = 'experiment_name',
		test_name : str = "",
		include_inds : list = [0,1],
		return_confidence : bool = False):
		
		self.name=name
		self.checkpoint_dir=checkpoint_dir
		self.model = model
		self.return_confidence = return_confidence
		self.model_file = os.path.join(
			self.checkpoint_dir,'%s.pt' % self.name)
		if os.path.isfile(self.model_file):
			self.state_dicts = torch.load(self.model_file)
			if 'model_state_dict' not in self.state_dicts:
				x = {}
				x['model_state_dict'] = self.state_dicts
				self.state_dicts = x
			if 'encoded_record' in self.state_dicts:
				self.state_dicts['model_state_dict']['encoded_record'] = self.state_dicts['encoded_record']
				del self.state_dicts['encoded_record']
			self.model.load_state_dict(self.state_dicts['model_state_dict'])
		else:
			self.state_dicts = None
		self.model.eval()
		self.out_record_folder = out_record_folder
		os.makedirs(self.out_record_folder,exist_ok=True)
		self.name = name
		self.test_name = test_name
		if self.out_record_folder is not None and self.name is not None:
			self.stats_record = {}
			for l in self.model.classifiers:
				self.stats_record[l] = _StatsRecord(
					os.path.join(self.out_record_folder,"json",l),
					self.name,
					test_name=self.test_name)
		self.pid_records = None
		self.remove_inds = []
		self.include_cbar = True
		self.mv_limit = 0.5
		self.x_file_pid = False
		self.database = database
		self.include_inds = include_inds
		self.verbose = verbose
		self.grad_cam_group = {}
		self.encoding_Xfile_record = None
		self.encoding_record = None
	def out_state_dict(self):
		if self.state_dicts is not None:
			self.state_dicts['model_state_dict'] = self.model.state_dict()
			torch.save(self.state_dicts,self.model_file)
	def record(self,reset=False):
		for s in self.stats_record:
			self.stats_record[s].record(reset=reset)
	def acc(self,target_label,database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False,
						save = False,
						ind = 0,
						acc_or_auc = "acc",
						min_pids=1,
						top_not_mean=False):
		if "ICD" in target_label: ind = 1
		if acc_or_auc == "auc":
			group_dict = self.pid_records[target_label].auc(ind=ind,
					database_key=database_key,
					opt=opt,
					divides=divides,
					same_pids_across_groups=same_pids_across_groups,
					min_pids=min_pids,top_not_mean=top_not_mean)
		elif acc_or_auc == "acc":
			group_dict = self.pid_records[target_label].acc(database_key=database_key,
					opt=opt,
					divides=divides,
					same_pids_across_groups=same_pids_across_groups,
					min_pids = min_pids,top_not_mean=top_not_mean)
		else:
			raise Exception("Invalid arg for acc_or_auc: %s" % acc_or_auc)
		if save:
			out_json_folder = os.path.join(self.out_record_folder,
				"json_res",target_label)
			os.makedirs(out_json_folder,exist_ok=True)
			database_key_title = database_key.replace("/","_")
			out_json_file = os.path.join(
				out_json_folder,
				f"{database_key_title}_{acc_or_auc}_{opt}_same_pids_{same_pids_across_groups}.json")
			with open(out_json_file,'w') as fileobj:
				json.dump(group_dict,fileobj,indent=4)
		return group_dict
	def plot(self,
			target_label,
			ind = 0,
			x_axis_opts = "images",
			acc_or_auc = "auc",
			database_key = None,
			opt = None,
			divides = None,
			same_pids_across_groups = False,
			min_pids = 1,
			do_adjust_text=True,top_not_mean=False):
		if "ICD" in target_label: ind = 1
		group_set = self.acc(target_label,
					database_key = database_key,
					opt = opt,
					divides = divides,
					same_pids_across_groups = same_pids_across_groups,
					ind = ind,
					acc_or_auc = acc_or_auc,
					min_pids = min_pids,top_not_mean=top_not_mean
				)
		assert(group_set is not None)
		if len(group_set) == 0:
			raise NotEnoughPatients("Not enough patients in plot...?")
		plt.clf()
		texts = []
		max_c,min_c = -1 * float('inf'),float('inf')
		cmap = cm.jet(range(255))
		for group in group_set:
			if group is None: continue
			if group_set[group]["confidence"] == np.nan or group_set[group]["confidence"] is None: continue
			max_c = max(group_set[group]["confidence"],max_c)
			min_c = min(group_set[group]["confidence"],min_c)
		min_c,max_c = 0.8,1.9
		for group in group_set:
			if group is None: continue
			x = group_set[group][acc_or_auc]
			if is_nan(x): continue
			if group_set[group]["patients"] < min_pids:
				continue
			if x_axis_opts == "images_per_patient":
				if group_set[group]["patients"] == 0:
					y = 0
				else:
					y = group_set[group]["images"]/group_set[group]["patients"]
			else:
				y = group_set[group][x_axis_opts]
				if x_axis_opts == "images": assert(y > 0)
			confidence = group_set[group]["confidence"]
			if confidence is None or confidence == np.nan: continue
			confidence = max(min_c,confidence)
			confidence = min(max_c,confidence)
			confidence = ((confidence - min_c) / (max_c - min_c))
			try:
				plt.scatter(y,x,c=cmap[int(confidence * 254)],label=group,s=100,edgecolors='black')
			except ValueError:
				continue
			if len(group_set) <=15:
				texts.append(plt.text(y,x,group[:20],fontsize=16))
			else: do_adjust_text = False
		out_plot_folder = os.path.join(self.out_record_folder,"plots",target_label)
		os.makedirs(out_plot_folder,exist_ok=True)
		database_key_title = database_key.replace("/","_")
		out_plot_file = os.path.join(
			out_plot_folder,
			f"{database_key_title}_{x_axis_opts}_{acc_or_auc}_{opt}_same_pids_{same_pids_across_groups}.png")
		xlabel = "# " + x_axis_opts.replace("_"," ").title()
		if same_pids_across_groups:
			assert(group_set[group]["patients"] >= min_pids)
			xlabel = xlabel + " (%s Patients)" % (group_set[group]["patients"])
		plt.xlabel(xlabel,fontsize=16)
		plt.ylabel(acc_or_auc.upper(),fontsize=16)
		if do_adjust_text:
			adjust_text(texts)
		plt.savefig(out_plot_file)

	def loop(self,pr: BatchRecord,target_label=None,record_encoding=False):
		"""Tests one input and saves it.
		
		Args:
			pr (BatchRecord) : Image batch
		
		"""
		y_pred = self.model(pr,
			return_regress = True,
			target_label=target_label,
			return_confidence = self.return_confidence,
			dataloader = self)
		if self.return_confidence:
			y_pred,confidence = y_pred
		else: confidence = None
		if record_encoding:
			self.record_encodings(pr.get_X_files())
		if isinstance(y_pred,tuple):
			y_pred,c_pred = y_pred
		else:
			c_pred = torch.Tensor(np.zeros(y_pred.shape))
		if target_label is None and len(self.model.classifiers) == 1:
			for _ in self.model.classifiers: target_label = _
		if pr.batch_by_pid:
			self.stats_record[target_label].update(
				pr.get_Y(label=target_label),
				y_pred,
				pr.pid,
				pr.get_X_files(),
				age_encode = self.model.encode_age,
				static_inputs = [] if not self.model.use_static_input \
					else pr.get_static_inputs(),
				confidence = confidence
			)
		else:
			for i,im in enumerate(pr.image_records):
				self.stats_record[target_label].update(
					im.get_Y(label=target_label),
					y_pred[i,...],
					im.get_ID(),
					im.npy_file,
					age_encode = self.model.encode_age,
					static_inputs = [] if not self.model.use_static_input \
					else pr.get_static_inputs(),
					confidence = confidence
				)
		return
	def read_json(self,target_label):
		"""Reads all json files output by MultiInputTester."""
		
		if self.pid_records is None:
			self.pid_records = {}
		if target_label not in self.pid_records:
			self.pid_records[target_label] = _AllRecords(self.database,
				remove_inds = self.remove_inds,
				mv_limit = self.mv_limit,
				include_inds = self.include_inds,
				name = self.name,
				verbose = self.verbose)
		json_files = glob.glob(os.path.join(self.out_record_folder,"json",
			target_label,"*.json"))
		for json_file in json_files:
			try:
				with open(json_file,'r') as fileobj:
					json_dict = json.load(fileobj)
			except:
				if self.verbose: print("Error in opening %s" % json_file)
				continue
			X_files = []
			Ys = []
			y_preds = []
			for pid in json_dict:
				for mm in range(len(json_dict[pid])):
					xf = np.array(json_dict[pid][mm]["X_files"]).flatten()
					yf = np.array(json_dict[pid][mm]["Y"])#.flatten()
					for i in self.remove_inds:
						#print(yf)
						if np.any(yf[i,:] == 1):
							#print(xf)
							break
					yf = yf.flatten()
					ypf = np.array(json_dict[pid][mm]["y_pred"]).flatten()
					if "age_encode" in json_dict[pid][mm]:
						age_encode = json_dict[pid][mm]["age_encode"]
					else:
						age_encode = False
					if "static_inputs" in json_dict[pid][mm]:
						static_inputs = json_dict[pid][mm]["static_inputs"]
					else:
						static_inputs = []
					if "confidence" in json_dict[pid][mm]:
						confidence = float(np.max(json_dict[pid][mm]["confidence"]))
					else:
						confidence = None
					self.pid_records[target_label].add_record(
						pid + str(np.argmax(yf)),
						xf,yf,ypf,self._json_title_parse(json_file),
						age_encode=age_encode,static_inputs=static_inputs,
						confidence = confidence)
		#if self.same_patients:
		#	self.pid_records.merge_group_pids()

#	def read_encodings(self):
#		out_encoding_folder = os.path.join(self.out_record_folder,'encodings')
#		if not os.path.isdir(out_encoding_folder):
#			return
#		out_encoding_file = os.path.join(out_encoding_folder,"encodings.json")
#		if not os.path.isfile(out_encoding_file): return
#		with open(out_encoding_file,'r') as fileobj:
#			self.encoding_record = json.load(fileobj)
#		for X_file in self.encoding_record:
#			self.encoding_record[X_file] = np.array(
#				self.encoding_record[X_file])
		
#	def record_encodings(self,X_files):
#		if self.encoding_record is None:
#			self.encoding_record = {}
#		assert(len(X_files) == self.model.saved_encoding.shape[0])
#		for i,X_file in enumerate(X_files):
#			self.encoding_record[X_file] = \
#				np.squeeze(self.model.saved_encoding[i,...])

	def save_encodings(self):
		out_encoding_folder = os.path.join(self.out_record_folder,'encodings')
		os.makedirs(out_encoding_folder,exist_ok=True)
		out_encoding_file = os.path.join(out_encoding_folder,"encodings.json")
		
		for X_file in self.encoding_record:
			self.encoding_record[X_file] = \
				[float(_) for _ in self.encoding_record[X_file]]
		with open(out_encoding_file,'w') as fileobj:
			json.dump(self.encoding_record,fileobj)
		for X_file in self.encoding_record:
			self.encoding_record[X_file] = np.array(
						self.encoding_record[X_file])
	
	def _get_encoding_arrs(self):
		X_files = []
		encode_arr = []
		for X_file in self.encoding_record:
			X_files.append(X_file)
			encode_arr.append(self.encoding_record[X_file])
		encode_arr = np.array(encode_arr)
		return X_files,encode_arr
	def _get_ml_model(self,encode_arr,ml_model):
		if ml_model == "pca":
			from sklearn.decomposition import PCA
			pca = PCA(n_components = 2)
			encode_2dim = pca.fit_transform(encode_arr)
			return encode_2dim
		elif ml_model == "autoencoder":
			from .models import AutoEncoder1D
			ae = AutoEncoder1D(encode_arr.shape[1])
			loss_fn = nn.MSELoss()
			optimizer = torch.optim.Adam(ae.parameters(),lr=1e-5)
			encode_arr = np.expand_dims(encode_arr,axis=1)
			for j in range(20):
				running_loss = 0
				for i in range(encode_arr.shape[0] // 64):
					optimizer.zero_grad()
					batch = torch.tensor(encode_arr[i*64:(i+1)*64,:,:]).float()
					latents,outputs = ae(batch)
					loss = loss_fn(outputs,batch)
					loss.backward()
					optimizer.step()
					running_loss += loss.item()
				print("%d: Loss: %.6f" % (j,float(running_loss) / (encode_arr.shape[0] // 64)))
			encode_2dim,_ = ae(torch.tensor(encode_arr).float())
			encode_2dim = encode_2dim.detach().cpu().numpy()
			encode_2dim = np.squeeze(encode_2dim)
			return encode_2dim
		else:
			raise Exception("Invalid option: %s" % ml_model)
	def pca_analysis(self,database_keys:list,ml_model="pca"):
		out_encoding_folder = os.path.join(self.out_record_folder,'encodings')
		X_files,encode_arr = self._get_encoding_arrs()
		encode_2dim = self._get_ml_model(encode_arr,ml_model)
		xs,ys = encode_2dim[:,0],encode_2dim[:,1]
		for database_key in database_keys:
			if "/" in database_key:
				database_key,alt = database_key.split("/")
			else:
				alt = None
			out_encoding_plot = os.path.join(out_encoding_folder,
				f"{database_key}_encoding_plot.png")
			var_vals = [self.database.loc_val(X_file,database_key) \
							for X_file in X_files]
			for i,val in enumerate(var_vals):
				if is_nan(val) and alt is not None:
					var_vals[i] = self.database.loc_val(X_files[i],alt)
			var_vals = list(map(lambda k : "None" if is_nan(k) else k,var_vals))
			plt.clf()
			fig,ax = plt.subplots()
			colors = ['red','blue','green','purple',
					'chocolate','papayawhip','palegreen','goldenrod',
					'mediumaquamarine','maroon','slategrey','lightgreen',
					'darkseagreen','teal','beige','olive','deepskyblue',
					'mediumorchid','crimson','navy','indigo','azure',
					'skyblue','deepskyblue','darkolivegreen','yellow',
					'chartreuse','tomato']
			uvals = sorted(list(np.unique(var_vals)),key=lambda k: var_vals.count(k),reverse=True)
			for i,val in enumerate(uvals):
				ix = np.where(np.array(var_vals) == val)
				x_ = np.array(xs)[ix]
				y_ = np.array(ys)[ix]
				ind = uvals.index(val)
				color = colors[ind % len(colors)]
				ax.scatter(x_,y_,c = color,alpha=0.5,label=val)
			if len(uvals) < len(colors):
				ax.legend()
			if ml_model == "pca":
				plt.xlabel("Principal Component 1")
				plt.ylabel("Principal Component 2")
			elif ml_model == "autoencoder":
				plt.xlabel("Latent Dim 1")
				plt.ylabel("Latent Dim 2")
			plt.title(f"{database_key}")
			plt.savefig(out_encoding_plot)

	def _json_title_parse(self,json_file):
		return "_".join(
				os.path.basename(json_file).replace('.json','').split("_")[:-1]
			)

	def attn_map_vis(self, att_mat : list,X_dim,patch_size=16):
		if att_mat[0].size()[0] != 1:
			return [self.attn_map_vis([_[i,:,:,:].unsqueeze(0) for _ in att_mat],X_dim,patch_size)[0] for i in range(att_mat[0].size()[0])]
		att_mat = torch.stack(att_mat).squeeze(1)
		# Average the attention weights across all heads.
		att_mat = torch.mean(att_mat, dim=1)
		# To account for residual connections, we add an identity matrix to the
		# attention matrix and re-normalize the weights.
		residual_att = torch.eye(att_mat.size(1))
		aug_att_mat = att_mat + residual_att
		aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
		# Recursively multiply the weight matrices
		joint_attentions = torch.zeros(aug_att_mat.size())
		joint_attentions[0] = aug_att_mat[0]
		
		for n in range(1, aug_att_mat.size(0)):
			joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
		# Attention from the output token to the input space.
		v = joint_attentions[-1]
		grid_size = [int(_ / patch_size) for _ in X_dim ]
		mask = v[0, 1:].reshape(grid_size).detach().numpy()
	#	mask = resize_np(mask,im.shape)
	#	#mask = cv2.resize(mask / mask.max(), im.size())[..., np.newaxis]
	#	result = (mask * im).astype("uint8")
		return [mask]


	def grad_cam(self,pr: BatchRecord,
		add_symlink: bool = True,
		grad_layer: int = 7,
		save : bool = True,
		database_key : str = None,
		target_label : str = None,
		confidence_thresh : float = float('inf'),
		register : bool = False) -> torch.Tensor:
		"""Outputs a gradient class activation map for the input record
		
		Args:
			pr (BatchRecord): Image batch to apply Grad-Cam to
			add_symlink (bool): If true, adds a symbolic link to the original image in the same folder as the grad-cam is stored in (default True)
			grad_layer (int):  (default 7)
			save (bool): Save the output to the results folder (default True)
		"""
		self.model.grad_layer = grad_layer
		if pr.image_records[0].Y_dim[0] > 1:
			raise Exception(
				("Grad Cam cannot be applied to" + \
				" multilabel models (Y_dim: %s)") % str(pr.Y_dim))
		
		
		#y_pred = self.model(pr.get_X(),dates=pr.get_dates(),bdate=pr.get_bdate(),static_input=pr.get_static_input()grad_eval=True)
		if self.model.vision_transformer:
			y_pred,attns = self.model(pr,target_label=target_label,output_attentions=True)
			y_pred,confidence = self.model(pr,target_label=target_label,return_confidence=True)
			if np.max([_.cpu() for _ in confidence]) > confidence_thresh:
				print([_.cpu() for _ in confidence])
				return
			attns = [a.cpu() for a in attns]
			t = self.attn_map_vis(attns,pr.image_records[0].X_dim)#pr.get_X())
			t = [np.expand_dims(a,axis=0) for a in t]
			t = np.concatenate(t,axis=0)
			del attns
			#t = torch.stack(t).cpu().detach().numpy()
		else:
			Y = pr.get_Y()
			
			y_pred = self.model(pr,target_label=target_label,grad_eval=True)
		
			ymax = Y.argmax(dim=2)
		
			y_pred[:,:,ymax].backward()
		
			gradients = self.model.get_activations_gradient()
		
			# pool the gradients across the channels
			pooled_gradients = torch.mean(gradients, dim=[2, 3, 4])
		
			# get the activations of the last convolutional layer
			activations = self.model.get_activations(pr.get_X()).detach()
		
			# weight the channels by corresponding gradients
			for j in range(activations.size()[0]):
				for i in range(activations.size()[1]):
				    activations[j, i, :, :, :] *= pooled_gradients[j,i]
		
			# average the channels of the activations
			heatmap = torch.mean(activations, dim=1)
		
			# relu on top of the heatmap
			# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
		
			t = heatmap.cpu().detach().numpy()

		for i in range(t.shape[0]):
			im = pr.image_records[i]
			
			npsqueeze = np.squeeze(t[i,...])
			if database_key is not None:
				if database_key == "all":
					if "all" not in self.grad_cam_group:
						self.grad_cam_group["all"] = npsqueeze
					else:
						self.grad_cam_group["all"] = \
							self.grad_cam_group["all"] + npsqueeze
				else:
					if database_key not in self.grad_cam_group:
						self.grad_cam_group[database_key] = {}
					group = self.database.loc_val(im.npy_file,database_key)
					if group is None: continue
					assert(group is not None)
					if group not in self.grad_cam_group[database_key]:
						self.grad_cam_group[database_key][group] = npsqueeze
					else:
						self.grad_cam_group[database_key][group] = \
							self.grad_cam_group[database_key][group] + npsqueeze

			if save:
				npsqueeze = resize_np(npsqueeze,im.X_dim)
				out_folder = os.path.join(self.out_record_folder,
								"grads",target_label,im.group_by)
				os.makedirs(out_folder,exist_ok=True)
				bname = os.path.splitext(os.path.basename(im.npy_file))[0]
				out_name = f"{bname}_grad.npy"
				orig_name = f"{bname}_orig.npy"
				if self.model.vision_transformer:
					with open(os.path.join(out_folder,f"{bname}_info.txt"),'w') as fileobj:
						fileobj.write(
							"Label: %s, Confidence: %f, Y: %s, y_pred: %s" % \
								(target_label,
								confidence[i],
								str(pr.get_Y(label=target_label)),
								str(y_pred))
						)
				np.save(os.path.join(out_folder,out_name),npsqueeze)
				if add_symlink and not os.path.isfile(
						os.path.join(out_folder,orig_name)):
					os.symlink(im.npy_file,os.path.join(out_folder,orig_name))
			#if register:
			#	nifti_file = im.filename
				
		return t
	
	def out_grad_cam_groups(self,prefix=None):
		if prefix is None:
			out_folder = os.path.join(self.out_record_folder,"grads")
		else:
			out_folder = os.path.join(self.out_record_folder,f"{prefix}_grads")
		for database_key in self.grad_cam_group:
			if database_key == "all":
				out_all_folder = os.path.join(out_folder,"all")
				os.makedirs(out_all_folder,exist_ok=True)
				npsqueeze = self.grad_cam_group["all"]
				npsqueeze = npsqueeze - npsqueeze.min()
				npsqueeze = npsqueeze / npsqueeze.max()
				np.save(os.path.join(out_all_folder,"all.npy"),npsqueeze)
			else:
				out_all_folder = os.path.join(out_folder,database_key)
				os.makedirs(out_all_folder,exist_ok=True)
				for group in self.grad_cam_group[database_key]:
					npsqueeze = self.grad_cam_group[database_key][group]
					npsqueeze = npsqueeze - npsqueeze.min()
					npsqueeze = npsqueeze / npsqueeze.max()
					np.save(os.path.join(out_all_folder,f"{group}.npy"),
						npsqueeze)

class NotEnoughPatients(Exception):
	def __init__(self, message):
		self.message = message
	
class _StatsRecord():
	def __init__(self,out_record_folder,_name,test_name=""):
		self.test_name=test_name
		self._name=_name
		self.out_record_folder = out_record_folder
		os.makedirs(out_record_folder,exist_ok=True)
		self.name = self.get_name(self.out_record_folder,self._name,self.test_name)
		self.x_files_read = set()
		self.pids_read = set()
		self.all_acc = None
		self.all_recon_score = None
		self.all_y_pred = None
		self.all_Y = None
		self.all_C = None
		self.all_c_pred = None
		self.out_record = {}
		self.out_conf_record = {}
		self.all_IDs = None
		self.out_C = False
		self.all_confidence = None
	def get_name(self,out_record_folder,name,test_name):
		if self.test_name != "":
			n,ext = os.path.splitext(name)
			name = n + "_" + self.test_name + ext
		num = 0
		while os.path.isfile(
				os.path.join(out_record_folder,"%s_%d.json" % (name,num))
			):
			num += 1
		Path(os.path.join(out_record_folder,"%s_%d.json" % (name,num))).touch()
		return "%s_%d" % (name,num)
	def update(self,Y,y_pred,ID,X_files,age_encode=False,static_inputs=[],
			confidence=None):
		if torch.is_tensor(Y): Y = Y.cpu().detach().numpy()
		if torch.is_tensor(y_pred): y_pred = y_pred.cpu().detach().numpy()
		#if torch.is_tensor(C): C = C.cpu().detach().numpy()
		#if torch.is_tensor(c_pred): c_pred = c_pred.cpu().detach().numpy()
		if confidence is not None and torch.is_tensor(confidence[0]): confidence = [ c.cpu().detach().numpy() for c in confidence]
		self.x_files_read = self.x_files_read.union(set(X_files))
		self.pids_read.add(ID)

		if len(Y.shape) == 2:
			Y = np.expand_dims(Y,axis=1)
		if len(y_pred.shape) == 2:
			y_pred = np.expand_dims(y_pred,axis=1)
		assert(len(Y.shape) == 3)
		assert(len(y_pred.shape) == 3)
		if self.all_Y is None and self.out_C: self.all_Y = Y
		elif self.out_C: self.all_Y = np.concatenate((self.all_Y,Y),axis=0)
		if self.all_y_pred is None and self.out_C: self.all_y_pred = y_pred
		elif self.out_C: self.all_y_pred = np.concatenate((self.all_y_pred,y_pred),axis=0)
		if self.all_C is None and self.out_C: self.all_C = C
		elif self.out_C: self.all_C = np.concatenate((self.all_C,C),axis=0)
		if self.all_c_pred is None and self.out_C: self.all_c_pred = c_pred
		elif self.out_C:
			self.all_c_pred = np.concatenate((self.all_c_pred,c_pred),axis=0)
		if self.all_IDs is None:
			self.all_IDs = set([ID])
		else:
			self.all_IDs.add(ID)
		if ID not in self.out_record:
			self.out_record[ID] = []
		if len(Y.shape) == 2:
			self.out_record[ID].append({
							'X_files' : [str(_) for _ in X_files],
							'Y' : [float(_) for _ in list(Y[0])],
							'y_pred' : [float(_) for _ in list(y_pred[0])],
							'age_encode' : age_encode,
							'static_inputs' : [str(_) for _ in list(static_inputs)]
							})
		else:
			self.out_record[ID].append({
							'X_files' : [str(_) for _ in X_files],
							'Y' : [[float(Y[:,i,j]) for i in range(y_pred.shape[1]) ] for j in range(Y.shape[2])],
							'y_pred' : [[float(y_pred[:,i,j]) for i in range(y_pred.shape[1]) ] for j in range(y_pred.shape[2])],
							'age_encode' : age_encode,
							'static_inputs' : [str(_) for _ in list(static_inputs)]
							})
		if confidence is not None:
				self.out_record[ID][-1].update({
					'confidence' : [float(_) for _ in list(confidence)]
			})
		if self.out_C:
			for l,X_file in enumerate(X_files):
				if X_file not in self.out_conf_record: self.out_conf_record[X_file] = {}
				self.out_conf_record[X_file] = {
						'C' : [[float(C[l,j,k]) for k in range(C.shape[2])] for j in range(C.shape[1])],
						'c_pred': [[float(c_pred[l,j,k]) for k in range(c_pred.shape[2])] for j in range(c_pred.shape[1])]
					}
	def output_auc(self):
		self.all_auroc = []
		self.all_c_auroc = []
		for j in range(self.all_Y.shape[1]):
			if True:
				cc = []
				for k in range(min(self.all_Y.shape[2],self.all_y_pred.shape[2])):
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						fpr, tpr, thresholds = roc_curve(
											self.all_Y[:,j,k],
											self.all_y_pred[:,j,k])
						cc.append(auc(fpr,tpr))
				self.all_auroc.append(cc)
			else:
				if self.verbose:
					print("Mismatched shape")
					print("all_Y.shape: %s"%str(all_Y.shape))
					print("all_y_pred.shape: %s"%str(all_y_pred.shape))
		if self.out_C:
			for j in range(min(self.all_C.shape[1],self.all_c_pred.shape[1])):
				cc = []
				for k in range(min(self.all_C.shape[2],self.all_c_pred.shape[2])):
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						fpr, tpr, thresholds = roc_curve(
										self.all_C[:,j,k],
										self.all_c_pred[:,j,k])
						cc.append(auc(fpr,tpr))
				self.all_c_auroc.append(cc)
	def record(self,reset=False):
		self.out_record_file = os.path.join(
			self.out_record_folder,"%s.json" % self.name)
		self.out_record_file_txt=os.path.join(
			self.out_record_folder,"%s.txt" % self.name)
		self.out_conf_record_file = os.path.join(
			self.out_record_folder,"%s_conf.json" % self.name)
		self.out_conf_record_file_txt = os.path.join(
			self.out_record_folder,"%s_conf.txt" % self.name)
		#if self.all_Y is None:
		#	print("Nothing to record")
		#	return
		#if self.out_C:
		#	self.output_auc()
		with open(self.out_record_file,'w') as fileobj:
			json.dump(self.out_record,fileobj,indent=4)
		if reset:
			self.out_record = {}
			self.name = self.get_name(self.out_record_folder,
				self._name,self.test_name)
		if self.out_C:
			with open(self.out_record_file_txt,'w') as fileobj:
				fileobj.write( "%s - # files: %d; num patients: %d; %s; %s" % \
					(self.name,len(self.x_files_read),len(self.pids_read),
						str(self.all_auroc),str(self.all_c_auroc)))
		#json.dump(self.out_conf_record,open(self.out_conf_record_file,'w'),
		#	indent=4)
			with open(self.out_conf_record_file_txt,'w') as fileobj:
				for a in self.all_c_auroc:
					fileobj.write(str(a)+"\n")

class _FileRecord:
	def __init__(self,X_files,
				Y,
				y_pred,
				pid,
				database,
				age_encode=False,
				static_inputs=[],
				remove_inds=[],
				confidence : float = None):
		self.X_files = sorted(X_files)
		self.pid=pid #+ str(np.argmax(Y))
		self.Y = np.array(Y)
		self.confidence = confidence
		if len(self.Y.shape) == 1:
			self.Y = np.expand_dims(self.Y,axis=0)
		self.y_pred = np.array(y_pred)
		if len(self.y_pred.shape) == 1:
			self.y_pred = np.expand_dims(self.y_pred,axis=0)
		self.database=database
		self.dates = None
		for i in remove_inds:
			self.y_pred[i] = 0
		self.age_encode = age_encode
		self.static_inputs = static_inputs
	def get_group(self,
						database_key = None,
						opt = None,
						divides = None):
		"""Returns the group the file belongs to
		"""
		if len(self.static_inputs) == 0 or not self.age_encode: return None
		if self.confidence is not None and self.confidence > 1.125: return None
		if opt is None:
			return "all"
		elif opt == "name_num":
			return self.get_filetypes_name_num(divides = divides)
		elif opt == "diff_date":
			return self.get_filetypes_diff_date(divides = divides)
		elif opt == "name_mod":
			assert(database_key is not None)
			return self.get_filetypes_name_group(database_key=database_key) #self.group
		elif opt == "name_num_group":
			assert(database_key is not None)
			return self.get_filetypes_name_num_group(
							database_key=database_key) #self.group
		elif opt == "age_dem":
			foo = ("Age" if self.age_encode else "")
			foo = foo + " "
			foo = foo + ("Demo" if len(self.static_inputs) > 0 else "")
			if foo == " ": foo = "Only images"
			return foo
		else:
			raise Exception("Invalid argument: %s" % opt)
		#self.filetypes_name = " "
	def get_filetypes_diff_date(self,divides = None):
		"""Returns groups of files with dates more than X years apart
		"""
		
		if divides is None: divides = [5]
		
		if len(self.X_files) == 1:
			return None # "One Image"
		if self.dates is None:
			self.dates = [self.database.get_exam_date(X_file) for X_file in self.X_files]
		date_diff = max(self.dates) - min(self.dates)
		date_diff = date_diff.days / 365.25
		ret = self.get_divides(date_diff,divides)
		ret = ret + (" encoded" if self.age_encode else "")
		return ret
			
	def get_filetypes_name_num(self,divides = None):
		"""Returns groupings of numbers of files
		"""
		
		if divides is None: divides = [1,5,10,14]
		return self.get_divides(len(self.X_files),divides)
		
	def get_divides(self,val,divides):
		divides = sorted(divides)
		for i,d in enumerate(divides):
			s1 = 0 if i == 0 else divides[i-1]
			s2 = d
			if val <= s2:
				if s1+1 == s2:
					return "%d"%s2
				return "%d-%d" %(s1+1,s2)
		return "%s+" % divides[-1]
	def _get_group_set(self,database_key):
		if "/" in database_key:
			database_key,alt_key = database_key.split("/")
		else:
			alt_key = None
		
		ftypes = [self.database.loc_val(_,database_key) \
			for _ in self.X_files]
		
		for i,f in enumerate(ftypes):
			if is_nan(f,inc_null_str=True):
				if alt_key is not None:
					alt = self.database.loc_val(self.X_files[i],alt_key)
					if not is_nan(alt):
						ftypes[i] = alt
					else:
						ftypes[i] = "None"
				else:
					ftypes[i] = "None"
			
			ftypes[i] = ftypes[i].strip().upper()
		ftypes = set(ftypes)
		if "NONE" in ftypes:
			ftypes.remove("NONE")
		return ftypes
	def get_filetypes_name_num_group(self,database_key):
		"""Returns the number of different modalities in a given set of images
		"""
		ftypes = self._get_group_set(database_key)
		if len(ftypes) == 0: return None
		return str(len(ftypes))
		
	def get_filetypes_name_group(self,database_key):
		"""Returns a list of unique modalities in a given set of images
		"""
		ftypes = self._get_group_set(database_key)
		if len(ftypes) == 0: return None # or len(ftypes) > 2: return None
		#if len(ftypes) > 1: return None
		return "\n".join(sorted(list(ftypes)))
		
	def get_acc(self):
		return ((np.argmax(self.Y)) == (np.argmax(self.y_pred))).astype(float)

	def print_record(self,indent=0):
		print("-")
		print((indent * " ") + str(self.get_acc()))
		for X_file in self.X_files:
			print((indent*" ") + get_file_str(X_file))
		print("-")

class _PIDRecord:
	def __init__(self,
			pid,
			database,
			remove_inds=[]):
		self.remove_inds=remove_inds
		self.pid = pid
		self.file_records = []
		self.database = database
	def add_file_record(self,
			X_files,
			Y,
			y_pred,
			age_encode=False,
			static_inputs=[],
			confidence=None):
		f = _FileRecord(
				X_files,
				Y,
				y_pred,
				self.pid,
				self.database,
				age_encode = age_encode,
				static_inputs = static_inputs,
				remove_inds = self.remove_inds,
				confidence=confidence)
		self.file_records.append(f)
	
	def get_group_dict(self,
						database_key = None,
						opt = None,
						divides = None):
		""" Returns a dictionary of all groupings of the files for this 
		particular patient.
		"""
		
		group_dict = {}
		for f in self.file_records:
			group = f.get_group(
						database_key = database_key,
						opt = opt,
						divides = divides
					)
			#except:
			#	continue
			if group not in group_dict:
				group_dict[group] = []
			group_dict[group].append(f)
		return group_dict
	
	def get_mean_group(self,
						group,
						database_key = None,
						opt = None,
						divides = None,
						top_not_mean = False,
						mv_limit = 0.0):
		
		group_dict = self.get_group_dict(database_key = database_key,
										opt = opt,
										divides = divides)
		if group not in group_dict:
			return -1
		else:
			Ys = []
			yps = []
			X_file_set = set()
			for fr in sorted(group_dict[group],
				key=lambda k: len(k.X_files),
				reverse=False):
				Ys.append(fr.Y)
				yps.append(fr.y_pred)
				for X_file in fr.X_files: X_file_set.add(X_file)
				if top_not_mean:
					break
		Ys = np.array(Ys)
		yps = np.array(yps)
		Ys = np.mean(Ys,axis=0)
		yps = np.mean(yps,axis=0)
		Ys_ = np.zeros(yps.shape)
		Ys_[np.argmax(Ys)] = 1
		Ys = Ys_
		max_val = np.max(yps)
		if max_val < mv_limit:
			return -1
		if not np.all([(_ == 0 or _ == 1) for _ in Ys]):
			return -1
		return Ys,yps,len(X_file_set)
	
	def get_mean_accuracy(self,
						group,
						database_key = None,
						opt = None,
						divides = None,
						top_not_mean = False,
						mv_limit = 0.0):
		
		group_dict = self.get_group_dict(database_key = None,
										opt = None,
										divides = None)
		if group not in group_dict:
			return -1
		else:
			sum_ = 0.0
			c = 0
			for fr in group_dict[group]:
				sum_ += fr.get_acc()
				c += 1
			return sum_ / c
	def print_record(self):
		print("---")
		print(self.pid)
		print(" ")
		for group in self.file_records:
			print(group)
			print(self.get_mean_accuracy(group))
			for file_record in self.file_records[group]:
				file_record.print_record(indent=4)

class _AllRecords:
	def __init__(self,database,
			remove_inds=[],
			database_key="ProtocolNameSimplified",
			use_auc=True,
			include_cbar=False,
			min_pids=20,
			mv_limit=0.5,
			include_inds=[0,1],
			x_axis_opts="images",
			same_patients=False,
			name="experiment_name",
			verbose=False):
		self.use_auc=use_auc
		self.pid_records = {}
		self.group_pid_sets = {} # Sets of patients who have a certain group
		self.group_set = set()
		self.remove_inds = remove_inds
		self.database = database
		self.database_key = database_key
		self.include_cbar = include_cbar
		self.min_pids = min_pids
		self.mv_limit = mv_limit
		self.include_inds = include_inds
		self.x_axis_opts = x_axis_opts
		self.same_patients = same_patients
		self.name=name
		self.verbose=verbose
	def add_record(self,
			pid,
			X_files,
			Y,
			y_pred,
			group,
			age_encode=False,
			static_inputs=[],
			confidence : float = None):

		if pid not in self.pid_records:
			self.pid_records[pid] = _PIDRecord(pid,
					self.database,
					remove_inds=self.remove_inds)
		
		self.pid_records[pid].add_file_record(X_files,
						Y,
						y_pred,
						age_encode=age_encode,
						static_inputs=static_inputs,
						confidence = confidence)
	
	# Makes it so that this only outputs records of the same set of patients for
	# all considered modalities
	def get_group_dict(self,
						database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False,
						min_pids = 1
						):
		""" Returns a dictionary in which file records are sorted by groups
		"""
		group_dicts = {}
		group_pids = {}
		for pid in self.pid_records:
			group_dict = self.pid_records[pid].get_group_dict(
										database_key=database_key,
										opt=opt,
										divides=divides)
			for group in group_dict:
				if group not in group_dicts:
					group_dicts[group] = []
					group_pids[group] = set()
				group_dicts[group] = group_dicts[group] + group_dict[group]
				group_pids[group].add(pid)
		
		if same_pids_across_groups:
			intersect_pids = None
			groups = set()
			for group in sorted(group_pids,key = lambda k: len(group_pids[k]), reverse=True):
				if intersect_pids is None:
					
					intersect_pids = group_pids[group]
					if len(intersect_pids) < min_pids:
						intersect_pids = None
						break
				else:
					intersect = intersect_pids.intersection(group_pids[group])
					if len(intersect) < min_pids:
						continue
					else:
						intersect_pids = intersect
						groups.add(group)
				print("min pids, intersect pids")
				print(min_pids)
				print(len(intersect_pids))
				assert(len(intersect_pids) >= min_pids)
			if intersect_pids is None or len(intersect_pids) == 0:
				raise NotEnoughPatients("No patients that intersect between groups")
			
			group_dicts_same_pid = {}
			for group in groups:
				filerec_list = []
				for filerec in group_dicts[group]:
					if filerec.pid in intersect_pids:
						filerec_list.append(filerec)
				if len(filerec_list) > 0:
					group_dicts_same_pid[group] = filerec_list
			group_dicts = group_dicts_same_pid
		return group_dicts
	
	def get_group_pred_arrs(self,
						database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False,
						min_pids=1,
						top_not_mean=False
						):
		"""Returns set of prediction arrays with the 
		
		"""
		
		group_dict = self.get_group_dict(
							database_key = database_key,
							divides = divides,
							opt = opt,
							same_pids_across_groups = same_pids_across_groups,
							min_pids=min_pids)
		
		group_stat = {}
		
		for group in group_dict:
			group_stat[group] = {}
			for filerec in group_dict[group]:
				
				assert(len(filerec.Y.shape) == 2)
				assert(len(filerec.y_pred.shape) == 2)
				if filerec.pid not in group_stat[group]:
					group_stat[group][filerec.pid] = (filerec.Y,
						filerec.y_pred,
						1,
						set(filerec.X_files),
						filerec.confidence)
				elif not top_not_mean:
					Y_all,y_pred_all,image_count,image_set,all_confidence = \
						group_stat[group][filerec.pid]
					group_stat[group][filerec.pid] = (filerec.Y+Y_all,
											filerec.y_pred+y_pred_all,
											image_count+1,
											image_set.union(set(filerec.X_files)),
											filerec.confidence+all_confidence)
				

		# Mean by patient ID
		for group in group_stat:
			Y_all_group = None
			y_all_pred_group = None
			patient_count = 0
			image_count = 0
			image_sets = set()
			assert(len(group_stat[group]) > 0)
			if same_pids_across_groups:
				assert(len(group_stat[group]) >= min_pids)
			for pid in group_stat[group]:
				Y_all,y_pred_all,imc,image_set,m_conf = group_stat[group][pid]
				if Y_all_group is None:
					Y_all_group = Y_all / imc
					y_all_pred_group = y_pred_all / imc
					m_conf_group = [m_conf / imc]
				else:
					Y_all_group = np.concatenate(
						(Y_all_group,Y_all / imc),
						axis=0)
					y_all_pred_group = np.concatenate(
						(y_all_pred_group,y_pred_all / imc),
						axis=0)
					m_conf_group.append(m_conf / imc)
				assert(len(Y_all.shape) == 2)
				assert(len(y_pred_all.shape) == 2)
				patient_count += 1
				image_count += imc
				image_sets = image_sets.union(image_set)
			
			assert(len(Y_all_group.shape) == 2)
			assert(len(y_all_pred_group.shape) == 2)
			assert(Y_all_group.shape == y_all_pred_group.shape)
			group_stat[group] = (Y_all_group,
								y_all_pred_group,
								len(image_sets),
								patient_count,
								m_conf_group)
		
		return group_stat
	
	def auc(self,
				ind : int = 0,
				database_key : str = None,
				opt : str = None,
				divides : list = None,
				same_pids_across_groups : bool = False,
				min_pids=1,top_not_mean=False) -> dict:
		"""Returns the AUROC of the test
		
		Args:
			database_key (str): If set, returns the name of the variable to group by. For example, "Sex" would return the accuracy of male and female classification separately (default None)
			opt (str): Type of group to classify by. Can be opt "age_dem" (with and without age encoding and demographic inputs), "name_num" (Number of images input),"name_mod" (name of each group),"name_num_group" (number of modalities input into one classification),"diff_date" (span of dates between earliest and latest image)
			divides (list[int]): For opts that involve counting, this indicates the ranges to count
			same_pids_across_groups (bool): For multigroup accuracy, curates a set of PIDs that are present in each group.
			save (bool): Saves the results in a directory
		
		Returns:
			dictionary of groups, AUROC for that group, number of patients and images in that group
		"""
		group_stat = self.get_group_pred_arrs(
						database_key = database_key,
						opt=opt,divides=divides,
						same_pids_across_groups=same_pids_across_groups,
						min_pids=min_pids,top_not_mean=top_not_mean)
		group_aucs = {}
		for group in group_stat:
			Ys,y_preds,image_count,patient_count,m_confidence = group_stat[group]

			Ys = np.array(Ys)                                                                
			col_sel = np.zeros((Ys.shape[1],),dtype=bool)                                    
			col_sel = np.any(Ys,axis=0)                                                      
			if col_sel[2]: col_sel[0] = False                                                
			col_sel[-1] = False
			if np.sum(col_sel) != 2:
				continue
			assert(np.sum(col_sel) == 2)
			Ys = Ys[:,col_sel]
			y_preds = np.array(y_preds)[:,col_sel]

			y_preds = softmax(np.array(y_preds),axis=1)
			
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				fpr, tpr, thresholds = roc_curve(Ys[:,ind], y_preds[:,ind])
				auc_ = auc(fpr,tpr)
				group_aucs[group] = {"auc": auc_,
									"images":image_count,
									"patients":patient_count,
									"confidence":np.mean(m_confidence)}
		return group_aucs
	
	def acc(self,
				database_key : str = None,
				opt : str = None,
				divides : list = None,
				same_pids_across_groups : bool = False,
				save : bool = False,
				min_pids : int = 1,
				top_not_mean=False) -> dict:
		"""Returns the accuracy of the test
		
		Args:
			database_key (str): If set, returns the name of the variable to group by. For example, "Sex" would return the accuracy of male and female classification separately (default None)
			opt (str): Type of group to classify by. Can be opt "age_dem" (with and without age encoding and demographic inputs), "name_num" (Number of images input),"name_mod" (name of each group),"name_num_group" (number of modalities input into one classification),"diff_date" (span of dates between earliest and latest image)
			divides (list[int]): For opts that involve counting, this indicates the ranges to count
			same_pids_across_groups (bool): For multigroup accuracy, curates a set of PIDs that are present in each group.
			save (bool): Saves the results in a directory
		
		Returns:
			dictionary of groups, AUROC for that group, number of patients and images in that group
		"""
		
		group_stat = self.get_group_pred_arrs(
						database_key = database_key,
						opt = opt,
						divides = divides,
						same_pids_across_groups = same_pids_across_groups,
						min_pids=min_pids,
						top_not_mean=top_not_mean)
		group_accs = {}
		for group in group_stat:
			Ys,y_preds,image_count,patient_count,m_confidence = group_stat[group]
			acc = np.mean(np.argmax(Ys,axis=1) == np.argmax(y_preds,axis=1))
			
			group_accs[group] = {"acc":acc,
								"images":image_count,
								"patients":patient_count,
								"confidence":m_confidence}
		assert(group_accs is not None)
		return group_accs
	
	def get_group_auc(self,group,top_not_mean=False,mv_limit=0.5):
		assert(group in self.group_set)
		Ys= []
		y_preds = []
		n_images_total = 0
		confs = []
		for pid in self.group_pid_sets[group]:
			pid_record = self.pid_records[pid]
			val = pid_record.get_mean_group(group,top_not_mean=top_not_mean,
				mv_limit=mv_limit)
			if val == -1: continue
			y_,yp_,n_images,m_conf_ = val
			Ys.append(y_)
			y_preds.append(yp_)
			confs.append(m_conf_)
			n_images_total += n_images
		Ys = np.array(Ys)
		confs = np.array(confs)
		y_preds = np.array(y_preds)
		if len(Ys.shape) < 2: return -1
		if len(Ys.shape) < 2:
			Ys = np.expand_dims(Ys,axis=0)
			y_preds = np.expand_dims(y_preds,axis=0)
		#for i in range(Ys.shape[1]):
		Ys = Ys[:,self.include_inds]
		y_preds = y_preds[:,self.include_inds]
		
		if self.verbose:
			print("---")
			print("Ys.shape: %s" %  str(Ys.shape))
			print("y_preds.shape: %s" %  str(y_preds.shape))
		selection = np.any(Ys == 1,axis=1)
		if self.verbose:
			Ys = Ys[selection,:]
			y_preds = y_preds[selection,:]
			print("Ys.shape: %s" %  str(Ys.shape))
			print("y_preds.shape: %s" %  str(y_preds.shape))

		tot_n_patients = Ys.shape[0]
		if self.verbose:
			print("tot_n_patients: %d" % tot_n_patients)
		if tot_n_patients == 0: return -1
		mean_auc = 0
		c = 0
		if len(Ys.shape) < 2: return -1
		for i in range(Ys.shape[1]):# self.include_inds:# range(Ys.shape[1]):
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				fpr, tpr, thresholds = roc_curve(Ys[:,i], y_preds[:,i])
				auc_ = auc(fpr,tpr)
				if self.verbose:
					print("{i:d}: {auc:.4f}".format(i=i,auc=auc_))
				mean_auc += auc_
				c += 1
		if c == 0: return -1
		mean_auc = mean_auc / c
		return mean_auc,tot_n_patients,n_images_total,np.mean(Ys[:,0]),np.mean(confs)
	def get_group_acc(self,group,top_not_mean=False,mv_limit=0.5):
		assert(group in self.group_set)
		Ys= []
		y_preds = []
		n_images_total = 0
		for pid in self.group_pid_sets[group]:
			pid_record = self.pid_records[pid]
			val = pid_record.get_mean_group(group,top_not_mean=top_not_mean,
				mv_limit=mv_limit)
			if val == -1:
				#print("Skipping %s" % pid)
				continue
			y_,yp_,n_images = val
			Ys.append(y_)
			y_preds.append(yp_)
			n_images_total += n_images
		Ys = np.array(Ys)
		y_preds = np.array(y_preds)
		mean_acc = 0
		c = 0
		if len(Ys.shape) < 2: return -1
		if len(Ys.shape) < 2:
			Ys = np.expand_dims(Ys,axis=0)
			y_preds = np.expand_dims(y_preds,axis=0)
		#for i in range(Ys.shape[1]):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			Ys = Ys[:,self.include_inds]
			y_preds = y_preds[:,self.include_inds]
			selection = np.any(Ys == 1,axis=1)
			Ys = Ys[selection,:]
			y_preds = y_preds[selection,:]
			tot_n_patients = Ys.shape[0]
			am_Ys = np.argmax(Ys,axis=1)
			am_y_preds = np.argmax(y_preds,axis=1)
			equals = am_Ys == am_y_preds
			mean_acc += np.mean(equals)
		return mean_acc,tot_n_patients,n_images_total,np.mean(Ys[:,0])

	def greatest_group_difference(self):
		m = -1
		mprec = None
		tuples = []
		for pid in self.pid_records:
			prec = self.pid_records[pid]
			cur_m = prec.get_group_difference()
			tuples.append((cur_m,prec))
			if cur_m > m or m == -1:
				mprec = prec
				m = cur_m
		shuffle(tuples)
		tuples = sorted(tuples,key=lambda k: k[0],reverse=True)
		for m,t in tuples:
			for f in t.file_records:
				if t.file_records[f][0].Y[1] == 3:
					return t
		return t
		#t = tuples[int(len(tuples) * 0.86)]
		#print(t)
		#return t[1]

