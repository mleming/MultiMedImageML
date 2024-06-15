import torch,os
from torch import nn
from pathlib import Path
import numpy as np
import pandas as pd
import json
from .Records import BatchRecord
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#from pytorch_grad_cam import GradCAMPlusPlus,GradCAM
from sklearn.metrics import auc,roc_curve
import glob
import warnings
from .utils import resize_np,is_nan

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
		database_key : str = "ProtocolNameSimplified",
		min_pids : int = 1,
		top_not_mean : bool = False,
		include_inds : list = [0,1],
		same_patients : bool = False,
		x_axis_opts : str = "images"):
		
		self.name=name
		self.checkpoint_dir=checkpoint_dir
		self.model = model
		self.model_file = os.path.join(
			self.checkpoint_dir,'%s.pt' % self.name)
		if os.path.isfile(self.model_file):
			state_dicts = torch.load(self.model_file)
			self.model.load_state_dict(state_dicts['model_state_dict'])
		self.model.eval()
		self.out_record_folder = out_record_folder
		os.makedirs(self.out_record_folder,exist_ok=True)
		self.name = name
		self.test_name = test_name
		if self.out_record_folder is not None and self.name is not None:
			self.stats_record = _StatsRecord(
				os.path.join(self.out_record_folder,"json"),
				self.name,
				test_name=self.test_name)
		self.pid_records = None
		self.use_auc=True
		self.remove_inds = []
		self.include_cbar = True
		self.mv_limit = 0.5
		self.x_axis_opts = x_axis_opts # images, patients, or images_per_patient
		self.x_file_pid = False
		self.database = database
		self.min_pids = min_pids
		self.top_not_mean = top_not_mean
		if not isinstance(self.database,pd.DataFrame):
			self.database = self.database.database
		self.database_key = database_key
		self.include_inds = include_inds
		self.same_patients = same_patients
		self.verbose = verbose
	def plot(self,
			ind=0,
			database_key = None,
			opt = None,
			divides = None,
			same_pids_across_groups = False,
			x_axis_opts = "images",
			acc_or_auc = "auc"
		):
		"""Plots results to a graph. Allows one to analyze individual groups.
		
		Args:
			database_key (str): Variable to group by. Must not be continuous.
			opt (str): "age_dem", "name_num","name_mod","name_num_group","diff_date"
			divides (list): List of numbers by which certain options, like name_num, may be grouped.
			same_pids_across_groups (bool): When analyzing across different groups, allows one to make sure all groups have the same patients (default False)
			x_axis_opts (str): What to plot on the x axis (may be "images", "patients", or "images per patient")
			
		"""
		os.makedirs(os.path.join(self.out_record_folder,"plots"),exist_ok=True)
		self.pid_records.plot(
			database_key = database_key,
			ind = ind,
			x_axis_opts = x_axis_opts,
			acc_or_auc = acc_or_auc,
			opt = opt,
			divides = divides,
			same_pids_across_groups = same_pids_across_groups
			)
	def auc(self,ind = 0,
						database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False):
		return self.pid_records.auc(ind=ind,
					database_key=database_key,
					opt=opt,
					divides=divides,
					same_pids_across_groups=same_pids_across_groups)
	def acc(self,database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False):
		return self.pid_records.acc(database_key=database_key,
					opt=opt,
					divides=divides,
					same_pids_across_groups=same_pids_across_groups)
	def plot(self,
			ind = 0,
			x_axis_opts = "images",
			acc_or_auc = "auc",
			database_key = None,
			opt = None,
			divides = None,
			same_pids_across_groups = False):
		if acc_or_auc == "acc":
			group_set = self.acc(
						database_key = database_key,
						opt = opt,
						divides = divides,
						same_pids_across_groups = same_pids_across_groups
					)
		elif acc_or_auc == "auc":
			group_set = self.auc(ind = ind,
						database_key = database_key,
						opt = opt,
						divides = divides,
						same_pids_across_groups = same_pids_across_groups
					)
		else:
			raise Exception("Invalid opt: %s" % acc_or_auc)
		plt.clf()
		for group in group_set:
			x = group_set[group][acc_or_auc]
			if is_nan(x): continue
			if x_axis_opts == "images_per_patient":
				if group_set[group]["patients"] == 0:
					y = 0
				else:
					y = group_set[group]["images"]/group_set[group]["patients"]
			else:
				y = group_set[group][x_axis_opts]
			plt.scatter(x,y,label=group)
			plt.text(x,y,group[:10])
		out_plot_folder = os.path.join(self.out_record_folder,"plots")
		os.makedirs(out_plot_folder,exist_ok=True)
		
		out_plot_file = os.path.join(
			out_plot_folder,
			f"{database_key}_{x_axis_opts}_{acc_or_auc}_{opt}.png")
		plt.savefig(out_plot_file)
		return
		X = []
		Y = []
		L = []
		C = []
		num_skips = 0
		for l in self.group_set:
			if self.use_auc:
				val = self.get_group_auc(l,mv_limit=self.mv_limit)
			else:
				val = self.get_group_acc(l,mv_limit=self.mv_limit)
			if val == -1:
				#print("Skipping %s" % l)
				continue
			x,n_patients,n_images,portion = val
			if n_patients < self.min_pids and self.min_pids > -1:
				num_skips += 1
				continue
			L.append(l)
			X.append(x)
			C.append(portion)
			if self.x_axis_opts == "images":
				Y.append(n_images)
				xlabel = "Num Images" + (" (%d total patients)" % n_patients) \
					if self.same_patients else ""
			elif self.x_axis_opts == "patients":
				Y.append(n_patients)
				xlabel = "Num Patients"
			elif self.x_axis_opts == "images_per_patient":
				Y.append(float(n_images)/n_patients)
				xlabel = "Mean images per patient"
			else:
				raise Exception("Nothing")
		if num_skips == len(self.group_set):
			raise Exception(
				"Not enough patients in any one group with %d minimum PIDs" % \
					self.min_pids)
		for x,y,l,c in zip(X,Y,L,C):
			if np.isnan(x): continue
			if self.verbose:
				print("{l:s}: {x:.4f} {y:.4f}".format(c=c,y=y,l=l,x=x))
			#plt.scatter(y,x,c=c,cmap=plt.cm.viridis)
			plt.text(y,x,l[:10])#.replace("_","\n"),rotation=0)
		if self.include_cbar:
			sc = plt.scatter(np.array(Y),np.array(X),c=np.array(C),
				cmap=plt.cm.viridis,vmin=0,vmax=1)
			plt.colorbar(sc,label=cbar_label)
		else:
			plt.scatter(np.array(Y),np.array(X))
		plt.title(title)
		if self.use_auc:
			plt.ylabel("AUC")
		else:
			plt.ylabel("Accuracy")
		try:
			plt.xlabel(self.xlabel)
		except:
			pass
			#print("No data in %s" % self.name)
			#exit()
		if output is None:
			plt.show()
		else:
			plt.savefig(output)
	
	def loop(self,pr: BatchRecord):
		"""Tests one input and saves it.
		
		Args:
			pr (BatchRecord) : Image batch
		
		"""
		y_pred = self.model(pr,
			return_regress = True
			)
		if isinstance(y_pred,tuple):
			y_pred,c_pred = y_pred
		else:
			c_pred = torch.Tensor(np.zeros(y_pred.shape))
		if pr.batch_by_pid:
			self.stats_record.update(
				pr.get_Y(),
				y_pred,
				pr.get_C(),
				c_pred,
				pr.pid,
				pr.get_X_files(),
				age_encode = self.model.encode_age,
				static_inputs = [] if self.model.static_dropout \
					else pr.get_static_inputs()
			)
		else:
			for i,im in enumerate(pr.image_records):
				self.stats_record.update(
					im.get_Y(),
					y_pred[i,...],
					im.get_C(),
					c_pred[i,...],
					im.get_ID(),
					im.npy_file,
					age_encode = self.model.encode_age,
					static_inputs = [] if self.model.static_dropout \
					else pr.get_static_inputs()
				)
		return
	def read_json(self):
		"""Reads all json files output by MultiInputTester."""
		
		if self.pid_records is None:
			self.pid_records = _AllRecords(self.database,
				remove_inds = self.remove_inds,
				database_key = self.database_key,
				use_auc = self.use_auc,
				min_pids = self.min_pids,
				mv_limit = self.mv_limit,
				top_not_mean = self.top_not_mean,
				include_inds = self.include_inds,
				x_axis_opts = self.x_axis_opts,
				same_patients = self.same_patients,
				name = self.name,
				verbose = self.verbose)
		json_files = glob.glob(os.path.join(self.out_record_folder,"json",
			"*.json"))
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
					self.pid_records.add_record(pid,
						xf,yf,ypf,self._json_title_parse(json_file),
						age_encode=age_encode,static_inputs=static_inputs)
		if self.same_patients:
			self.pid_records.merge_group_pids()
	def _json_title_parse(self,json_file):
		return "_".join(
				os.path.basename(json_file).replace('.json','').split("_")[:-1]
			)
	def grad_cam(self,pr: BatchRecord,
		add_symlink: bool = True,
		grad_layer: int = 7) -> torch.Tensor:
		"""Outputs a gradient class activation map for the input record
		
		Args:
			pr (BatchRecord): Image batch to apply Grad-Cam to
			add_symlink (bool): If true, adds a symbolic link to the original image in the same folder as the grad-cam is stored in (default True)
			grad_layer (int):  (default 7)
		"""
		self.model.grad_layer = grad_layer
		if pr.image_records[0].Y_dim[0] > 1:
			raise Exception(
				("Grad Cam cannot be applied to" + \
				" multilabel models (Y_dim: %s)") % str(pr.Y_dim))
		
		Y = pr.get_Y()
		y_pred = self.model(pr.get_X(),grad_eval=True)
		
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
			npsqueeze = resize_np(npsqueeze,im.X_dim)
			out_folder = os.path.join(self.out_record_folder,
							"grads",im.group_by)
			os.makedirs(out_folder,exist_ok=True)
			bname = os.path.splitext(os.path.basename(im.npy_file))[0]
			out_name = f"{bname}_grad.npy"
			orig_name = f"{bname}_orig.npy"
			np.save(os.path.join(out_folder,out_name),npsqueeze)
			if add_symlink and not os.path.isfile(os.path.join(out_folder,orig_name)):
				os.symlink(im.npy_file,os.path.join(out_folder,orig_name))
		return t
		
class _StatsRecord():
	def __init__(self,out_record_folder,name,test_name=""):
		self.test_name=test_name
		self.out_record_folder = out_record_folder
		os.makedirs(out_record_folder,exist_ok=True)
		self.name = self.get_name(self.out_record_folder,name,self.test_name)
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
	def update(self,Y,y_pred,C,c_pred,ID,X_files,age_encode=False,static_inputs=[]):
		if torch.is_tensor(Y): Y = Y.detach().numpy()
		if torch.is_tensor(y_pred): y_pred = y_pred.detach().numpy()
		if torch.is_tensor(C): C = C.detach().numpy()
		if torch.is_tensor(c_pred): c_pred = c_pred.detach().numpy()
		
		self.x_files_read = self.x_files_read.union(set(X_files))
		self.pids_read.add(ID)

		if len(Y.shape) == 2:
			Y = np.expand_dims(Y,axis=1)
		if len(y_pred.shape) == 2:
			y_pred = np.expand_dims(y_pred,axis=1)
		assert(len(Y.shape) == 3)
		assert(len(y_pred.shape) == 3)
		if self.all_Y is None: self.all_Y = Y
		else: self.all_Y = np.concatenate((self.all_Y,Y),axis=0)
		if self.all_y_pred is None: self.all_y_pred = y_pred
		else: self.all_y_pred = np.concatenate((self.all_y_pred,y_pred),axis=0)
		if self.all_C is None: self.all_C = C
		else: self.all_C = np.concatenate((self.all_C,C),axis=0)
		if self.all_c_pred is None: self.all_c_pred = c_pred
		else:
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
	def record(self):
		self.out_record_file = os.path.join(
			self.out_record_folder,"%s.json" % self.name)
		self.out_record_file_txt=os.path.join(
			self.out_record_folder,"%s.txt" % self.name)
		self.out_conf_record_file = os.path.join(
			self.out_record_folder,"%s_conf.json" % self.name)
		self.out_conf_record_file_txt = os.path.join(
			self.out_record_folder,"%s_conf.txt" % self.name)
		if self.all_Y is None:
			print("Nothing to record")
			return
		self.output_auc()
		with open(self.out_record_file,'w') as fileobj:
			json.dump(self.out_record,fileobj,indent=4)
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
				remove_inds=[]):
		self.X_files = sorted(X_files)
		self.Y = Y
		self.y_pred = y_pred
		self.database=database
		self.pid=pid + str(np.argmax(Y))
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
		
		if opt is None:
			return "all"
		elif opt == "name_num":
			return self.get_filetypes_name_num(divides = divides)
		elif opt == "diff_date":
			return self.get_filetypes_diff_date(divides = divides)
		elif opt == "name_mod":
			assert(database_key is not None)
			return self.get_filetypes_name_group(database_key=database_key,
							divides =divides) #self.group
		elif opt == "name_num_group":
			assert(database_key is not None)
			return self.get_filetypes_name_num_group(
							database_key=database_key) #self.group
		elif opt == "age_dem":
			foo = ("Age" if age_encode else "")
			foo = foo + " "
			foo = foo + ("Demo" if len(static_inputs) > 0 else "")
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
			self.dates = [self.database.loc[X_file,'ExamEndDTS'] \
				for X_file in self.X_files]
			self.dates = list(filter(lambda k: k is not None,self.dates))
			self.dates = [_.split(":")[0].replace("_","-") for _ in self.dates]
			self.dates = [dateutil.parser.parse(_) for _ in self.dates]
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
		
	def get_filetypes_name_num_group(self,database_key):
		"""Returns the number of different modalities in a given set of images
		"""
		
		ftypes = [self.database.loc[_,database_key] \
			for _ in self.X_files]
		for i,f in enumerate(ftypes):
			if f is None: ftypes[i] = "None"
			ftypes[i] = ftypes[i].strip().upper()
		ftypes = set(ftypes)
		return str(len(ftypes))
		
	def get_filetypes_name_group(self,database_key):
		"""Returns a list of unique modalities in a given set of images
		"""
		
		ftypes = [self.database.loc[_,database_key] \
			for _ in self.X_files]
		for i,f in enumerate(ftypes):
			if f is None: ftypes[i] = "None"
			ftypes[i] = ftypes[i].strip().upper()
		return  "_".join(sorted(list(set(ftypes))))
		
	def get_filetypes_name_group_num(self,database_key):
		"""Returns a list of unique modalities in a given set of images
		"""

		ftypes = [self.database.loc[_,database_key] \
			for _ in self.X_files]
		for i,f in enumerate(ftypes):
			if f is None: ftypes[i] = "None"
			ftypes[i] = ftypes[i].strip().upper()
		ftypes = set(ftypes)
		return  "_".join(sorted(list(ftypes)))

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
			remove_inds=[],
			top_not_mean=False):
		self.remove_inds=remove_inds
		self.pid = pid
		self.file_records = []
		self.database = database
		self.top_not_mean=top_not_mean
	def add_file_record(self,
			X_files,
			Y,
			y_pred,
			age_encode=False,
			static_inputs=[]):
		f = _FileRecord(
				X_files,
				Y,
				y_pred,
				self.pid,
				self.database,
				age_encode = age_encode,
				static_inputs = static_inputs,
				remove_inds = self.remove_inds)
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
		
		group_dict = self.get_group_dict(database_key = None,
										opt = None,
										divides = None)
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
			top_not_mean=False,
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
		self.top_not_mean = top_not_mean
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
			static_inputs=[]):

		if pid not in self.pid_records:
			self.pid_records[pid] = _PIDRecord(pid,
					self.database,
					remove_inds=self.remove_inds)
		
		self.pid_records[pid].add_file_record(X_files,
						Y,
						y_pred,
						age_encode=age_encode,
						static_inputs=static_inputs)
	
	# Makes it so that this only outputs records of the same set of patients for
	# all considered modalities
	def get_group_dict(self,
						database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False,
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
			for group in group_pids:
				if intersect_pids is None: intersect_pids = group_pids[group]
				else: intersect_pids = intersect_pids.intersect(group_pids[group])
			group_dicts_same_pid = {}
			for group in group_dicts:
				group_dicts_same_pid[group] = []
				for file in group_dicts[group]:
					if file.pid in intersect_pids:
						group_dicts_same_pid[group].append(file)
			group_dicts = group_dicts_same_pid
		return group_dicts
	
	def get_group_pred_arrs(self,database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False,
						):
		"""Returns set of prediction arrays with the 
		
		"""
		group_dict = self.get_group_dict(
							database_key = database_key,
							divides=divides,
							opt=opt,
							same_pids_across_groups=same_pids_across_groups)
		group_stat = {}
		for group in group_dict:
			group_stat[group] = {}
			for file in group_dict[group]:
				if file.pid not in group_stat[group]:
					group_stat[group][file.pid] = (file.Y,file.y_pred,1)
				else:
					Y_all,y_pred_all,image_count = group_stat[group][file.pid]
					group_stat[group][file.pid] = (file.Y+Y_all,
											file.y_pred+y_pred_all,
											image_count+1)
		# Mean by patient ID
		for group in group_stat:
			Y_all_group = []
			y_all_pred_group = []
			patient_count = 0
			image_count = 0
			for pid in group_stat[group]:
				Y_all,y_pred_all,imc = group_stat[group][pid]
				Y_all_group.append(np.array(Y_all)/imc)
				y_all_pred_group.append(np.array(y_pred_all)/imc)
				patient_count += 1
				image_count += imc
			Y_all_group = np.array(Y_all_group)
			y_all_pred_group = np.array(y_all_pred_group)
			group_stat[group] = (Y_all_group,
								y_all_pred_group,
								image_count,
								patient_count)
		
		return group_stat
	
	def auc(self,ind = 0,
						database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False):
		group_stat = self.get_group_pred_arrs(
						database_key = database_key,
						opt=opt,divides=divides,
						same_pids_across_groups=same_pids_across_groups)
		group_aucs = {}
		for group in group_stat:
			Ys,y_preds,image_count,patient_count = group_stat[group]
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				fpr, tpr, thresholds = roc_curve(Ys[:,ind], y_preds[:,ind])
				auc_ = auc(fpr,tpr)
				group_aucs[group] = {"auc": auc_,
									"images":image_count,
									"patients":patient_count}
		return group_aucs
	
	def acc(self,database_key = None,
						opt = None,
						divides = None,
						same_pids_across_groups = False):
		group_stat = self.get_group_pred_arrs(
						database_key = database_key,
						opt=opt,divides=divides,
						same_pids_across_groups=same_pids_across_groups)
		group_accs = {}
		for group in group_stat:
			Ys,y_preds,image_count,patient_count = group_stat[group]
			acc = np.mean(np.argmax(Ys,axis=1) == np.argmax(y_preds,axis=1))
			
			group_accs[group] = {"acc":acc,
								"images":image_count,
								"patients":patient_count}
		return group_accs
	
	def get_group_auc(self,group,mv_limit=0.5):
		assert(group in self.group_set)
		Ys= []
		y_preds = []
		n_images_total = 0
		for pid in self.group_pid_sets[group]:
			pid_record = self.pid_records[pid]
			val = pid_record.get_mean_group(group,mv_limit=mv_limit)
			if val == -1: continue
			y_,yp_,n_images = val
			Ys.append(y_)
			y_preds.append(yp_)
			n_images_total += n_images
		Ys = np.array(Ys)
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
		return mean_auc,tot_n_patients,n_images_total,np.mean(Ys[:,0])
	def get_group_acc(self,group,mv_limit=0.5):
		assert(group in self.group_set)
		Ys= []
		y_preds = []
		n_images_total = 0
		for pid in self.group_pid_sets[group]:
			pid_record = self.pid_records[pid]
			val = pid_record.get_mean_group(group,mv_limit=mv_limit)
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


