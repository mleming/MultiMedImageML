import torch,os
from torch import nn
import numpy as np
from .Records import BatchRecord
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pytorch_grad_cam import GradCAMPlusPlus,GradCAM 

# Tests either the model directly or the output files
class MultiInputTester():
	def __init__(self,model=None,name=None,out_record_folder=None):
		self.model = model
		self.model.eval()
		self.out_record_folder = out_record_folder
		self.name = name
		if out_record_folder is not None and name is not None:
			self.stats_record = StatsRecord(self.out_record_folder,self.name)
		self.pid_records = None
	def test(self,pr: BatchRecord):
		return
	def read_json(self):
		if self.pid_records is None: self.pid_records = AllRecords()
		for json_file in json_files:
			try:
				json_dict = json.load(open(json_file,'r'))
			except:
				print("Error in opening %s" % json_file)
				continue
			X_files = []
			Ys = []
			y_preds = []
			for pid in json_dict:
				for mm in range(len(json_dict[pid])):
					xf = np.array(json_dict[pid][mm]["X_files"]).flatten()
					yf = np.array(json_dict[pid][mm]["Y"])#.flatten()
					for i in remove_inds:
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
					pid_records.add_record(pid,xf,yf,ypf,title_parse(json_file),
						age_encode=age_encode,static_inputs=static_inputs)

		
	def test_grad_cam(self,pr: BatchRecord):
		target_layers = []
		for name, module in self.model.named_modules():
			if not name.startswith('params'):
				if name == 'encoder.encoder.1':# or name == 'encoder.encoder.1':
					target_layers.append(module)
		self.model.regressor_freeze()
		G = GradCAMPlusPlus(model=self.model,target_layers=target_layers)
		t = G(input_tensor = pr.get_X(), targets = None)
		for i in range(t.shape[0]):
			nn = np.squeeze(t[i,...])
			out_name = os.path.join(
				self.output_folder,"%s_%s.npy" % (self.model,
					os.path.splitext(os.path.basename(args.images[i]))[0]))
			ind = args.images[i]
			np.save(out_name,nn)
			mod = database.loc[ind,"ProtocolNameSimplified"]
			mod = "None" if mod is None else mod
			title = "{mod:s}".format(mod = mod)
			command_str = "image_slices_viewer.py {image_name:s} {out_name:s} --title={title:s} --rg &".format(title=title,out_name=out_name,image_name=args.images[i])
			#print(command_str)

		
class StatsRecord():
	def __init__(self,out_record_folder,name,test_name=""):
		self.test_name=test_name
		self.name = self.get_name(out_record_folder,name,self.test_name)
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
			#return  os.path.join(out_record_folder,
			#			"%s_%s.json" % (name,self.test_name))
			n,ext = os.path.splitext(name)
			name = n + "_" + self.test_name + ext
		num = 0
		while os.path.isfile(
				os.path.join(out_record_folder,"%s_%d.json" % (name,num))
			):
			num += 1
		Path(os.path.join(out_record_folder,"%s_%d.json" % (name,num))).touch()
		return "%s_%d" % (name,num)
	def update(self,Y,y_pred,C,c_pred,IDs,X_files,age_encode=False,static_inputs=[]):
		self.x_files_read = self.x_files_read.union(set(X_files))
		self.pids_read = self.pids_read.union(set(IDs))

		#if len(Y.shape) < 2:
		#	Y = np.expand_dims(Y,axis=0)
		if len(Y.shape) == 2:
			Y = np.expand_dims(Y,axis=1)
		#if len(y_pred.shape) < 2:
		#	y_pred = np.expand_dims(y_pred,axis=0)
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
		#acc = get_multilabel_acc(y_pred,pr.Y)
		if self.all_IDs is None: all_IDs = IDs
		else: self.all_IDs = np.concatenate((self.all_IDs,IDs),axis=0)
		if IDs[0] not in self.out_record:
			self.out_record[IDs[0]] = []
		
		#print(Y)
		#print(Y.shape)
		#print(y_pred)
		#print(y_pred.shape)
		if len(Y.shape) == 2:
			self.out_record[IDs[0]].append({
							'X_files' : [str(_) for _ in X_files],
							'Y' : [float(_) for _ in list(Y[0])],
							'y_pred' : [float(_) for _ in list(y_pred[0])],
							'age_encode' : age_encode,
							'static_inputs' : [str(_) for _ in list(static_inputs)]
							})
		else:
			self.out_record[IDs[0]].append({
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
					fpr, tpr, thresholds = roc_curve(self.all_Y[:,j,k], self.all_y_pred[:,j,k])
					cc.append(auc(fpr,tpr))
				self.all_auroc.append(cc)
			else:
				print("Mismatched shape")
				print("all_Y.shape: %s"%str(all_Y.shape))
				print("all_y_pred.shape: %s"%str(all_y_pred.shape))
				exit()
		for j in range(min(self.all_C.shape[1],self.all_c_pred.shape[1])):
			cc = []
			for k in range(min(self.all_C.shape[2],self.all_c_pred.shape[2])):
				fpr, tpr, thresholds = roc_curve(self.all_C[:,j,k], self.all_c_pred[:,j,k])
				cc.append(auc(fpr,tpr))
			self.all_c_auroc.append(cc)

	def record(self):
		self.out_record_file = os.path.join(out_record_folder,"%s.json" % self.name)
		self.out_record_file_txt=os.path.join(out_record_folder,'%s.txt' % self.name)
		self.out_conf_record_file = os.path.join(out_record_folder,"%s_conf.json" % self.name)
		self.out_conf_record_file_txt = os.path.join(out_record_folder,"%s_conf.txt" % self.name)
		if self.all_Y is None:
			print("Nothing to record")
			return
		self.output_auc()
		json.dump(self.out_record,open(self.out_record_file,'w'),indent=4)
		with open(self.out_record_file_txt,'w') as fileobj:
			fileobj.write( "%d: %s - # files: %d; num patients: %d; %s; %s" % \
				(i,self.name,len(self.x_files_read),len(self.pids_read),
					str(self.all_auroc),str(self.all_c_auroc)))

		#json.dump(self.out_conf_record,open(self.out_conf_record_file,'w'),
		#	indent=4)
		with open(self.out_conf_record_file_txt,'w') as fileobj:
			for a in self.all_c_auroc:
				fileobj.write(str(a)+"\n")


class FileRecord:
	def __init__(self,X_files,Y,y_pred,modality="",
			age_encode=False,static_inputs=[]):
		self.X_files = sorted(X_files)
		self.Y = Y
		self.y_pred = y_pred
		for i in remove_inds:
			self.y_pred[i] = 0
		self.age_encode = age_encode
		self.static_inputs = static_inputs
		if modality != "":
			self.modality = modality
		# ProtocolNameSimplified, MRModality, Modality
		if args.c == "name_num":
			self.filetypes_name = self.get_filetypes_name_num()
		elif args.c == "diff_date":
			self.filetypes_name = self.get_filetypes_diff_date()
		elif args.c == "name_mod":
			self.filetypes_name = self.get_filetypes_name_modality() #self.modality
		elif args.c == "age_dem":
			self.filetypes_name = ("Age" if age_encode else "") + " " +("Demo" if len(static_inputs) > 0 else "")
			if self.filetypes_name == " ": self.filetypes_name = "Only images"
		else:
			raise Exception("Invalid argument for args.c: %s" % args.c)
		#self.filetypes_name = " "
	def get_filetypes_diff_date(self):
		if len(self.X_files) == 1:
			return None # "One Image"
		self.dates = [database.loc[X_file,'ExamEndDTS'] for X_file in self.X_files]
		self.dates = list(filter(lambda k: k is not None,self.dates))
		self.dates = [_.split(":")[0].replace("_","-") for _ in self.dates]
		self.dates = [dateutil.parser.parse(_) for _ in self.dates]
		date_diff = max(self.dates) - min(self.dates)
		date_diff = date_diff.days / 365.25
		divides = [5]
		ret = self.get_divides(date_diff,divides)
		ret = ret + (" encoded" if self.age_encode else "")
		return ret
	
	def get_filetypes_name_num(self):
		divides = [1,5,10,14]
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

	def get_filetypes_name_num_modality(self):
		self.filetypes = [database.loc[_,database_key] \
			for _ in self.X_files]
		for i,f in enumerate(self.filetypes):
			if f is None: self.filetypes[i] = "None"
			self.filetypes[i] = self.filetypes[i].strip().upper()
		self.filetypes = set(self.filetypes)
		return  str(len(self.filetypes))
	def get_filetypes_name_modality(self):
		self.filetypes = [database.loc[_,database_key] \
			for _ in self.X_files]
		for i,f in enumerate(self.filetypes):
			if f is None: self.filetypes[i] = "None"
			self.filetypes[i] = self.filetypes[i].strip().upper()
		self.filetypes = set(self.filetypes)
		return  "_".join(sorted(list(self.filetypes)))
	def get_filetypes_name_modality_num(self):
		self.filetypes = [database.loc[_,database_key] \
			for _ in self.X_files]
		for i,f in enumerate(self.filetypes):
			if f is None: self.filetypes[i] = "None"
			self.filetypes[i] = self.filetypes[i].strip().upper()
		self.filetypes = set(self.filetypes)
		if len(self.filetypes) < 3 or True:
			return  "_".join(sorted(list(self.filetypes)))
		else:
			return  str(len(self.filetypes)) + " modalities"
	def get_acc(self):
		return ((np.argmax(self.Y)) == (np.argmax(self.y_pred))).astype(float)
	def print_record(self,indent=0):
		print("-")
		print((indent * " ") + str(self.get_acc()))
		for X_file in self.X_files:
			print((indent*" ") + get_file_str(X_file))
		print("-")

class PIDRecord:
	def __init__(self,pid):
		self.pid = pid
		self.file_records = {}
	def add_file_record(self,X_files,Y,y_pred,modality,
			age_encode=False,static_inputs=[]):
		f = FileRecord(X_files,Y,y_pred,modality,age_encode=age_encode,
			static_inputs=static_inputs)
		if record_key == "filetypes":
			key = f.filetypes_name
		elif record_key == "modality":
			key = f.modality
		else:
			raise Exception("Invalid record key: %s" % key)
		if key is None: return None
		if key not in self.file_records:
			self.file_records[key] = []
		self.file_records[key].append(f)
		return key
	def get_mean_modality(self,modality, mv_limit = 0.0):
		if modality not in self.file_records:
			return -1
		else:
			Ys = []
			yps = []
			X_file_set = set()
			for fr in sorted(self.file_records[modality],
				key=lambda k: len(k.X_files),
				reverse=False):
				Ys.append(fr.Y)
				yps.append(fr.y_pred)
				for X_file in fr.X_files: X_file_set.add(X_file)
				if top_not_mean: break
		Ys = np.array(Ys)
		yps = np.array(yps)
		Ys = np.mean(Ys,axis=0)
		yps = np.mean(yps,axis=0)
		Ys_ = np.zeros(yps.shape)
		Ys_[np.argmax(Ys)] = 1
		Ys = Ys_
		max_val = np.max(yps)
		if max_val < mv_limit:
			#print("SKIPSIES")
			return -1
		if not np.all([(_ == 0 or _ == 1) for _ in Ys]):
			#print("NOPESIES")
			return -1
		return Ys,yps,len(X_file_set)
	def get_mean_accuracy(self,modality):
		if modality not in self.file_records:
			return -1
		else:
			sum_ = 0.0
			c = 0
			for fr in self.file_records[modality]:
				sum_ += fr.get_acc()
				c += 1
			return sum_ / c
	def get_modality_difference(self):
		modvals = {}
		for modality in set(self.file_records):
			val = self.get_mean_accuracy(modality)
			if val == -1: continue
			modvals[modality] = val #np.mean((np.argmax(Y,axis=0) == np.argmax(y_pred,axis=0)).astype(float))
		if len(modvals) == 0: return 0
		return max([modvals[m] for m in modvals]) - \
			min([modvals[m] for m in modvals])
	def print_record(self):
		print("---")
		print(self.pid)
		print(" ")
		for modality in self.file_records:
			print(modality)
			print(self.get_mean_accuracy(modality))
			for file_record in self.file_records[modality]:
				file_record.print_record(indent=4)

class AllRecords:
	def __init__(self):
		self.pid_records = {}
		self.modality_pid_sets = {} # Sets of patients who have a certain modality
		self.modality_set = set()
	def add_record(self,pid,X_files,Y,y_pred,modality,
			age_encode=False,
			static_inputs=[]):
		if x_file_pid:
			pid = pid + ",".join(X_files)
		if pid not in self.pid_records:
			self.pid_records[pid] = PIDRecord(pid)
		
		key = self.pid_records[pid].add_file_record(X_files,Y,y_pred,modality,
			age_encode=age_encode,static_inputs=static_inputs)
		if key is not None:
			self.modality_set.add(key)
			if key not in self.modality_pid_sets:
				self.modality_pid_sets[key] = set()
			self.modality_pid_sets[key].add(pid)
	# Makes it so that this only outputs records of the same set of patients for
	# all considered modalities
	def merge_modality_pids(self,min_pids = -1):
		pid_set = set(self.pid_records)
		if min_pids > len(pid_set):
			raise Exception("Min pids is %d, but total pids is %d" % \
				(min_pids,len(pid_set)))
		exclude_modalities = set()
		lenlist = sorted([(len(self.modality_pid_sets[m]),m)\
			 for m in self.modality_set],reverse=True)
		for l,m in lenlist:
			temp = pid_set.intersection(self.modality_pid_sets[m])
			if len(temp) < min_pids and min_pids > -1:
				exclude_modalities.add(m)
			else:
				pid_set = temp
		self.modality_set = self.modality_set - exclude_modalities
		for modality in self.modality_set:
			self.modality_pid_sets[modality] = pid_set
	def get_modality_auc(self,modality,mv_limit=0.5):
		assert(modality in self.modality_set)
		Ys= []
		y_preds = []
		n_images_total = 0
		for pid in self.modality_pid_sets[modality]:
			pid_record = self.pid_records[pid]
			val = pid_record.get_mean_modality(modality,mv_limit=mv_limit)
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
		Ys = Ys[:,include_inds]
		y_preds = y_preds[:,include_inds]
		print("---")
		print("Ys.shape: %s" %  str(Ys.shape))
		print("y_preds.shape: %s" %  str(y_preds.shape))
		selection = np.any(Ys == 1,axis=1)
		Ys = Ys[selection,:]
		y_preds = y_preds[selection,:]
		print("Ys.shape: %s" %  str(Ys.shape))
		print("y_preds.shape: %s" %  str(y_preds.shape))

		tot_n_patients = Ys.shape[0]
		print("tot_n_patients: %d" % tot_n_patients)
		if tot_n_patients == 0: return -1
		mean_auc = 0
		c = 0
		if len(Ys.shape) < 2: return -1
		for i in range(Ys.shape[1]):# include_inds:# range(Ys.shape[1]):
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				fpr, tpr, thresholds = roc_curve(Ys[:,i], y_preds[:,i])
				auc_ = auc(fpr,tpr)
				print("{i:d}: {auc:.4f}".format(i=i,auc=auc_))
				mean_auc += auc_
				c += 1
		if c == 0: return -1
		mean_auc = mean_auc / c
		return mean_auc,tot_n_patients,n_images_total,np.mean(Ys[:,0])
	def get_modality_acc(self,modality,mv_limit=0.5):
		assert(modality in self.modality_set)
		Ys= []
		y_preds = []
		n_images_total = 0
		for pid in self.modality_pid_sets[modality]:
			pid_record = self.pid_records[pid]
			val = pid_record.get_mean_modality(modality,mv_limit=mv_limit)
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
			print("---")
			print("Ys")
			print(Ys.shape)
			print(y_preds.shape)
			print("Include")
			Ys = Ys[:,include_inds]
			y_preds = y_preds[:,include_inds]
			selection = np.any(Ys == 1,axis=1)
			Ys = Ys[selection,:]
			y_preds = y_preds[selection,:]
			tot_n_patients = Ys.shape[0]
			print(Ys.shape)
			print(y_preds.shape)
			print("Argmax")
			am_Ys = np.argmax(Ys,axis=1)
			am_y_preds = np.argmax(y_preds,axis=1)
			print(am_Ys.shape)
			print(am_y_preds.shape)
			print("Equals")
			equals = am_Ys == am_y_preds
			print(equals.shape)
			print("Acc")
			mean_acc += np.mean(equals)
			print(mean_acc)
		return mean_acc,tot_n_patients,n_images_total,np.mean(Ys[:,0])

	def greatest_modality_difference(self):
		m = -1
		mprec = None
		tuples = []
		for pid in self.pid_records:
			prec = self.pid_records[pid]
			cur_m = prec.get_modality_difference()
			tuples.append((cur_m,prec))
			if cur_m > m or m == -1:
				mprec = prec
				m = cur_m
		shuffle(tuples)
		tuples = sorted(tuples,key=lambda k: k[0],reverse=True)
		for m,t in tuples:
			for f in t.file_records:
				if t.file_records[f][0].Y[1] == 3:
					print("Motorized seesaw")
					return t
		return t
		#t = tuples[int(len(tuples) * 0.86)]
		#print(t)
		#return t[1]
	def plot(self,output=None):
		X = []
		Y = []
		L = []
		C = []
		num_skips = 0
		for l in self.modality_set:
			if use_auc:
				val = self.get_modality_auc(l,mv_limit=mv_limit)
			else:
				val = self.get_modality_acc(l,mv_limit=mv_limit)
			if val == -1:
				#print("Skipping %s" % l)
				continue
			x,n_patients,n_images,portion = val
			if n_patients < min_pids and min_pids > -1:
				num_skips += 1
				continue
			L.append(l)
			X.append(x)
			C.append(portion)
			if x_axis_opts == "images":
				Y.append(n_images)
				xlabel = "Num Images" + (" (%d total patients)" % n_patients) \
					if same_patients else ""
			elif x_axis_opts == "patients":
				Y.append(n_patients)
				xlabel = "Num Patients"
			elif x_axis_opts == "images_per_patient":
				Y.append(float(n_images)/n_patients)
				xlabel = "Mean images per patient"
			else:
				raise Exception("Nothing")
		if num_skips == len(self.modality_set):
			raise Exception(
				"Not enough patients in any one group with %d minimum PIDs" % \
					min_pids)
		for x,y,l,c in zip(X,Y,L,C):
			if np.isnan(x): continue
			print("{l:s}: {x:.4f} {y:.4f}".format(c=c,y=y,l=l,x=x))
			#plt.scatter(y,x,c=c,cmap=plt.cm.viridis)
			plt.text(y,x,l[:10])#.replace("_","\n"),rotation=0)
		if include_cbar:
			sc = plt.scatter(np.array(Y),np.array(X),c=np.array(C),
				cmap=plt.cm.viridis,vmin=0,vmax=1)
			plt.colorbar(sc,label=cbar_label)
		else:
			plt.scatter(np.array(Y),np.array(X))
		plt.title(title)
		if use_auc:
			plt.ylabel("AUC")
		else:
			plt.ylabel("Accuracy")
		try:
			plt.xlabel(xlabel)
		except:
			print("No data in %s" % args.name)
			exit()
		if output is None:
			plt.show()
		else:
			plt.savefig(output)

