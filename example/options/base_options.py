import argparse
import os
import torch
import json
import pandas as pd

class BaseOptions():
	"""This class defines options used during both training and test time.

	It also implements several helper functions such as parsing, printing, and saving the options.
	It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
	"""

	def __init__(self):
		"""Reset the class; indicates the class hasn't been initailized"""
		self.initialized = False

	def initialize(self, parser):
		"""Define the common options that are used in both training and test."""
		# basic parameters
		self.wd = os.path.dirname(os.path.realpath(__file__))
		#parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
		parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
		parser.add_argument('--X_dim',nargs=3,help='Dimensions of images',default=[96,96,96])
		parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(os.path.dirname(self.wd),'checkpoint'), help='models are saved here')
		parser.add_argument('--label',default=['AlzStage'],nargs='+',help='Which labels to read in')
		parser.add_argument('--y_weight', type=int, default=6, help='Amount of weight to give to label when training it')
		parser.add_argument('--confounds',type=str,default=['SexDSC','Ages_Buckets','Angle','MRModality','Modality','ScanningSequence'],nargs='+',help='Which confounds to read in')
		parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
		parser.add_argument('--match_confounds',default=['SexDSC'],nargs='+',help='Which confounds to match')
		# model parameters
		parser.add_argument('--all_vars',type=str,default=os.path.join(os.path.dirname(self.wd),'MGH_ML_pipeline','pandas','cache','all_vars.pkl'),
			help='Pandas table with variable names')
		parser.add_argument('--group_by',type=str,default='PatientID',help='Returns data grouped by this variable')
		parser.add_argument('--batch_size',type=int,default=256)
		parser.add_argument('--get_encoded',action='store_true',default=False)
		parser.add_argument('--use_attn',action='store_true',default=False,help='Use an attention layer')
		parser.add_argument('--encode_age',action='store_true',default=False,help='Encode age with positional encoder')
		parser.add_argument('--exclude_protocol',type=str,default='')
		parser.add_argument('--include_protocol',nargs="+",default=[])
		parser.add_argument('--static_inputs',type=str, nargs='+',
			default=['SexDSC','EthnicGroupDSC'],
			help='What to encode as patient-wide input')
		parser.add_argument('--variational',action='store_true',default=False,help='Trains encoder with variational sampling and KL divergence')
		parser.add_argument('--zero_input',action='store_true',default=False,help='')
		parser.add_argument('--remove_alz_exclusion',action='store_true',default=False,help="Removes AlzStage from val_ranges argument")
		parser.add_argument('--precedence',default=[],nargs='+',help="If included, precedence of label values in multilabel patients")
		self.initialized = True
		return parser

	def gather_options(self):
		"""Initialize our parser with basic options(only once).
		Add additional model-specific and dataset-specific options.
		These options are defined in the <modify_commandline_options> function
		in model and dataset classes.
		"""
		if not self.initialized:  # check if it has been initialized
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)
		# get the basic options
		opt, _ = parser.parse_known_args()

		# modify model-related parser options
		#model_name = opt.model
		#model_option_setter = models.get_option_setter(model_name)
		#parser = model_option_setter(parser, self.isTrain)
		#opt, _ = parser.parse_known_args()  # parse again with new defaults

		# modify dataset-related parser options
		#dataset_name = opt.dataset_mode
		#dataset_option_setter = data.get_option_setter(dataset_name)
		#parser = dataset_option_setter(parser, self.isTrain)

		# save and return the parser
		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		"""Print and save options

		It will print both current options and default values(if different).
		It will save options into a text file / [checkpoint_dir] / opt.txt
		"""
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(opt).items()):
			if isinstance(v,pd.DataFrame):
				continue
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------'
		print(message)

		# save to the disk
		expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
		os.make_dirs(expr_dir,exist_ok=True)
		file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
		with open(file_name, 'wt') as opt_file:
			opt_file.write(message)
			opt_file.write('\n')

	def parse(self):
		"""Parse our options, create checkpoints directory suffix, and set up gpu device."""
		opt = self.gather_options()
		opt.isTrain = self.isTrain   # train or test
		if isinstance(opt.val_ranges,str):
			if os.path.isfile(opt.val_ranges):
				with open(opt.val_ranges,'r') as fileobj:
					opt.val_ranges = json.load(fileobj)
			else:
				opt.val_ranges = json.loads(opt.val_ranges.replace("'",'"'))
		elif not isinstance(opt.val_ranges,dict): opt.val_ranges={}
		if not os.path.isfile(opt.all_vars):
			print("%s not a file" % opt.all_vars)
			exit()
		assert(os.path.isfile(opt.all_vars))
		opt.confounds = sorted(list(set(opt.confounds) - set(opt.label)))
		opt.match_confounds = sorted(list(set(opt.match_confounds) - set(opt.label)))
		# process opt.suffix
		#if opt.suffix:
		#	suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
		#	opt.name = opt.name + suffix

		#self.print_options(opt)

		# set gpu ids
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)
		if len(opt.gpu_ids) > 0:
			if torch.cuda.is_available():
				torch.cuda.set_device(opt.gpu_ids[0])
			else:
				opt.gpu_ids = None
		self.opt = opt
		return self.opt
