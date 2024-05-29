import argparse
import os
from util import util
import torch
import models
import data
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
		self.wd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
		#parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
		parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
		parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		parser.add_argument('--checkpoints_dir', type=str, default=os.path.join(self.wd,'checkpoints'), help='models are saved here')
		parser.add_argument('--label',default=['AlzStage'],nargs='+',help='Which labels to read in')
		parser.add_argument('--y_weight', type=int, default=6, help='Amount of weight to give to label when training it')
		parser.add_argument('--confounds',type=str,default=['SexDSC','Ages_Buckets','Angle','MRModality','Modality','ScanningSequence'],nargs='+',help='Which confounds to read in')
		parser.add_argument('--match_confounds',default=['SexDSC'],nargs='+',help='Which confounds to match')
		# model parameters
		parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
		parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
		parser.add_argument('--all_vars',type=str,default=os.path.join(os.path.dirname(self.wd),'MGH_ML_pipeline','pandas','cache','all_vars.pkl'),
			help='Pandas table with variable names')
		parser.add_argument('--group_by',type=str,default='PatientID',help='Returns data grouped by this variable')
		parser.add_argument('--batch_size',type=int,default=14)
		parser.add_argument('--get_encoded',action='store_true',default=False)
		parser.add_argument('--no_recurrent',action='store_true',default=False,help='Set to remove recurrent links between datasets')
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
		It will save options into a text file / [checkpoints_dir] / opt.txt
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
		expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
		with open(file_name, 'wt') as opt_file:
			opt_file.write(message)
			opt_file.write('\n')

	def parse(self):
		"""Parse our options, create checkpoints directory suffix, and set up gpu device."""
		opt = self.gather_options()
		opt.isTrain = self.isTrain   # train or test
		
		if not os.path.isfile(opt.all_vars):
			print("%s not a file" % opt.all_vars)
			exit()
		assert(os.path.isfile(opt.all_vars))
		opt.all_vars = pd.read_pickle(opt.all_vars)
		
		if opt.exclude_protocol != "":
			t = ['T1_AX','T2_AX','T1_SAG','SWI_AX','T1_COR','T2_AX_FLAIR','T1_SAG_MPRAGE','T1_AX_MPRAGE','DWI_UNKNOWN','T2_COR','T1_SAG_FLAIR','T2_SAG_FLAIR','T2_SAG','T2_UNKNOWN','SWI_UNKNOWN','T2_UNKNOWN_FLAIR','T1_UNKNOWN','T1_AX_FLAIR','T2_COR_FLAIR','DWI_AX','T1_COR_MPRAGE','T1_UNKNOWN_FLAIR','T1_UNKNOWN_MPRAGE','T1_COR_FLAIR','SWI_COR','SWI_SAG','T2_SAG_MPRAGE','DWI_COR','T2_UNKNOWN_MPRAGE','SWI_AX_FLAIR']
			q = ["MR","PT","CT"]
			if opt.exclude_protocol in t:
				t.remove(opt.exclude_protocol)
				opt.val_ranges['ProtocolNameSimplified'] = t
			elif opt.exclude_protocol in q:
				q.remove(opt.exclude_protocol)
				opt.val_ranges['Modality'] = q
			else:
				raise Exception("Invalid exclude protocol: %s"%opt.exclude_protocol)
		if len(opt.include_protocol) > 0:
			opt.val_ranges['ProtocolNameSimplified'] = opt.include_protocol

		
		if "ICD_one_G35" in opt.label:
			opt.val_ranges['ICD_one_G35'] =['NOT_ICD_one_G35','ICD_one_G35']
		if "AlzStage" in opt.label:
			opt.val_ranges['AlzStage'] =['AD','CONTROL']
		if "DiffDem" in opt.label:
			opt.val_ranges["DiffDem"] = ["G30","F01","G31.83","G31.0"]
		
		if opt.remove_alz_exclusion:
			del opt.val_ranges['AlzStage']
		opt.confounds = sorted(list(set(opt.confounds) - set(opt.label)))
		opt.match_confounds = sorted(list(set(opt.match_confounds) - set(opt.label)))
		# process opt.suffix
		#if opt.suffix:
		#	suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
		#	opt.name = opt.name + suffix

		self.print_options(opt)

		# set gpu ids
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)
		if len(opt.gpu_ids) > 0:
			torch.cuda.set_device(opt.gpu_ids[0])

		self.opt = opt
		return self.opt
