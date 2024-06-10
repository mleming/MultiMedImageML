from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	"""This class includes training options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# visdom and HTML visualization parameters
		parser.add_argument('--pretrained_model',type=str,default=None, help='Pretrained model to load in')
		#parser.add_argument('--val_ranges',default={'DiffDem':['None','G30','F01'],'InstitutionNameSimplified':'MGH'},help="What stuff to load into the training set")
		parser.add_argument('--val_ranges',default={'InstitutionNameSimplified_Date':['MGH_BEFORE_2019','MGH_AFTER_2019'],'ICD_one_G35':['ICD_one_G35','NOT_ICD_one_G35']})
		#parser.add_argument('--val_ranges',default={'InstitutionNameSimplified_Date':['MGH_BEFORE_2019','MGH_AFTER_2019'],'AlzStage':['AD','CONTROL']})
		# network saving and loading parameters
		parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
		parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
		parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
		parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
		# training parameters
		parser.add_argument('--epochs', type=int, default=1000, help='number of epochs with the initial learning rate')
		#parser.add_argument('--epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
		parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
		parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate for adam')
		#parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
		#parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
		#parser.add_argument('--max_iters',type=int,default=-1)
		parser.add_argument('--use_mix',action='store_true',default=False,help='Uses an alternate loss function for the reconstruction')
		parser.add_argument('--augment',action='store_true',default=True,help='Augments 3D images')
		#parser.add_argument('--total_load',type=int,default=30*45000,help='Total number of data to load into main memory')
		parser.add_argument('--no_regress',action='store_true',default=False)
		self.isTrain = True
		return parser
