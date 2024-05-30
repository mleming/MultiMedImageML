from .base_options import BaseOptions


class TestOptions(BaseOptions):
	"""This class includes test options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)  # define shared options
		parser.add_argument('--val_ranges',default={'InstitutionNameSimplified':['BWH','OTHER']},help="What stuff to load into the training set")
		#parser.add_argument('--val_ranges',default={'AlzStage':['CONTROL','AD'],'InstitutionNameSimplified':['BWH','OTHERS'],'MRModality':['T1'],'Angle':['AX']},help="What stuff to load into the test set")
		parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
		parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
		parser.add_argument('--save_net',default=False,action='store_true',help="Saves the full netG_A and netG_B models for loading later")
		# Dropout and Batchnorm has different behavioir during training and test.
		parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
		parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
		# rewrite devalue values
		parser.set_defaults(model='test')
		# To avoid cropping, the load_size should be the same as crop_size
		parser.set_defaults(load_size=parser.get_default('crop_size'))
		parser.add_argument('--total_load',type=int,default=100000,help='Total number of data to load into main memory')
		parser.add_argument('--end_recurrent_only',action='store_true',help='Only evaluates the accuracy of the end sequence of recurrent data')
		parser.add_argument('--augment',action='store_true',default=False,help="Augment 3d images")
		parser.add_argument('--block_static_input',action='store_true',help='Masks all the static input into the model')
		parser.add_argument('--train_autoencoder',action='store_true',default=False,help='Train an autoencoder on sparse features for visualization purposes')
		parser.add_argument('--test_name',type=str,default="")
		parser.add_argument('--rand_encode_test',default=False,action='store_true')
		self.isTrain = False
		return parser
