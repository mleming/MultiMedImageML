from options.train_options import *

import os
import sys
wd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,wd)
sys.path.insert(0,os.path.join(wd,'src'))
sys.path.insert(0,os.path.join(wd,'src','multi_med_image_ml'))

from src.multi_med_image_ml.MedImageLoader import *
from src.multi_med_image_ml.models import *
from src.multi_med_image_ml.MultiInputTrainer import *
# Path function, used when switching platforms.

def path_func(filename,reverse=False):
	prefix = os.path.join(os.path.dirname(os.path.dirname(wd)),'MGH_ML_pipeline')
	suffix = '_resized_96.npy'
	if reverse:
		return os.path.join(prefix,filename[1:] + suffix)
	else:
		return os.sep + filename[len(prefix):-len(suffix)]

opt = TrainOptions().parse()

model = MultiInputModule(weights=opt.pretrained_model)

medim_loader = MedImageLoader(opt.all_vars,
	dim=opt.dim,
	cache=True,
	label=opt.label,
	confounds=opt.confounds,
	return_obj = True,
	dtype="torch",
	val_ranges=opt.val_ranges,
	static_inputs=opt.static_inputs,
	augment=opt.augment)

trainer = MultiInputTrainer(model,
		batch_size=opt.batch_size,
		lr=opt.lr,
		name=opt.name,
		checkpoint_dir=opt.checkpoint_dir,
		loss_image_dir=os.path.join(opt.results_dir,
						opt.name,"loss_ims")
	)

for i in range(opt.epochs):
	print(f"Epoch {i}")
	for p in medim_loader:
		print(medim_loader.mode)
		print(medim_loader.tl())
		trainer.loop(p,dataloader=medim_loader)
