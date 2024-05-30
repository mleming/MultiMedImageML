from options.train_options import *

import os
import sys
wd = os.path.dirname(os.path.realpath(__file__))

from multi_med_image_ml.MedImageLoader import *
from multi_med_image_ml.models import *
from multi_med_image_ml.MultiInputTrainer import *
# Path function, used when switching platforms.

def key_to_filename(key,reverse=False):
	prefix = os.path.join(os.path.dirname(wd),'MGH_ML_pipeline')
	suffix = '_resized_96.npy'
	if not reverse:
		string = os.path.join(prefix,key[1:] + suffix)
	else:
		string = key[len(prefix):-len(suffix)]
	return string

opt = TrainOptions().parse()

model = MultiInputModule(weights=opt.pretrained_model)
model.cuda(opt.gpu_ids[0])

medim_loader = MedImageLoader(opt.all_vars,
	dim=opt.dim,
	cache=True,
	label=opt.label,
	confounds=opt.confounds,
	return_obj = True,
	dtype="torch",
	val_ranges=opt.val_ranges,
	static_inputs=opt.static_inputs,
	augment=opt.augment,
	key_to_filename=key_to_filename,
	gpu_ids = opt.gpu_ids
	)

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
		trainer.loop(p,dataloader=medim_loader)
