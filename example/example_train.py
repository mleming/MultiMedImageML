from options.train_options import *

import os
import sys
import time
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

model = MultiInputModule(weights=opt.pretrained_model,variational=opt.variational)
model.cuda(opt.gpu_ids[0])

assert(os.path.isfile(opt.all_vars))
medim_loader = MedImageLoader(opt.all_vars,
	X_dim=opt.X_dim,
	cache=True,
	label=opt.label,
	confounds=opt.confounds,
	return_obj = True,
	dtype="torch",
	val_ranges=opt.val_ranges,
	static_inputs=opt.static_inputs,
	augment=opt.augment,
	key_to_filename=key_to_filename,
	gpu_ids = opt.gpu_ids,
	precedence=opt.precedence,
	batch_by_pid=True
	)

trainer = MultiInputTrainer(model,
		batch_size=opt.batch_size,
		lr=opt.lr,
		name=opt.name,
		checkpoint_dir=opt.checkpoint_dir,
		loss_image_dir=os.path.join(opt.results_dir,
						opt.name,"loss_ims"),
		save_latest_freq = opt.save_latest_freq
	)

for i in range(opt.epochs):
	print(f"Epoch {i}")
	j=0
	t = time.time()
	l = len(medim_loader)
	for p in medim_loader:
		j += 1
		plen = len(p.image_records)
		trainer.loop(p,dataloader=medim_loader)
		sys.stdout.write('\r')
		sys.stdout.write(f"Iters: {j}/{l}; len(image_records): {plen}")
		sys.stdout.flush()
	t = time.time() - t
	print(f" Time: {t} seconds")
