from options.test_options import *

import os
import sys
import time
wd = os.path.dirname(os.path.realpath(__file__))

from multi_med_image_ml.MedImageLoader import *
from multi_med_image_ml.models import *
from multi_med_image_ml.MultiInputTester import *
# Path function, used when switching platforms.

def key_to_filename(key,reverse=False):
	prefix = os.path.join(os.path.dirname(wd),'MGH_ML_pipeline')
	suffix = '_resized_96.npy'
	if not reverse:
		string = os.path.join(prefix,key[1:] + suffix)
	else:
		string = key[len(prefix):-len(suffix)]
	return string

opt = TestOptions().parse()

model = MultiInputModule(variational=opt.variational)
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

tester = MultiInputTester(
		medim_loader.database,
		model,
		name=opt.name,
		test_name=opt.test_name,
		checkpoint_dir=opt.checkpoint_dir,
		out_record_folder=os.path.join(opt.results_dir,opt.name),
		database_key="ProtocolNameSimplified"
	)

t = time.time()
j=0
l=4
for p in medim_loader:
	plen = len(p.image_records)
	if plen == 1: continue
	tester.grad_cam(p)
	sys.stdout.write('\r')
	sys.stdout.write(f"Iters: {j}/{l}; len(image_records): {plen}")
	sys.stdout.flush()
	j += 1
	if j == l: break
print("")
