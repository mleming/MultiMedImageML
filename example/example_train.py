from options import *
from multi_med_image_ml import *


opt = options.TrainOptions()

model = MultiInputModule((32,32),regressor_dims=(32,32))

medim_loader = MedImageLoader(imfolder1,imfolder2,
	cache=True,
	label=["MRAcquisitionType","ImageOrientationPatientDICOM"],
	confounds=["Slice Thickness","Repetition Time"],
	return_obj = True,
	dtype="torch",
	batch_size=14)


trainer = MultiInputTrainer(model,batch_size=2)

for i in range(opt.epochs):
	print(f"Epoch {i}")
	for p in medim_loader:
		trainer.loop(p,dataloader=medim_loader)
