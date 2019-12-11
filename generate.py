import torch
import torchvision
import sys
import types
import os
import warnings

warnings.simplefilter("ignore")

# create directories
model_dir = "models"
log_dir = "logs"
base = os.path.dirname(os.path.abspath(__file__))
model_path = base+"/"+model_dir
log_path = base+"/"+log_dir

try:
    os.mkdir(model_path)
except OSError:
    print("Creation of the directory %s failed" % model_path)
else:
    print("Successfully created the directory %s " % model_path)

try:
	os.mkdir(log_dir)
except OSError:
	print("Creation of the directory %s failed" % log_path)
else:
	print("Successfully create the directory %s" % log_path)

# generate some models for visualization
models = {model: getattr(torchvision.models, model) for model in dir(torchvision.models)
if isinstance(getattr(torchvision.models, model), types.FunctionType)}

for name, f in models.items():
	try:
		model = f(pretrained=True)
		model.eval()
		torch.save(model, model_path+"/"+name+".pt")
	except:
		print("Unexpected error: ", sys.exc_info())
		pass
