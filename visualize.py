import torch
import torchvision
from torchsummary import summary
import sys
import argparse

parser = argparse.ArgumentParser(description='A script for visualizing Pytorch model')
parser.add_argument("--input_file", default=None, required=True, type=str, help="Pytorch model file (.pth)")
parser.add_argument("--output_file", default="output.log", type=str, help="Human readable log file")
parser.add_argument("--summary", action="store_true", help="Generate a keras-like model summary")
parser.add_argument("--architecture", action="store_true", help="Print model")
parser.add_argument("--parameters", action="store_true", help="Print parameters")

args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file
give_summary = args.summary
give_architecture = args.architecture
give_parameters = args.parameters

original_stdout = sys.stdout
sys.stdout = open(output_file, 'w+')

## Error: given files are weights-only file
## they do not have model architecture
#model = torchvision.models.resnet18()
#model.load_state_dict(torch.load('cifar10_resnet18_retrained_acc_94.130_41.8x_irregular.pt'))
#model.eval()

## Expects a model file
model = torch.load(input_file)
model.eval()

# print all elements of the tensor
torch.set_printoptions(threshold=50000)

if give_summary:
	print('\n\n========== Summary ==========\n\n')
	# run torchsummary library
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	summary(model, input_size=(3, 224, 224))

# print model architecture
if give_architecture:
	print('\n\n========== Architecture ==========\n\n')
	print(model)

# print layer by layer information
if give_parameters:
	print('\n\n===== Parameters =====\n\n')
	for name, param in model.named_parameters():
		print('name: ', name)
		print(type(param))
		print('param.shape: ', param.shape)
		print('param.requires_grad: ', param.requires_grad)
		print('param.data: ', param.data)
		print('\n======\n')

sys.stdout = original_stdout
