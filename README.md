# Visualize
A script for visualizing Pytorch models

### Dependencies

1. Pytorch

Install Pytorch from [Pytorch website](https://pytorch.org). You can use pip or conda for it. 

2. Torch Summary

```
pip install torchsummary
```

### Usage

Follow the below mentioned steps.

1. Setup directories and sample models for visualization

```
python generate.py
```

This should have created two directories - `models` and `logs`. Ignore generated error or warning messages.

2. Run visualize help command to preview command line arguments

```
python visualize.py --h
```

```
usage: visualize.py [-h] --input_file INPUT_FILE [--output_file OUTPUT_FILE]
                    [--summary] [--architecture] [--parameters]

A script for visualizing Pytorch model

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Pytorch model file (.pth)
  --output_file OUTPUT_FILE
                        Human readable log file
  --summary             Generate a keras-like model summary
  --architecture        Print model
  --parameters          Print parameters
```

`summary` presents a table with `Layer`, `Output Shape` and `Param #` as columns. This is generated using [Torch Summary](https://github.com/sksq96/pytorch-summary) library.

`architecture` presents the model using Pytorch's `print()` method.

`parameters` presents all weights in the network as complete tensors. This is generated using Pytorch's `named_parameters()` method. One can map `name` with a layer in `architecture`. 

3. Run visualize with a sample model

```
python visualize.py --summary --architecture --parameters --input_file=models/resnext50_32x4d.pt --output_file=logs/resnext50_32x4d.log
```

This creates `resnext50_32x4d.log` in `logs` directory with all requested information.

4. Run visualize with a given model

Copy your model to `models` directory.

```
python visualize.py --summary --architecture --parameters --input_file=models/<model_name>.pt --output_file=logs/<model_name>.log
``` 

Replace `<model_name>` with your model file name. You can disable one or more of `summary`, `architecture` or `parameters` by not adding its corresponding flag to the command. 

Running this command should generate a `<model_name>.log` output file in `logs`.

Note `<model_name>.pt` should be a complete Pytorch model file, meaning it should have both model architecture as well pretrained weights. If it is a weight-only file, you would observe the following error message - `AttributeError: 'collections.OrderedDict' object has no attribute 'eval` if using `--summary` option.

To run a weight-only file, remove `--summary` and `--parameters` flags, like below

```
python visualize.py --architecture --input_file=models/imagenet_resnet18_acc_89.082_6.4x.pt --output_file=logs/imagenet_resnet18_acc_89.082_6.4x.log
``` 

This should print out the complete weights in log file.
