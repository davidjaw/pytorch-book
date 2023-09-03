# Lab 5: Initialization and Batch Normalization for Neural Networks
This Lab illustrates the construction and utilization of a convolutional neural network (ConvNet) for image classification tasks. This ConvNet is flexible in incorporating batch normalization layers and integrates various image augmentation techniques to bolster model performance. Moreover, the code offers a comparison between different weight initialization methods, such as Xavier and He initializations, and their impacts on network training.

![](https://i.imgur.com/s7ry56Y.png)

## Core concepts
* Dynamic Weight Initialization: Applying different weight initialization schemes such as Xavier and He initializations through `torch.nn.init`.
* Batch Normalization: Dynamic incorporation of `nn.BatchNorm2d()` via boolean flags to study its effect on model convergence.