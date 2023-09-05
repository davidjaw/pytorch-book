# Lab 6: Advanced usage of pytorch

In this laboratory exercise, we shall explore prevalent file architectures and acquaint ourselves with the methodology of training and evaluation via custom-designed model wrappers. Moreover, the utility of TensorBoard will be employed to facilitate real-time monitoring of the training procedure. In addition to this, a sophisticated computational block will be instantiated through the utilization of PyTorch's `nn.Module`. The incorporation of `nn.Sequential` and `nn.ModuleList` further enables us to architect a model with dynamic control over computational complexity and scalability.

![](https://i.imgur.com/gMa8H5V.png)

![](https://i.imgur.com/RomlKoV.png)

## Core concepts
* Common File Architecture: A canonical file hierarchy tailored for the systematic organization of deep learning endeavors.
* Model Wrapper: An abstraction layer in the form of a wrapper class that encapsulates both the training and evaluation workflows, thereby streamlining the model lifecycle.
* PyTorch Module: Utilization of PyTorch's `nn.Module`, `nn.Sequential`, and `nn.ModuleList` for the construction of a model endowed with the capability for dynamic configurability in terms of computational complexity and scale.
* Save and Load Model: The practice of persisting and retrieving model parameters through `torch.save` and `torch.load`, essential for model versioning and transfer learning.
* tqdm: A real-time, interactive progress bar utility designed to offer visibility into the training and validation procedures.


