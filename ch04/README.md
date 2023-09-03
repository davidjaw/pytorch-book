# Ch04 Classification Problem

## Lab4-1 
This lab contains Python implementations that are examples of various operations commonly employed in the realm of Convolutional Neural Networks (CNNs). 
Built on the PyTorch framework, the code samples include creating a basic convolutional network, applying convolution with a fixed kernel, and performing image augmentation.
### ex1_build_convolution_net
This example elucidates the architecture of a simple convolutional neural network (CNN). The CNN employs a single convolutional layer to process an image with a shape of 3x28x28. Moreover, the architecture's parameters are inspected and summarized using the `torchinfo.summary` function.

#### Core Concepts
* Convolutional Layer (`nn.Conv2d`)
* Rectified Linear Unit (`F.relu`)
* Parameter Summary (`torchinfo.summary`)

### ex2_convolution_with_fixed_kernel
This code snippet presents an image processing pipeline that employs predetermined convolutional kernels, such as Sobel and Laplacian, to manipulate an image. The image is first downloaded from a URL, transformed into a tensor, and then convolved with these kernels.

#### Core Concepts
* Sobel, Laplacian, Sharpen, Gaussian Blur Filters
* Convolutional Operation (`F.conv2d`)
* Dynamic Resource Allocation (`urllib.request.urlretrieve`)

### ex3_image_augmentation
The example demonstrates how to leverage the `torchvision.datasets` and `torchvision.transforms` modules for performing image augmentation techniques such as rotation, cropping, and color adjustments.

#### Core Concepts
* Image Augmentation (Random Affine, Random Crop, Color Jitter)
* Data Loading (DataLoader)
* Data Transformation (`torchvision.transforms`)

### ex4_random_augmentation
This example provides an exploration into the stochastic data augmentation techniques available in the torchvision.transforms module. It emphasizes the role of random transformations, such as random flipping, color alterations, rotational perturbations, and scaling, in enhancing the generalization capabilities of machine learning models. The effects of these transformations are visualized using matplotlib.

#### Core Concepts
* Stochastic Data Augmentation (`torchvision.transforms`)
* Probability Parameterization (`p` in `RandomHorizontalFlip`)
* Image Rendering (matplotlib)

## Lab 4-2
This lab explores the nuances between Fully Connected Networks and Convolutional Neural Networks in classification tasks. It additionally investigates the impact of data augmentation techniques on performance metrics. The lab also serves as a hands-on tutorial for leveraging TensorBoard for visualization and provides a framework for constructing Neural Network training pipelines.

#### Core Concepts
* TensorBoard Visualization
* Data Augmentation
* Training Pipeline
* Convolutional Neural Networks
* Fully Connected Networks
* Performance Metrics
