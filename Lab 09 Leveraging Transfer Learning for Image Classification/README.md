# Lab 9 Leveraging Transfer Learning for Image Classification
In this laboratory exercise, we delve into the realms of Transfer Learning, a predominant approach in contemporary deep learning techniques for leveraging pre-trained models to facilitate the training process in related tasks. Our focal point will be the implementation and optimization of a binary classification model distinguishing between images of cats and dogs by fine-tuning a pre-trained model acquired from the rich reservoir of pre-trained models available in the PyTorch hub.

## Dataset
We use the Dog vs. Cat Kaggle dataset, which can be downloaded from [here](https://www.kaggle.com/c/dogs-vs-cats). 
The dataset contains 25,000 images of cats and dogs (12,500 from each class). 

## Core concepts
* Transfer Learning: Understand and implement transfer learning by utilizing a pre-trained MobileNetV3 model and adapting it to our specific task. We will experiment with scenarios both with and without leveraging the pre-trained parameters to comprehend the benefits 
* Model Architecture Modification: Learn to manipulate the architecture of a deep neural network by altering its final layers to suit the demands of our binary classification task.
* Layer Freezing: Grasp the concept and practice of freezing layers during training to retain pre-learned features and expedite the training process.
* Dataset Loading and Preprocessing: Acquire the skills to load and preprocess a locally stored dataset, an essential step in data preparation for training deep learning models.
