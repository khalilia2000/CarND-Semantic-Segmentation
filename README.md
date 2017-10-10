# Semantic Segmentation
### Introduction
This is one of the projects in Udacity Self-Driving Car Nanodegree. In this project, pixels of a road in images are classified using a Fully Convolutional Network (FCN). The FCN architecture is explained in this [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). 

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
[Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip) were used for training and testing.  

### Rubric
#### Building the Network
1. Does the project load the pre-trained vgg model?  
`load_vgg()` function is implemented in lines 21 to 47 of `main.py`, and loads the vgg model correctly. 

2. Does the project learn the correct features from the images?  
`layers()` function is implemented in lines 53 to 105 of `main.py`, and creates the architecture of the FCN model.

3. Does the project optimize the neural network?  
`optimize()` function is implemented in lines 110 to 138 of `main.py`, and returns `train_op`, `logits`, and `cross_entropy_loss`, which are used for optimizing the network. `total_loss`, which includes L2 regularization loss is minimized using `AdamOptimizer()`.

4. Does the project train the neural network?  
`train_nn()` function is implemented in lines 143 to 182 of `main.py`, and is used for training the network.

#### Training the Network
5. Does the project train the model correctly?  
I trained the model for 30 epochs using a learning rate of 0.0003 (on CPU). L2 regularization as well as dropout was used to avoid over-fitting. Below are the results of the training:
```

Epoch: 1 - loss: 0.356 - training time: 645.2
Epoch: 2 - loss: 0.061 - training time: 652.9
Epoch: 3 - loss: 0.038 - training time: 610.1
Epoch: 4 - loss: 0.160 - training time: 618.9
Epoch: 5 - loss: 0.088 - training time: 618.6
Epoch: 6 - loss: 0.040 - training time: 619.7
Epoch: 7 - loss: 0.049 - training time: 620.6
Epoch: 8 - loss: 0.055 - training time: 620.7
Epoch: 9 - loss: 0.148 - training time: 619.6
Epoch: 10 - loss: 0.115 - training time: 620.3
Epoch: 11 - loss: 0.030 - training time: 620.3
Epoch: 12 - loss: 0.058 - training time: 620.3
Epoch: 13 - loss: 0.044 - training time: 621.0
Epoch: 14 - loss: 0.043 - training time: 619.8
Epoch: 15 - loss: 0.068 - training time: 620.2
Epoch: 16 - loss: 0.036 - training time: 620.3
Epoch: 17 - loss: 0.027 - training time: 620.3
Epoch: 18 - loss: 0.023 - training time: 621.8
Epoch: 19 - loss: 0.043 - training time: 620.2
Epoch: 20 - loss: 0.065 - training time: 620.3
Epoch: 21 - loss: 0.028 - training time: 620.6
Epoch: 22 - loss: 0.038 - training time: 620.6
Epoch: 23 - loss: 0.088 - training time: 620.3
Epoch: 24 - loss: 0.040 - training time: 620.1
Epoch: 25 - loss: 0.032 - training time: 621.0
Epoch: 26 - loss: 0.026 - training time: 619.5
Epoch: 27 - loss: 0.063 - training time: 620.2
Epoch: 28 - loss: 0.059 - training time: 619.8
Epoch: 29 - loss: 0.048 - training time: 620.1
Epoch: 30 - loss: 0.018 - training time: 621.2

```
The loss generally decreases with increasing number of epochs. However, in some instances the loss increases from epoch to epoch likely due to high learning rate. Note that `AdamOptimizer()` automatically decays the learning rate.  


6. Does the project use reasonable hyper-parameters?  
Training was performed for 30 epochs, using a batch_size of 4, and a learning rate of 0.0003 along with keep_prob of 0.5. batch_size was kept at a low value given that the model is very memory intensive, to avoid any issues.  

7. Does the project correctly label the road?  
Pictures from the latest run are included in the run folder. With augmentation of training data I am sure the results can be further improved.
 
