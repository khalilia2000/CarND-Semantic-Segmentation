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
`layers()` function is implemented in lines 53 to 107 of `main.py`, and creates the architecture of the FCN model.

3. Does the project optimize the neural network?  
`optimize()` function is implemented in lines 110 to 133 of `main.py`, and returns `train_op`, `logits`, and `cross_entropy_loss`, which are used for optimizing the network.

4. Does the project train the neural network?  
`train_nn()` function is implemented in lines 138 to 177 of `main.py`, and is used for training the network.

#### Training the Network
5. Does the project train the model correctly?  
I trained the model for 5 epochs using a learning rate of 0.0005 (on CPU). L2 regularization as well as dropout was used to avoid over-fitting. Below are the results of the training:
```

Epoch: 1 - loss: 0.733 - training time: 1477.5
Epoch: 2 - loss: 0.511 - training time: 1460.3
Epoch: 3 - loss: 0.369 - training time: 1457.0
Epoch: 4 - loss: 0.318 - training time: 1457.3
Epoch: 5 - loss: 0.196 - training time: 1490.0

```

6. Does the project use reasonable hyper-parameters?  
Training was performed for 5 epochs, using a batch_size of 64, and a learning rate of 0.0005 along with keep_prob of 0.5

7. Does the project correctly label the road?  
Pictures from the latest run are included in the run folder. With more epochs and augmented training data I am sure the results can be improved; however, given that I was running on CPU, 5 epochs was judged to be sufficient to show that the model works.
 
