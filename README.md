# Deep-pwning (dpwn)
Deep-pwning is a lightweight framework for experimenting with machine learning models with the goal of evaluating their robustness against a motivated adversary.
![dpwn-splash](https://github.com/cchio/deep-pwning/blob/master/dpwn/repo-assets/dpwn-splash.png "Deep-pwning splash")

### Background
Researchers have found that it is surprisingly trivial to trick a machine learning model (classifier, clusterer, regressor etc.) into making an objectively wrong decisions. This field of research is called [Adversarial Machine Learning](https://people.eecs.berkeley.edu/~tygar/papers/SML2/Adversarial_AISEC.pdf). It is not hyperbole to claim that any motivated attacker can bypass any machine learning system, given enough information and time. However, this issue is often overlooked when architects and engineers design and build machine learning systems. The consequences are worrying when these systems are put into use in critical scenarios, such as in the medical, transportation, financial, or security-related fields.

Hence, when one is evaluating the efficacy of applications using machine learning, their malleability in an adversarial setting should be measured alongside the system's precision and recall.

This tool was released at DEF CON 24 in Las Vegas, August 2016, during a talk titled [Machine Duping 101: Pwning Deep Learning Systems](https://www.defcon.org/html/defcon-24/dc-24-speakers.html#Chio).

### Structure
This framework is built on top of [Tensorflow](https://www.tensorflow.org/), and many of the included examples in this repository are modified Tensorflow examples obtained from the [Tensorflow GitHub repository](https://github.com/tensorflow/tensorflow).

All of the included examples and code implement **deep neural networks**, but they can be used to generate adversarial images for similarly tasked classifiers that are not implemented with deep neural networks. 

### Components
Deep-pwning is modularized into several components to minimize code repetition. Because of the vastly different nature of potential classification tasks, the current iteration of the code is optimized for classifying **images** and **phrases** (using word vectors).

These are the code modules that make up the current iteration of Deep-pwning:

1. Drivers

   The **drivers** are the *main* execution point of the code. This is where you can tie the different modules and comoponents together, and where you can inject more customizations into the adversarial generation processes.

2. Models

   This is where the actual machine learning model implementations are located. For example, the provided ```lenet5``` model definition is located in the ```model()``` function witihn ```lenet5.py```. It defines the network as the following: 
   ```
      -> Input
      -> Convolutional Layer 1
      -> Max Pooling Layer 1
      -> Convolutional Layer 2
      -> Max Pooling Layer 2
      -> Dropout Layer
      -> Softmax Layer
      -> Output
   ```
   ![lenet5](https://github.com/cchio/deep-pwning/blob/master/dpwn/repo-assets/lenet5.png "LeNet5")
   
   LeCun et al. [LeNet-5 Convolutional Neural Network](http://yann.lecun.com/exdb/lenet/)

3. Adversarial (advgen)

   This module contains the code that generates adversarial output for the models. The ```run()``` function defined in each of these ```advgen``` classes take in an ```input_dict```, that contains several predefined tensor operations for the machine learning model defined in Tensorflow. If the model that you are generating the adversarial sample for is known, the variables in the input dict should be based off that model definition. Else, if the model is unknown, (black box generation) a substitute model should be used/implemented, and that model definition should be used. Variables that need to be passed in are the input tensor placeholder variables and labels (often refered to as ```x``` -> input and ```y_``` -> labels), the model output (often refered to as ```y_conv```), and the actual test data and labels that the adversarial images will be based off of.

4. Config

   Application configurations.

5. Utils

   Miscellaneous utilities that don't belong anywhere else. These include helper functions to read data, deal with Tensorflow queue inputs etc.

These are the resource directories relevant to the application:

1. Checkpoints

   Tensorflow allows you to load a partially trained model to resume training, or load a fully trained model into the application for evaluation or performing other operations. All these saved 'checkpoints' are stored in this resource directory.
   
2. Data

   This directory stores all the input data in whatever format that the driver application takes in.

3. Output

   This is the output directory for all application output, including adversarial images that are generated.

### Getting Started

### Requirements

### Contributing

### Acknowledgements

