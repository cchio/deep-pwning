<p align="center">
  <img src="https://github.com/cchio/deep-pwning/blob/master/dpwn/repo-assets/dpwn-splash.png?raw=true" alt="Deep-pwning splash"/>
</p>
Deep-pwning is a lightweight framework for experimenting with machine learning models with the goal of evaluating their robustness against a motivated adversary.

Note that deep-pwning in its current state is ***no where*** close to maturity or completion. It is meant to be experimented with, expanded upon, and extended by you. Only then can we help it truly become the goto __penetration testing toolkit for statistical machine learning models__.

## Background
Researchers have found that it is surprisingly trivial to trick a machine learning model (classifier, clusterer, regressor etc.) into making an objectively wrong decisions. This field of research is called [Adversarial Machine Learning](https://people.eecs.berkeley.edu/~tygar/papers/SML2/Adversarial_AISEC.pdf). It is not hyperbole to claim that any motivated attacker can bypass any machine learning system, given enough information and time. However, this issue is often overlooked when architects and engineers design and build machine learning systems. The consequences are worrying when these systems are put into use in critical scenarios, such as in the medical, transportation, financial, or security-related fields.

Hence, when one is evaluating the efficacy of applications using machine learning, their malleability in an adversarial setting should be measured alongside the system's precision and recall.

This tool was released at DEF CON 24 in Las Vegas, August 2016, during a talk titled [Machine Duping 101: Pwning Deep Learning Systems](https://www.defcon.org/html/defcon-24/dc-24-speakers.html#Chio).

## Structure
This framework is built on top of [Tensorflow](https://www.tensorflow.org/), and many of the included examples in this repository are modified Tensorflow examples obtained from the [Tensorflow GitHub repository](https://github.com/tensorflow/tensorflow).

All of the included examples and code implement **deep neural networks**, but they can be used to generate adversarial images for similarly tasked classifiers that are not implemented with deep neural networks. This is because of the phenomenon of 'transferability' in machine learning, which was Papernot et al. expounded expertly upon in [this paper](https://arxiv.org/abs/1605.07277). This means means that adversarial samples crafted with a DNN model *A* may be able to fool another distinctly structured DNN model *B*, as well as some other SVM model *C*.

This figure taken from the aforementioned paper (Papernot et al.) shows the percentage of successful adversarial misclassification for a source model (used to generate the adversarial sample) on a target model (upon which the adversarial sample is tested).

<p align="center">
  <img src="https://github.com/cchio/deep-pwning/blob/master/dpwn/repo-assets/transferability-misclassificaiton-matrix.png?raw=true" alt="Transferability Misclassificaiton Matrixh"/>
</p>

## Components
Deep-pwning is modularized into several components to minimize code repetition. Because of the vastly different nature of potential classification tasks, the current iteration of the code is optimized for classifying **images** and **phrases** (using word vectors).

These are the code modules that make up the current iteration of Deep-pwning:

1. Drivers

   The **drivers** are the *main* execution point of the code. This is where you can tie the different modules and components together, and where you can inject more customizations into the adversarial generation processes.

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
   <p align="center">
      <img src="https://github.com/cchio/deep-pwning/blob/master/dpwn/repo-assets/lenet5.png?raw=true" alt="LeNet-5"/>
   </p>
   LeCun et al. [LeNet-5 Convolutional Neural Network](http://yann.lecun.com/exdb/lenet/)

3. Adversarial (advgen)

   This module contains the code that generates adversarial output for the models. The ```run()``` function defined in each of these ```advgen``` classes takes in an ```input_dict```, that contains several predefined tensor operations for the machine learning model defined in Tensorflow. If the model that you are generating the adversarial sample for is known, the variables in the input dict should be based off that model definition. Else, if the model is unknown, (black box generation) a substitute model should be used/implemented, and that model definition should be used. Variables that need to be passed in are the input tensor placeholder variables and labels (often refered to as ```x``` -> input and ```y_``` -> labels), the model output (often refered to as ```y_conv```), and the actual test data and labels that the adversarial images will be based off of.

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

## Getting Started

### Installation
Please follow the directions to install tensorflow found here https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html which will allow you to pick the tensorflow binary to install. 
```bash
$ pip install -r requirements.txt
```

### Execution Example (with the MNIST driver)
To restore from a previously trained checkpoint. (configuration in config/mnist.conf)
``` bash
$ cd dpwn
$ python mnist_driver.py --restore_checkpoint
```
To train from scratch. (note that any previous checkpoint(s) located in the folder specified in the configuration will be overwritten)
``` bash
$ cd dpwn
$ python mnist_driver.py
```

## Task list
- [ ] Implement saliency graph method of generating adversarial samples
- [ ] Add ```defense``` module to the project for examples of some defenses proposed in literature
- [ ] Upgrade to Tensorflow 0.9.0
- [ ] Add support for using pretrained word2vec model in ```sentiment driver```
- [ ] Add SVM & Logistic Regression support in ```models``` (+ example that uses them)
- [ ] Add non-image and non-phrase classifier example
- [ ] Add multi-GPU training support for faster training speeds

## Requirements
+ [Tensorflow 0.8.0](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
+ [Matplotlib >= 1.5.1](http://matplotlib.org/faq/installing_faq.html)
+ [Numpy >= 1.11.1](https://pypi.python.org/pypi/numpy)
+ [Pandas >= 0.18.1](http://pandas.pydata.org/pandas-docs/stable/install.html)
+ [Six >= 1.10.0](https://pypi.python.org/pypi/six)

Note that dpwn requires Tensorflow 0.8.0. Tensorflow 0.9.0 introduces some

## Contributing 
(borrowed from the amazing [Requests repository](https://github.com/kennethreitz/requests) by kennethreitz)

+ Check for open issues or open a fresh issue to start a discussion around a feature idea or a bug.
+ Fork the repository on GitHub to start making your changes to the **master** branch (or branch off of it).
+ Write a test which shows that the bug was fixed or that the feature works as expected.
+ Send a pull request and bug the maintainer until it gets merged and published. :) Make sure to add yourself to ```AUTHORS.md```.

## Acknowledgements

There is so much impressive work from so many machine learning and security researchers that directly or indirectly contributed to this project, and inspired this framework. This is an inconclusive list of resources that was used or referenced in one way or another:

### Papers
+ Szegedy et al. [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)
+ Papernot et al. [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528)
+ Papernot et al. [Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples](https://arxiv.org/abs/1602.02697)
+ Goodfellow et al. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
+ Papernot et al. [Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples](https://arxiv.org/abs/1605.07277)
+ Grosse et al. [Adversarial Perturbations Against Deep Neural Networks for Malware Classification](http://arxiv.org/abs/1606.04435)
+ Nguyen et al. [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](http://arxiv.org/abs/1412.1897)
+ Xu et al. [Automatically Evading Classifiers: A Case Study on PDF Malware Classifiers](https://www.cs.virginia.edu/~evans/pubs/ndss2016/)
+ Kantchelian et al. [Evasion and Hardening of Tree Ensemble Classifiers](https://arxiv.org/abs/1509.07892)
+ Biggio et al. [Support Vector Machines Under Adversarial Label Noise](http://www.jmlr.org/proceedings/papers/v20/biggio11/biggio11.pdf)
+ Biggio et al. [Poisoning Attacks against Support Vector Machines](http://arxiv.org/abs/1206.6389)
+ Papernot et al. [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](http://arxiv.org/abs/1511.04508)
+ Ororbia II et al. [Unifying Adversarial Training Algorithms with Flexible Deep Data Gradient Regularization](http://arxiv.org/abs/1601.07213)
+ Jin et al. [Robust Convolutional Neural Networks under Adversarial Noise](http://arxiv.org/abs/1511.06306)
+ Goodfellow et al. [Deep Learning Adversarial Examples – Clarifying Misconceptions](http://www.kdnuggets.com/2015/07/deep-learning-adversarial-examples-misconceptions.html)

### Code
+ [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
+ WildML [Implementing a CNN for Text Classification in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
+ [wgrathwohl/captcha_crusher](https://github.com/wgrathwohl/captcha_crusher)
+ [josecl/cool-php-captcha](https://github.com/josecl/cool-php-captcha)

### Datasets
+ Krizhevsky et al. [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
+ LeCun et al. [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
+ Pang et al. [Movie Review Data (v2.0 from Rotten Tomatoes)](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
