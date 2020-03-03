# Understanding the Intrinsic Robustness of Image Distributions using Conditional Generative Models 

*A repository for understanding the intrinsic robustness limits for robust learning against adversarial examples. Created by [Xiao Zhang](https://people.virginia.edu/~xz7bc/) and [Jinghui Chen](https://web.cs.ucla.edu/~jhchen/). [Link to the ArXiv paper](https://arxiv.org/abs/2003.00378).* 

The goals of this project are to:

1. Theoretically, derive an intrinsic robustness bound with respect to L2 perturbations, for any input distribution that can be captured by some conditional generative model.

2. Empirically, evaluate the intrinsic robustness bound for various synthetically generated image distributions, with comparisons to the robustness of SOTA robust classifiers.

## Installation
The code was developed using Python3 on [Anaconda](https://www.anaconda.com/download/#linux).

* Install Pytorch 1.1.0: 
    ```text
    conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
    ```

* Install other dependencies:
    ```text
    pip install -r requirements.txt
    ```
    
## Examples for MNIST experiments

1. Train an ACGAN model using the original MNIST dataset:  
    ```text
    python build_generator_mnist.py --gan-type ACGAN --mode train
    ```

2. Estimate the local Lipschitz constant and reconstruct MNIST dataset using ACGAN:
    ```text
    python build_generator_mnist.py --gan-type ACGAN --mode evaluate
    ```

    ```text
    python build_generator_mnist.py --gan-type ACGAN --mode reconstruct
    ```

3. Train various robust classifiers under L2 perturbations on the generated MNIST dataset:
    ```text
    cd train_classifiers && python train_mnist.py --method zico
    ```

4. Evaluate unconstrained and/or in-distribution robustness for the trained classifiers:
    ```text
    python test_robustness_mnist.py --method madry --robust-type in
    ```

## Examples for ImageNet10 Experiments

1. Dowload  the pretrained BigGAN model, estimate Lipschitz and reconstruct ImageNet10:
    ```text
    python build_generator_imagenet.py --mode evaluate
    ```

    ```text
    python build_generator_imagenet.py --mode reconstruct
    ```

2. Train various robust classifiers under L2 perturbations on the generated ImageNet10:
    ```text
    cd train_classifier && python train_imagenet.py --method trades
    ```

3. Evaluate unconstrained and/or in-distribution robustness for the trained classifiers:
    ```text
    python test_robustness_imagenet.py --method trades --robust-type unc
    ```
    
## What is in this Respository?

* Folder ```geneartive```, including:
  * ```src```: folder that contains functions for building BigGAN
  * ```acgan.py, gan.py```: functions for training MNIST generative models
  * ```biggan.py```: neural network architecture for BigGAN generator
  * ```utils.py```: auxiliary functions for generative models

* Folder ```train_classifer```, including:
  * ```adv_loss.py```: adversarial loss functions for Madry and TRADES 
  * ```attack.py```: functions for generating unc/in-dist adversarial examples
  * ```problem.py```: define datasets, dataloaders and model architectures
  * ```trainer.py```: implements the train and evaluation functions using different methods
  * ```train_mnist.py, train_imagenet.py```: main functions for training classifiers on generated MNIST and ImageNet10

* ```build_generator_mnist.py, build_generator_mnist.py```: main functions for generating image datasets

* ```test_robustness_mnist.py, test_robustness_imagenet.py```: main functions for evaluating adversarial robustness
