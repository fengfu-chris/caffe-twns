# Ternary Weight Networks (TWNs)

This repository implements the benchmarks in our paper "Ternary Weight Networks" which was accepted by the 1st NIPS Workshop on Efficient Methods for Deep Neural Networks (EMDNN), 2016.  

Please cite TWNs in your publications if it helps your research:

    @article{li2016ternary,
      Author = {Li, Fengfu and Zhang, Bo and Liu, Bin},
      Journal = {arXiv preprint arXiv:1605.04711},
      Title = {Ternary Weight Networks},
      Year = {2016}
    }

## Build

Dependencies are identical with the master branch of Caffe. Check out [project site](http://caffe.berkeleyvision.org) for detailed instructions.

NOTE:  
1. Some layers may only have GPU implementation. Thus, CUDA support and GPU devices are required.  
2. The Makefile has been modified to accomodate Ubuntu 16.04. For previous version of Ubuntu, please replace the Makefile with the original one.

## Steps to run a demo  
1. Preparing data  
$./data/mnist/get_mnist.sh  

2. Converting data to lmdb  
$./examples/mnist/create_mnist.sh

3. Configurations  
3.1 setting the PRECISION in the train_lenet_tn.sh  
3.2 setting the DELTA value (0.7 default)  

4. Training  
$cd examples/mnist  
$sh train_lenet_tn.sh

5. Run-time usage (to be added)

## Contact

You are welcome to send message to (lifengfu12@mails.ucas.ac.cn) if you have any issue on this code.


