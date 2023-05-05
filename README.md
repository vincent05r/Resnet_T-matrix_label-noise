# 



requirements of running the code

• We recommends using a GPU to run our code, idealy a GPU with a VRM higher than 8 GB
• CPU can be used, however it will take forever to complete :)
• linux environment is preferred, our code is developed under linux environment
• Install the Python 3.9
• Install cudatoolkit 11.3.1, cudnn 8.2.1, pytorch 1.10.2, torchvision 0.11.3
• Install the Python package numpy 1.20.1, PIL 8.2.0



Instruction of running the code, and explanation of each file

• We have different python files for each part of the work
• Directories will be generated automatically to store training log, transition matrix, model configuration as well as probability matrix

• cifar_main.py will estimate transition matrix as well as train a classifier using the estimated transition matrix
    • type python cifar_main.py to run it
    • The model for training can be changed in line 26, We support Resnet18,34,50,154
    • configuration of the model can be adjusted in the configuration dictionary

• cifar_benchmark.py will train a NN without using transition matrix, it is used for benchmarking the performance between using transition matrix and without using transition matrix.
    • type python cifar_benchmark.py to run it

• FMN_estimation will estimate transition matrix for both FashionMINIST dataset
    • type python FMN_estimation.py to run it

• FMN_train will train 2 classifier(RESNET 18 AND RESNET 34) using the given transition matrix
    • type python FMN_train.py to run it


Other modules

• We implemented our Resnet for 3 and 1 color channel in the Resnet.py
• We implemented our data augmentation code in transformer.py 
• We implemented our data loading code in data_load.py, it is designed for pytorch's dataloader
• We implemented our data split function and computing transition matrix funtion in the tools.py
