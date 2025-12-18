## Source code for Continual Learning via Variational Bayesian Dropout [VBD-CL] (https://tunglamlqddb.github.io/files/BS_thesis_summarize.pdf)
The roles of the main implementation files are as follows:
- conv_net.py: CNN model for the Split CIFAR100 and Split CIFAR10-100 datasets.
- omniglot_conv_net.py: CNN model for the Split Omniglot dataset.
- model.py: MLP model for the Split MNIST and Permuted MNIST datasets.
- layers_VBD: implementation of the noisy (variational) Dropout layer.
- data.py: data generation for continual learning scenarios.
- Notes: Run the file test_vbd_sgd.py to conduct experiments. Parameters in this file need to be set manually, and the dataset folder path in data.py needs to be modified accordingly.
