# HyperParameter Optimization with VGG

---

## Assignment

### Part 1. Basic PyTorch
1. Install pytorch and torchvision, for questions first refer to https://pytorch.org/docs/stable/index.html.
2. Preliminary: read MNIST example https://github.com/pytorch/examples/tree/master/mnist.
3. Write a naive CNN on CIFAR10 based on the above script,

   a. use CIFAR10 instead of MNIST by modifying train_loader and test_loader
   
   b. adjust Net for data compatibility (MNIST is 28x28x1, but CIFAR10 is 32x32x3).
   
   c. misc: arguments shall be modified accordingly. 
         
      The test accuracy shall be around 72%, with minimum modification (only change input size for the first conv2d layer and the first linear layer).

           

### Part 2. VGG on CIFAR10
0. One can build VGG from scratch by rewriting Net. For VGG structure we refer readers to Table 2 of the paper https://arxiv.org/abs/1409.1556. Alternatively, a quick solution is to plug the torchvision predefined model in the above script. For the documentation check https://pytorch.org/vision/stable/models.html. To begin with the task, use Adam with learning rate 2e-4, b1 0.5, and b2 = 0.999. For Conv2d initialization, use nn.init.kaiming_normal_. Training a network takes about 30 min on single GPU. For model performance uses test set accuracy. If possible, plot train loss and test accuracy for better view.
   <br><br>
1. Examine the components of VGG and their effects on the model performances:
   
   a. Network going depth improves performance: build 11-layers VGG, run 100 epochs, monitor train loss and test accuracy.
   <br> Hint: find vgg11 from torchvision models or rewrite Net according to column A in aforementioned Table 2.
   <br> Reference test accuracy 78%
   <br><br>
   b. Use batch normalization for better performance
   <br> Hint: use vgg11_bn instead or insert nn.BatchNorm2d after Conv2d
   <br> Reference test accuracy 83%
   <br><br>
   c. Compare activation: relu used in the original paper, vs. leaky_relu, another activation.
   <br> Hint: reload vision.models.vgg and substitute nn.ReLU with nn.LeakyReLU, or modify your model accordingly.
   <br> Reference test accuracy 74%
   <br><br>

2. Different training methods influence the trained model, assume leaky_relu is used:

   a. Choose a different optimizer.
   <br> Hint: find optimizer in train, substitute torch.optim.Adam with torch.optim.SGD
   <br> Reference test accuracy: 66%
   <br><br>
   b. Change batch_size.
   <br> Hint: change batch size to 256 in arguments.
   <br> Reference test accuracy 77%
   <br><br>
   c. Different initializations also effect model performance.
   <br> Hint: find nn.init.kaiming_normal_ in vision.models.vgg, change to nn.init.xavier_uniform_.
   <br> Reference test accuracy 82%
   <br><br>
   d. Dropout helps model generalization.
   <br> Hint: find nn.Dropout and disable it.
   <br> Reference test accuracy 67%

---

## Solution

1. Install python3
2. Install pytorch use: https://pytorch.org/get-started/locally/
3. Install tensorboard and tensorflow use: 
      
         sudo pip3 install tensorboard tensorflow
   
4. To run the assignment
   
         python3 assignment_p1.py
         python3 assignment_p2.py
   
   assignment_p1.py is the solution for part 1 of assignment
   
   assignment_p2.py is the solution for part 2 of assignment

5. Result will be in the `data` folder, with final model saved in `model` folder and there accuracy and other information in `tensorboard` folder (viewed using `tensorboard --port <port> --logdir ./data/tensorboard`)

---

## Pre Computed Results

To see the pre computed results run `tensorboard --port <port> --logdir tensorboard` in project directory.

---

Note: The analysis of the result: `ECE_3195_Course_Project.pdf` and results in `tensorboard` folder.
