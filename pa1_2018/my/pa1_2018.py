# -*- coding: utf-8 -*-


# import module 

import torch
import torchvision 
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

'''
part1: linear regression
A :10000 * 10 matrix
x : 10 * 1 vector
noise = 10000 * 1 vector
y = A * x + noise 
y : 10000 * 1 vector
x_learn ：the x learned by using the gradient descent algorithm 
          when minimizing the mean square error
hint:functions you may use
torch.Tensor.mm(): matrix miltiply, treat n dimention voctor as n * 1 matrix
torch.Tensor.t(): matrix transposed
pytorch documentation: https://pytorch.org/docs/stable/index.html

'''

A = torch.randn(10000,10)*10 # randomly generate A
x = torch.rand(10,1)         # randomly generate x
noise = torch.randn(10000,1) #randomly generate noise
y = A.mm(x) + noise          #calculate y
lr = 0.0001                  # set learnning rate   
x_learn = torch.rand(10,1)   # initialise x_learn
iteration = 10000            # set number of iterations
for i in range(iteration):
    # update x_learn
    '''
    your code begin
    '''
    x = x_learn                # update x
    H = torch.Tensor.mm(A, x)       # hypothesis
    x_learn = x - lr / 10000 * A.t().mm(H - y)
    
    '''
    your code end
    '''
error = torch.pow( x_learn -  x , 2).sum() 
if error <= 1e-4:
    print('part1 is done')
else:
    print('not yet!')    


print('* '*20)

'''
part2: softmax regression in mnist

'''
# laod minist and creat iterable dataloader, do not change this part
transforms = transforms.ToTensor()

traindata = torchvision.datasets.MNIST(root = './data',
                                       train = True,
                                       download = True,
                                       transform = transforms) 
testdata = torchvision.datasets.MNIST(root = './data',
                                      train = False,
                                      download = True,
                                      transform = transforms)  

trainloader = torch.utils.data.DataLoader(traindata,
                                          batch_size = 60000,
                                          shuffle = True)
  
testloader = torch.utils.data.DataLoader(testdata,
                                         batch_size = 10000,
                                         shuffle = False)  

# loading  data  finishs

'''
each image's shape (channel,imagesize,imagesize),which is (1,28,28)
it has to be reshape to a 784 dimention vector to do softmax regression
so we have to call torch.tensor.view()
'''
print('show image size and image....')
img_show = traindata[0][0]
print(img_show.size())
plt.imshow(img_show.reshape(28,28),cmap = 'gray')
plt.show()

'''
hint: functions you may use
torch.tensor().mm()
torch.tensor().t()
torch.exp()
torch.tensor.sum()
torch.tensor.view()
pytorch documentation : https://pytorch.org/docs/stable/index.html

you may use code below to generate a suitable matrix

C = torch.tensor(range(10))
d = torch.tensor(labels==C,dtype = torch.float)

'''

softmax_weights = torch.randn(784,10)*0.005 # initialise weights
lr = 0.2      # set learning rate
epoch = 10      # set trainning epoch
grad = 0
for i in range(epoch):
    for data in trainloader:
        imgs , labels = data      # load data
        imgs = imgs.view(-1,784)  # reshape to right size
        labels = labels.view(-1,1) # reshape to right size,
        
        #Calculate grad(gradient):
        '''
        your code begin
        '''
        
        h = torch.exp(imgs.mm(softmax_weights))      
        sum_h = h.sum(1)                           
        
        for j in range(60000):
            h[j,:] = h[j,:] / sum_h[j]            # Normalization
        
        C = torch.tensor(range(10))
        d = torch.tensor(labels == C, dtype = torch.float)   
        
        # You need to calculate grad here
        
        grad = 1 / 60000 * torch.Tensor.t(imgs).mm(h - d)
        
        '''
        your code end
        '''
        # Update softmax_weights here:
        softmax_weights = softmax_weights - lr * grad
        
        # testdata performance
    for data in testloader:
        imgs ,labels = data
        imgs = imgs.view(-1,784)
        exp = torch.exp(imgs.mm(softmax_weights))
        pre = torch.argmax(exp,1)
        correct = torch.tensor((pre ==labels),dtype=torch.float).mean()
        print('epoch {}/{} ; test accuracy: {}'.format(i+1,epoch,correct))
