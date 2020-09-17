# -*- coding: utf-8 -*-

# improt python module that will be used in this .py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn  as nn 
import torch.optim as optim

print('linear MSE')

# define a linear regression class 
class linear_regression(nn.Module):       # class inherits nn.Module
    def __init__(self):
        super(linear_regression,self).__init__() # python inherits method
        self.fc = nn.Linear(10,1)                # fully connect layer
        
    def forward(self,xx):              # define module architecture
        xx = self.fc(xx)
        return xx
"""
generate data
y = A * x + noise
y , A , noise are generated randomly
the model is designed to learn x
torch.randn(), torch.rand() is pytorch randomly generate data method
torch.mm() is pytorch matirx multiply method,notice: you can treat a 
n dimention vecter as a n*1 matrix

"""
        
A = torch.randn(10000,10)*10
x = torch.rand(10,1)
noise = torch.randn(10000,1)
y = A.mm(x) + noise

model_linear  = linear_regression() # generate  a  instance
criterion_linear  = nn.MSELoss()  # Mininise mean square loss
optimizer_linear = optim.SGD(model_linear.parameters(),lr = 1e-4)   #optimiser setting     

# train 5000 epochs
for i in range(5000):
    y_pre = model_linear(A)   # input
    loss = criterion_linear(y_pre,y) # calculate loss
    optimizer_linear.zero_grad() # zero grad
    loss.backward()             # backpropagation
    optimizer_linear.step()      # optimize parameters

error =torch.pow( model_linear.fc.weight.t() -  x , 2).sum() #  square error
if error <= 1e-4:
    print('x has been learned!')
    print('* '*20)

"""
using softmax to classify mnist
MNIST database: https://en.wikipedia.org/wiki/MNIST_database
pytorch documentation for details of function used in this .py
https://pytorch.org/docs/stable/index.html

"""
print('sofrmax in mnist')
# define layer
class softmax(nn.Module):
    def __init__(self):
        super(softmax,self).__init__()
        self.fc = nn.Linear(28*28,10)
    def forward(self,x):
        x = x.view(-1,784)
        x = self.fc(x)
        return x

# load data and generate a iterable object
transforms = transforms.ToTensor()
traindata = torchvision.datasets.MNIST(root = '.\\data',
                                       train = True,
                                       download = True,
                                       transform = transforms) 
testdata  = torchvision.datasets.MNIST(root = '.\\data',
                                       train = False,
                                       download = True,
                                       transform = transforms)  

trainloader = torch.utils.data.DataLoader(traindata,
                                          batch_size = 100,
                                          shuffle = True)
  
testloader = torch.utils.data.DataLoader(testdata,
                                         batch_size = 10000,
                                         shuffle = False)  

# samething doing in the linear model
model = softmax()
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(),lr = 1e-4)


epoch = 5    # decide how much epochs to train
for i in range(epoch):
    for data  in trainloader:
        inputs , labels = data
        pre = model(inputs)
        loss = criterion(pre,labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # after training ,see test peformance
    with torch.no_grad():     
        for test in testloader:  
            img , labels = test
            pre = model(img)
            loss = criterion(pre,labels)       
            pre = torch.argmax(pre.data,1)
            correct = torch.tensor((pre ==labels),dtype=torch.float).mean()
            print('epoch {}/{}; test loss: {} ; test accuracy: {}'.format(i+1,epoch,loss,correct))    