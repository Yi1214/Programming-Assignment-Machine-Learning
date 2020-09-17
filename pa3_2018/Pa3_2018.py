# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
  

x = np.linspace(-np.pi,np.pi,140).reshape(140,-1)
y = np.sin(x)

lr = 0.02     #set learning rate


def mean_square_loss(y_pre,y_true):         #define loss 
    loss = np.power(y_pre - y_true, 2).mean()*0.5
    loss_grad = (y_pre - y_true) / y_pre.shape[0]
    return loss , loss_grad           # return loss and loss_grad
    
class ReLU():                     # ReLu layer
    def __init__(self):
        pass
    def forward(self,input):
        '''
        '''
        z = input
        a = np.where(z < 0, 0, z)
        return a
        '''
        '''
        
        # return  *
        
    def backward(self,input,grad_output):    #grad_output: grad from FC layer
        '''
        '''
        z = input
        delt_z = np.where(z <= 0, 0, 1) * grad_output
        return delt_z
        '''
        '''
        # return *
        
        

class FC():
    def __init__(self,input_dim,output_dim):    # initilize weights
        self.W = np.random.randn(input_dim,output_dim)*1e-2
        self.b = np.zeros((1,output_dim))
                       
    def forward(self,input):          
        '''
        '''
        X = input
        z = np.dot(X , self.W) + self.b
        return z
        '''
        '''
        # return *
        
        
    
    def backward(self,input,grad_out):       # backpropagation , update weights in this step
        '''
        '''
        X = input
        
        delt_W = 1/140 * np.dot(X.T, grad_out)
        delt_b = 1/140 * np.sum(grad_out, axis = 0)
        da = np.dot(grad_out , self.W.T)

        self.W -= lr * delt_W
        self.b -= lr * delt_b
        
        return da
        '''
        '''
        #self.W -= lr * delt_W
        #self.b -= lr * delt_b
        # return *
        



#  bulid the network      
layer1 = FC(1,80)
ac1 = ReLU()
out_layer = FC(80,1)

# count steps and save loss history
loss = 1
step = 0
l= []
while loss >= 1e-4 and step < 15000: # training 
            
    # forward     input x , through the network and get y_pre and loss_grad   
    
    
    '''
    '''
    z1 = layer1.forward(x)
    a = ac1.forward(z1)
    y_pre = out_layer.forward(a)
    [loss, loss_grad] = mean_square_loss(y_pre,y)
    '''
    '''
    
    #backward   # backpropagation , update weights through loss_grad
    
    '''
    '''
    delta_output = 140 * loss_grad
    da = out_layer.backward(a, delta_output)
    dz = ac1.backward(z1, da)
    layer1.backward(x, dz)
    
    
    
    
    '''
    '''
    
        
    step += 1
    l.append(loss)
    
    
    
# after training , plot the results
plt.plot(x,y,c='r',label='true_value')
plt.plot(x,y_pre,c='b',label='predict_value')
plt.legend()
plt.savefig('1.png')
plt.figure()
plt.plot(np.arange(0,len(l)), l )
plt.title('loss history')
    

