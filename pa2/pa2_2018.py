# -*- coding: utf-8 -*-


import numpy as np
from numpy import linalg

# load vowel data stored in npy
'''
NOTICE:
labels of y are: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
'''
x_test = np.load('x_test.npy')
print('x_test\'s shape: {}'.format(x_test.shape))
y_test = np.load('y_test.npy')
print('y_test\'s shape: {}'.format(y_test.shape))
x_train = np.load('x_train.npy')
print('x_train\'s shape: {}'.format(x_train.shape))
y_train = np.load('y_train.npy')
print('y_train\'s shape: {}'.format(y_train.shape))

pi = 3.1415926  # value of pi

'''
x : m * n matrix
u : 1 * n vector
sigma ： n * n mtrix
result : 1 * m vector
the function accept x,u ,sigma as parameters and return corresponding probability of N(u,sigma)
you can choise use it to claculate probability if you understand what this function is doing 
your choice!
'''
def density(x,u,sigma):
    n = x.shape[1]
    buff = -0.5*((x-u).dot(linalg.inv(sigma)).dot((x-u).transpose()))
    exp = np.exp(buff)
    C = 1 / np.sqrt(np.power(2*pi,n)*linalg.det(sigma))
    result = np.diag(C*exp)
    return result


'''
class GDA
self.X : training data X
self.Y : training label Y
self.is_linear : True for LDA ,False for QDA ,default True
please make youself konw basic konwledge about python class programming

tips : function you may use
np.histogram(bins = self.n_class)
np.reshape()
np.transpose()
np.dot()
np.argmax()

'''
class GDA():
    def __init__(self, X, Y, is_linear = True):
        self.X = X
        self.Y = Y
        self.is_linear =  is_linear
        self.n_class = len(np.unique(y_train)) # number of class , 11 in this problem 
        self.n_feature = self.X.shape[1]       # feature dimention , 10 in this problem
        
        self.pro = np.zeros(self.n_class)     # variable stores the probability of each class
        self.mean = np.zeros((self.n_class,self.n_feature)) #variable store the mean of each class
        self.sigma = np.zeros((self.n_class,self.n_feature,self.n_feature)) # variable store the covariance of each class
        
        
    def calculate_pro(self):  
        #calculate the probability of each class and store them in  self.pro
        '''
        '''
        m = self.X.shape[0]
        self.pro = 1 / m * np.histogram(self.Y, bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12])[0] 

        '''
        '''
        #self.pro = 
        return self.pro
    
    def claculate_mean(self):
        #calculate the mean of each class and store them in  self.mean
        '''
        '''
        m = self.X.shape[0]
        flag = np.zeros(self.n_class)
        
        for i in range (m):
            for j in range (self.n_class):
                if self.Y[i] == j + 1:
                    self.mean[j,:] = self.mean[j,:] + self.X[i,:]
                    flag[j] = flag[j] + 1
        for l in range (self.n_class):
            self.mean[l,:] = self.mean[l,:] / flag[l]
        '''
        '''
        #self.mean = 
        
        
            
    def claculate_sigma(self):
        #calculate the covariance of each class and store them in  self.sigma
        m = self.X.shape[0]
        if self.is_linear == True:
            for i in range (m):
                #for j in range (self.n_class):
                    #if self.Y[i] == (j + 1):
                self.sigma[0,:,:] = self.sigma[0,:,:] + np.dot((self.X[i,:] - self.mean[self.Y[i]-1,:]).transpose(), (self.X[i,:] - self.mean[self.Y[i]-1,:]))    
            self.sigma[0,:,:] = self.sigma[0,:,:] / m
            for i in range (self.n_class):
                self.sigma[i,:,:] = self.sigma[0,:,:]
        
        else:
            flag = np.zeros(self.n_class)
            for i in range (m):
                for j in range (self.n_class):
                    if self.Y[i] == j + 1:
                        #self.sigma[j,:,:] = self.sigma[j,:,:] + np.dot((self.X[i,:] - self.mean[j,:]).transpose(), (self.X[i,:] - self.mean[j,:]))
                        self.sigma[self.Y[i]-1,:,:] = self.sigma[self.Y[i]-1,:,:] + np.dot((self.X[i,:] - self.mean[self.Y[i]-1,:]).transpose(), (self.X[i,:] - self.mean[self.Y[i]-1,:]))
                        flag[j] = flag[j] + 1

            for l in range (self.n_class):
                self.sigma[l,:,:] = self.sigma[l,:,:] / flag[l]                   
        # self.sigma = 
        
     
    def classify(self, x_test):
        # after training , use the model to classify x_test, return y_pre
        n = self.X.shape[1]
        pred = np.zeros((y_test.shape[0],self.n_class))
        y_pre = np.zeros((y_test.shape[0], 1))
        for i in range (x_test.shape[0]):
            for j in range (self.n_class):
                buff = -0.5*((x_test[i,:]-self.mean[j,:]).dot(linalg.inv(self.sigma[j,:,:])).dot((x_test[i,:]-self.mean[j,:]).transpose()))
                exp = np.exp(buff)
                C = 1 / np.sqrt(np.power(2*pi,n)*linalg.det(self.sigma[j,:,:]))
                #pred[i, j] = density(x_test[i,:], self.mean[j,:] , self.sigma[j,:,:]) * self.pro[j]
                pred[i, j] = C * exp * self.pro[j]
            y_pre[i, 0] = np.argmax(pred[i]) + 1
        # y_pre = 
        return y_pre
        
        
LDA = GDA(x_train,y_train) # generate the LDA model
LDA.calculate_pro()        # calculate parameters
LDA.claculate_mean()
LDA.claculate_sigma()
LDA.claculate_sigma()
y_pre = LDA.classify(x_test) # do classify after training
LDA_acc = (y_pre == y_test).mean()
print ('accuracy of LDA is:{:.2f}'.format(LDA_acc))   
    

QDA = GDA(x_train,y_train,is_linear=False) # generate the QDA model
QDA.calculate_pro()                     # calculate parameters
QDA.claculate_mean()
QDA.claculate_sigma()
QDA.claculate_mean()
QDA.claculate_sigma()
y_pre = QDA.classify(x_test)          # do classify after training
QDA_acc = (y_pre == y_test).mean()
print ('accuracy of QDA is:{:.2f}'.format(QDA_acc))
