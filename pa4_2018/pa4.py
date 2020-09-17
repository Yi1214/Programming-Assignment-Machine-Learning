# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

num = np.load('num.npy')

'''
your codes
'''
#compute probability & B matrix
px = np.sum(num, axis = 1, keepdims = True) / np.sum(np.sum(num, axis = 0))
py = np.sum(num, axis = 0, keepdims = True) / np.sum(np.sum(num, axis = 0))
pxy = num / np.sum(np.sum(num, axis = 0))
B = pxy / np.matmul(np.sqrt(px), np.sqrt(py))

#Initialize gy
gy = np.sum(num, axis = 0, keepdims = True) / np.sum(np.sum(num, axis = 0))
#gy = (gy - np.matmul(gy, py.T)) / np.sqrt(np.matmul(gy * gy, py.T)) #regularize

# compute fx,gy
fx = np.matmul(num, gy.T) / np.sum(num, axis = 1, keepdims = True) 

gy = np.matmul(fx.T, num) / np.sum(num, axis = 0, keepdims = True)

gy = (gy - np.matmul(gy, py.T)) / np.sqrt(np.matmul(gy * gy, py.T)) #regularize



Exy_1 = np.matmul(np.matmul(fx.T, pxy), gy.T)
Exy_2 = 0

# Repeat until Exy stops to increase
while (Exy_1 - Exy_2) >= 0 :
    Exy_2 = Exy_1
    fx = np.matmul(num, gy.T) / np.sum(num, axis = 1, keepdims = True)
    gy = np.matmul(fx.T, num) / np.sum(num, axis = 0, keepdims = True)
    gy = (gy - np.matmul(gy, py.T)) / np.sqrt(np.matmul(gy * gy, py.T))
    Exy_1 = np.matmul(np.matmul(fx.T, pxy), gy.T)

x_vector = np.sqrt(px) * fx
y_vector = (np.sqrt(py) * gy).T
second_sv = np.sqrt(Exy_1[0,0])
#second_sv = np.sqrt((np.matmul(np.matmul(B.T, B), y_vector)) / y_vector)[0,0]

gy = gy.T

# gy = 
#second_sv = 
'''
'''
print('second_sv : {}'.format(second_sv))    
plt.plot(np.arange(15), gy, c = 'r')
plt.xlabel('y')
plt.ylabel('gy')

