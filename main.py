import numpy as np
import matplotlib.pyplot as plt 
import math
import pandas as pd


def sigmoid(h):
    return 1/(1+np.exp(-h))

def sigmoid_der(h):
    return h*(1-h)


X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y =np.array([[0],[1],[1],[0]])

df = pd.DataFrame(np.concatenate((X,Y),axis=1), columns = ['X1','X2','Y'])

W =np.random.uniform(size=(2,1)) 
B =np.random.uniform(size=(1,1))

times = 96
Epoch=np.arange(0,times)
Error=[0]*times

for i in range(0,times):
    
    #Forward propagation, a1 dot product to the Weights and then add the Biases. Then use the sigmoid function
    a1 = X    
    z2 = np.dot(a1, W) + B    
    a2 = sigmoid(z2)
    
    #Calculate the error actual - trained model 
    error = (Y - a2)
    #differentiate the error (Gradient descent)
    output = error * sigmoid_der(a2)
    
    #Back propagation, dot product transpoed a1 matrix with output, and thenadd the weights and the biases again. The forloop ensures that this neural network repeats the process. 
    update = np.dot(a1.T, output)  
    W = W + update    
    B = B + sum(output) 
    
    #Error calcaulted by actual - trained result
    Error[i]=error.mean()
    
    print("The Error is: ", Error[i]," ",  "trial:",  i)    # storing mean error
    

for value in output:
  print(np.round(value))
    
