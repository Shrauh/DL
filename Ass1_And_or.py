import numpy as np
import matplotlib.pyplot as plt
# step II: create datasets
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])# step III: Initialize weights,bias,learning rate, epochs
weights = np.zeros(2)
bias = 0
lr = 0.1 
epochs = 10
# step IV: Define the Activation Function
def activation(z):
    if z>=0:
        return 1
    return 0
# step V: Training loops
for epoch in range(epochs):
    # loop through all training examples
    for i in range(4):
        # apply forumula 
        z = np.dot(x[i],weights) + bias
        # predict
        y_pred = activation(z)
        # calculate error 
        error = y[i] - y_pred;

        # update the weights and bias
        weights = weights + lr*error*x[i]
        bias = bias + lr*error
    print("Epoch : ",epoch+1, weights,bias)
# Step VI: Testing 
for i in range(4):
    z = np.dot(x[i],weights) + bias
    y_pred = activation(z)
    print(x[i],y_pred)
# Step VII: Visualize
plt.plot(x[:,0],x[:,1],'bo',label='data points')
trial_x = np.linspace(-0.5,1.5,100)
trial_y = -(weights[0]*trial_x+bias)/weights[1]
plt.plot(trial_x,trial_y,'r-',label='decision boundary')
plt.xlabel("X1")
plt.ylabel("Y1")
plt.legend()
plt.grid()
plt.show()
