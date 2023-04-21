"""
Adapted from
Stanford Course CS231n 
Convolutional Neural Networks for Visual Recognition
"""
import time
start = time.time()

import numpy as np
from Spiral_training_data import *
import matplotlib.pyplot as plt
# Quadratic and sigmoid functions
def delta(z, a, y):
    """Return the error delta from the output layer."""

    return (a-y) * sigmoid_prime(z)

def sigmoid(z):

    """The sigmoid function."""

    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):

    """Derivative of the sigmoid function."""

    return sigmoid(z)*(1-sigmoid(z))


# Number of hidden layer neurons

h = 100 # 400 chiasson

# Initialize parameters randomly

W2 = 0.01*np.random.randn(h,D) 

b2 = np.zeros((h,1))

W3 = 0.01*np.random.randn(K,h) 

b3 = np.zeros((K,1))

# Hyperparameters

eta = 1e-3

lmbda = 0*1e-7

# Create data and one hot y matrices

X = np.transpose(Xt)   # X.shape = (2,300)

n = X.shape[1]         # n = 300

y_onehot = np.zeros((K,N*K))  # y_bin.shape: (3,300)

# y = [0,0,...0,1,1,...,1,2,2,...,2] training labels
# list(range(n)): (0,1,2,...,299)

y_onehot[y,range(n)]+=1   # one hot labels

# y_bin[:,0], y_bin[:,100], y_bin[:,200]

# Gradient descent loop

for i in range(200000):  

# Activations
    

  z2 = np.dot(W2,X) + b2  # z2.shape: (100,300)

  a2 = sigmoid(z2)        # a2.shape: (100,300)

  z3 = np.dot(W3,a2) + b3 # z3.shape: (3,300)

  a3 = sigmoid(z3)        # a3.shape: (3,300)

# Quadratic cost function

  # cost_y = 0.5*np.sum((y_onehot-a3)**2)/2

################################################################################
# Cross Entropy Function

  cost_y = np.sum(np.nan_to_num(-y_onehot*np.log(a3)-(1-y_onehot)*np.log(1-a3)))
################################################################################

# Regularization 

  cost_W2W3 = 0.5*lmbda*np.sum(W2*W2) + 0.5*lmbda*np.sum(W3*W3)

  avg_cost = cost_y/n + cost_W2W3/n 

# Print average cost every 10000 iterations
  j = i+1
  if j % 10000 == 0:

    print("iteration %d: avg_cost %f" % (j, avg_cost))

# Backprop output layer

  delta3 = delta(z3,a3,y_onehot)   # (a3-y_onehot) * sigmoid_prime(z3)

  dW3 = np.dot(delta3,a2.T)

  db3 = np.sum(delta3, axis=1, keepdims=True) 

# Backprop hidden layer

  delta2 = np.dot(W3.T,delta3)*sigmoid_prime(z2)

  dW2 = np.dot(delta2,X.T)

  db2 = np.sum(delta2, axis=1, keepdims=True)
  

# Regularization gradient contribution

  dW3  += lmbda * W3   

  dW2  += lmbda * W2

  
# Parameter update

  W2 += -eta * dW2

  b2 += -eta * db2

  W3 += -eta * dW3

  b3 += -eta * db3


# Evaluate training set accuracy

z2 = np.dot(W2,X) + b2

a2 = sigmoid(z2)

z3 = np.dot(W3,a2) + b3

a3 = sigmoid(z3)

predicted_class = np.argmax(a3, axis=0)

print('training accuracy: %.4f' % (np.mean(predicted_class == y)))


# Set up plot

g = 0.02

x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1

y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, g),

                     np.arange(y_min, y_max, g))     

print("")
print("xx.shape:", xx.shape)
print("")
print("yy.shape:", yy.shape)
print("")

# Plot results

Xt_grid = np.c_[xx.ravel(), yy.ravel()]

X_grid = np.transpose(Xt_grid)

a2_grid = sigmoid(np.dot(W2,X_grid) + b2)

Z = sigmoid(np.dot(W3,a2_grid) + b3)      

Z = np.argmax(Z, axis=0)

Z = Z.reshape(xx.shape)

fig = plt.figure()

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=1)

plt.scatter(X[0,:], X[1,:], c=y, s=40, marker = ".", cmap=plt.cm.gist_rainbow,edgecolors="k")

plt.xlim(xx.min(), xx.max())
##
plt.ylim(yy.min(), yy.max())
#
#fig.savefig('spiral_net.png')
end = time.time()
print('time needed to run program:', end - start)




