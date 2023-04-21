import MNIST_Loader
training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()
import network2 
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) 
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, evaluation_data=validation_data, lmbda=5,
        monitor_evaluation_accuracy=True, 
        monitor_training_cost=True)
net.save('MNIST_CrossEntropy.json')
#load('MNIST_CrossEntropy.json')
print()
print('net.weights[0].shape =', net.weights[0].shape)
print('net.weights[1].shape =', net.weights[1].shape)
print('net.biases[0].shape =', net.biases[0].shape)
print('net.biases[1].shape =', net.biases[1].shape)
print()
print('net.cost =', net.cost)
print()
print('Accuracy on Test Data = ', net.accuracy(test_data, convert=False))
print()
