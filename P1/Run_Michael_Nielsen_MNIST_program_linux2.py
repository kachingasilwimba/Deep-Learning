import sys
sys.path.insert(0, '/home/ksilwimba/')
import MNIST_Loader
training_data, validation_data, test_data = \
MNIST_Loader.load_data_wrapper()
import network_leakyReLU
net = network_leakyReLU.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 0.003, test_data=test_data)
