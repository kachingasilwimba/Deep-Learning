import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2 
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) 
net.large_weight_initializer()
net.SGD(training_data, 100, 10, 0.1, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)

data = {}   # data is a dictionary
data['w2'] = net.weights[0]  # w2 is a np.array
data['w3'] = net.weights[1] 
data['b2'] = net.biases[0]
data['b3'] = net.biases[1]
print('')
print('data.keys()')
print(data.keys())
print('')
import pickle
picklefile = "Network2_MNIST"+".pkl" ##  Name of the file
path = r"/Users/kachingasilwimba/Desktop/Fall2022/Deep_Learning/Assignments/HW3"
file = open(path+picklefile,'wb') ## The flag should be 'wb' for writing.
pickle.dump(data,file)  ## Dump the data.
file.close()    ## ALWAYS close the file handle or else data will corrupt.

#import pickle
#picklefile = "NN_3layer_MNIST"+".pkl" ##  Name of the file
#path =  r"/Users/kachingasilwimba/Desktop/Fall2022/Deep_Learning/Assignments/HW3"
#file = open(path+picklefile,'rb') ## The flag should be 'rb' for reading.
#data = pickle.load(file)          ## Read the file.
#file.close() # Always close the file.
#b2 = data['b2']
#w2 = data['w2']
#b3 = data['b3']
#w3 = data['w3']