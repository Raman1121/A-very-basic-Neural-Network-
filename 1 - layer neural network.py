import numpy as np

#Sigmoid Function
def sigmoid(x, derivative = False):
    if derivative == True:
        return x*(1 - x)
    
    return 1/(1+np.exp(-x))

#Input Dataset 
X = np.array([  [0,0,1],    
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

#Output Dataset
Y = np.array([[0,0,1,1]]).T     # .T for Transpose

#Random Seed Value
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

# Start the network training

for i in range(10000):

    #Forward Propagation
    l0 = X                          #l0 is the first layer that will take in the input dataset

    l1 = sigmoid(np.dot(l0, syn0))  #l1 is the final layer which is basically passing the 
                                    #dot product of layer l0 and weights through the activation function i.e. sigmoid function

    #compute the error in prediction value

    error = Y - l1

    #multiply the error by slope of the sigmoid function at the error.
    # This is done by calculating the derivative at the point of error. Setting derivative var = True

    error_delta = error*sigmoid(l1, True)                                

    #Update the weights accordingly
    syn0 += np.dot(l0.T,error_delta)

    print("Output after training")
    print(l1)





