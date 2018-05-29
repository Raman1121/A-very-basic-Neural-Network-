import numpy as np 

def sigmoid(x, deriv = False):
    if deriv == True:
        return x*(1 - x)

    return 1/(1 + np.exp(-x))

X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])
 
Y = np.array([[0],
             [1],
             [1],
             [0]])

#Setting the random seed
np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for i in range(60000):
    
    #Define the different layers
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    #Computing the final error
    l2_error = Y - l2

    #Printing the error value at every 10,000th step
    if (i% 10000) == 0:
        print( "Error:" + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, True)

    #Calculating the error of different layers

    l1_error = np.dot(l1.T, l2_delta)
    l1_delta = l1_error * sigmoid(l1, True)

    #Updating the weights
    syn1 = np.dot(l1.T, l2_delta) 
    syn0 = np.dot(l0.T, l1_delta)

    