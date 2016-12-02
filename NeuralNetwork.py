import numpy as np
import random

'''
Neural network capable of both entropy loss and squared loss
'''
class NeuralNetwork(object):
    
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, loss_func_string):
    
        # Assign neural network loss and activiation functions
        if loss_func_string == "entropy":
            self.cost = self.entropyLoss
            self.costPrime = self.entropyLossPrime
        else:
            self.cost = self.costSquaredLoss
            self.costPrime = self.costSquaredLossPrime
        # In the future these can be user specified
        self.hiddenLayerActFunc = np.tanh
        self.hiddenLayerActFuncPrime = self.tanhPrime 
        self.outputLayerActFunc = self.sigmoid
        self.outputLayerActFuncPrime = self.sigmoidPrime
            
        # Layer dimensions
        self.inputLayerSize=inputLayerSize # plus bias
        self.hiddenLayerSize=hiddenLayerSize # plus bias
        self.outputLayerSize=outputLayerSize
        
        # In the future these can be user specified
        self.eta = None
        np.random.seed(324)
        self.w1=0.01*np.random.randn(self.inputLayerSize+1, self.hiddenLayerSize+1)
        self.w2=0.01*np.random.randn(self.hiddenLayerSize+1, self.outputLayerSize)
        
        
        # Document the training progress
        self.training_accuracy = []
        self.validation_accuracy = []
        self.trainingTime = []
        
        # Current iteration variables
        self.z1 = None
        self.y1 = None
        self.z2 = None
        self.h = None # Y before one hot encoding
        self.y = None # One hot encoding 
        self.dEdz1 = None
        self.dEdz2 = None
        self.Ytrain = None # Actual digits
        self.Ytrain_encoded = None # One hot encoding
        self.Xtrain = None # data

    # Squared loss
    def costSquaredLoss(self):
        J=0.5*np.sum(np.square(self.y-self.h))
        return J
    
    # Squared loss derivative
    def costSquaredLossPrime(self):
        dJdh = -(self.y-self.h)
        return dJdh
    
    # Entropy loss
    def entropyLoss(self):
        # Need to threshold to prevent division by zero log of zero
        h = copy.copy(self.h) # Threshold h to not modify self.h
        h[h < 1E-5] = 1E-5
        one_minus_h = 1-h
        one_minus_h[one_minus_h < 1E-5] = 1E-5
        J = - np.sum(np.multiply(self.y,np.log(h))+np.multiply((1-self.y),np.log(one_minus_h))) # * is element-wise in python as well, but np.multiply is used here to be explicit         
        return J
    
    # Entropy derivative
    def entropyLossPrime(self):
        # Need to threshold to prevent division by zero log of zero
        h = copy.copy(self.h) # Threshold h to not modify self.h
        h[h < 1E-5] = 1E-5
        one_minus_h = 1-self.h 
        one_minus_h[one_minus_h < 1E-5] = 1E-5
        dJdh = -(self.y/h + (1-self.y)*(-1/(one_minus_h)))
        return dJdh
     
    # Activication function
    def sigmoid(self, z):
        sig = 1/(1+np.exp(-z))
        return sig
    
    # Activation function derivative
    def sigmoidPrime(self, z):
        sigP = np.divide(np.exp(-z),np.square(1+np.exp(-z)))
        return sigP
    
    # Activication function derivative
    def tanhPrime(self, z):
        tanhP = np.tanh(z)
        tanhP = 1-np.square(np.tanh(z))
        return tanhP
    
    # Forward propogation
    def propagateForward(self, X):
        self.w1[:, -1] = 0
        self.w1[-1,-1] = 1
        #X = np.append(X, np.ones((len(X), 1)), axis=1) # Add a 1 for the bias
        self.z1 = np.dot(X, self.w1)
        self.y1 = self.hiddenLayerActFunc(self.z1)
        #self.y1_with_bias = np.append(self.y1, np.ones((len(self.y1), 1)), axis=1) # Add a 1 for the bias
        #self.z2 = np.dot(self.y1_with_bias, self.w2)
        self.z2 = np.dot(self.y1, self.w2)
        self.h = self.outputLayerActFunc(self.z2)
        return 
    
    # Back propagation
    # X = Xtrain, Y = Ytrain = actual digit
    def propagateBackward(self, X, Y):
        # One hot encoding
        self.y = np.zeros((len(Y), 10))
        for i in xrange(0, len(Y)):
            self.y[i, int(Y[i])] = 1
        # dE/dz2 = dy_2/dz_2 * dE/dy_2
        self.dEdz2 = np.multiply(self.outputLayerActFuncPrime(self.z2), self.costPrime())
        # Calculate updates
        self.dEdw2 = np.dot(self.y1.T, self.dEdz2)

        # Don't back propagate the bias
        # dE/dy_1 = dz_2/dy_1 * dEdz_2
        self.dEdy1 = np.dot( self.dEdz2, self.w2.T)
        self.dEdz1 = np.multiply(self.tanhPrime(self.z1), self.dEdy1)
         # Calculate updates
        self.dEdw1 = np.dot(X.T, self.dEdz1)
        return 
    
    # Train the neural network
    def train(self, Xtrain, Ytrain, Xvalid, Yvalid, numiters, num_per_batch):
        start_time = timeit.default_timer()
        training_accuracy_of_last_it = 0.0
        
        # Initial learning rate
        eta = 0.01
        for it_i in xrange(0, numiters):
            # Randomly select data
            sample_ints = random.sample(range(0, Xtrain.shape[0]), num_per_batch)
            X = Xtrain[sample_ints]
            Y = Ytrain[sample_ints]
            self.propagateForward(X)
            self.propagateBackward(X, Y)
            self.y = np.zeros((len(Y), 10))
            for i in xrange(0, len(Y)):
                self.y[i, int(Y[i])] = 1
            

            # Gradient descient updates
            self.w1 = self.w1 - eta * self.dEdw1
            self.w2 = self.w2 - eta * self.dEdw2
            self.w1[:,-1] = 0
            self.w1[-1, -1] = 1
            
            # Calculate accuracy rates and update the learning rate
            if (it_i % 10000) == 0:
                # Calculate the training time
                end_time = timeit.default_timer()
                self.trainingTime.append(end_time - start_time)
                eta = 0.9*eta
                current_training_accuracy = self.calculateAccuracyRate(Xtrain, Ytrain)
                current_validation_accuracy = self.calculateAccuracyRate(Xvalid, Yvalid)
                self.training_accuracy.append(current_training_accuracy)
                self.validation_accuracy.append(current_validation_accuracy)
                print "Iteration number: \t \t", it_i
                print "Time elapsed: \t \t \t", self.trainingTime[-1]
                print "Current training accuracy: \t", current_training_accuracy
                print "Current Validation accuracy: \t", current_validation_accuracy
                training_accuracy_improvement = abs(training_accuracy_of_last_it - current_training_accuracy)
                if (training_accuracy_improvement < 1E-7):
                    return
                training_accuracy_of_last_it =  current_training_accuracy 
        return
    
    # Calculate accuracy
    def calculateAccuracyRate(self, X, Y):
        y_predicted = self.predict(X)
        num_correct = sum(y_predicted == Y)
        accuracy_rate = num_correct/float(Y.shape[0])
        return accuracy_rate
    
    # Used for prediction
    def predict(self, X):
        self.propagateForward(X)
        y_predict = np.zeros((X.shape[0]))
        for i in xrange(0, self.h.shape[0]):
            y_predict[i] = np.argmax(self.h[i,:])
        return y_predict
