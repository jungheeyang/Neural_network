import sys
import numpy as np
import random

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork:
    '''
    Initialise Neural Network base class.

    Args:
    NInput (int): number of neurons in input layer
    NHidden (int): number of neurons in input layer
    NOutput (int): number of neurons in input layer (must be 10)
    '''
    def __init__(self, NInput, NHidden, NOutput, epochs = 30, learning_rate = 3, mini_batch_size = 20):
        self.NInput = NInput
        self.NHidden = NHidden
        self.NOutput = NOutput
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.correct = 0

        # Random weights and zero biases
        weights = [np.random.randn(NHidden, NInput)*0.01, np.random.randn(NOutput, NHidden)*0.01] 
        self.weights = weights  # self.weights[0]: hidden layer, self.weights[1]: output layer

        biases = [np.zeros((NHidden, 1)), np.zeros((NOutput, 1))]
        self.biases = biases # self.biases[0]: hidden layer, self.biases[1]: output layer

        
    def forward_propagation(self, X, Y=None):
        '''
        Perform forward pass through network.

        Args:
        X (np.Array): input data of size (size of the input layer, m)

        Returns:
        a2 (np.Array): the sigmoid output of output layer
        '''
        X = np.reshape(X, (len(X),1))
        
        # First activation - hidden layer
        self.z1 = np.dot(self.weights[0], X) + self.biases[0] #(64,1)
        self.a1 = sigmoid(self.z1) # (64,1)

        # Second activation - output layer
        self.z2 = np.dot(self.weights[1], self.a1) + self.biases[1] # (10,1)
        self.a2 = sigmoid(self.z2) # (10,1)
    
        return self.a2


    def compute_cost(self, a2, Y, cross_entropy = False):
        '''
        Compute cost of output using quadratic or cross entrophy cost function.

        Args:
        a2 (np.Array): output of second activation 
        Y( flaot): a true label of the image

        Returns:
        cost (np.Array): quadratic or cross entrophy cost given a2 and Y
        '''
        temp = np.array([0]*10).reshape((10,1))
        temp[int(Y)] = 1
        Y = temp
        m = Y.shape[1]
        
        if not cross_entropy:
            # Quadratic cost
            loss = (1 / 2) * (a2 - Y)
            cost = np.sum(loss) / (2 * m)

            return cost
        else:
            # Cross-entropy cost
            logprobs = np.multiply(np.log(a2), Y) + np.multiply(np.log(1-a2), 1-Y) 
            cost = np.sum(logprobs) / -m
            cost = float(np.squeeze(cost))

            return cost


    def backward_propagation(self, X, Y):
        '''
        Perform backward propagation through network.

        Args:
        a2 (np.Array): output of second activation 
        Y( flaot): a true label of the image
        cross_entrophy (bool): True - calculate cost using cross entrophy fucntion
                               False - calculate cost using quadratic fucntion
        '''
        temp = np.array([0]*10).reshape((10,1))
        temp[int(Y)] = 1
        Y = temp
        
        X = np.reshape(X, (len(X), 1))
        
        # Output layer
        dz2 = self.a2 - Y # (10, 1)
        self.dw2 += (np.dot(dz2, self.a1.T)) # (10, 64)
        self.db2 += np.sum(dz2, axis=1, keepdims=True) # (10, 1)

        # Hidden layer    
        dz1 = np.dot(self.weights[1].T, dz2) * (1 - np.power(self.a1, 2)) # dz1 (64, 1)
        self.dw1 += np.dot(dz1, X.T)
        self.db1 += np.sum(dz1, axis=1, keepdims=True)


    def update_weights_biases(self, m):
        '''
        Update weights and biases.
        
        Argv:
        m (int): batch size
        '''
        # Calcualte the averages
        self.dw2 /= m
        self.db2 /= m
        self.dw1 /= m
        self.db1 /= m
        
        # Hidden layer
        self.weights[0] -= self.dw1 * self.learning_rate
        self.biases[0] -= self.db1 * self.learning_rate

        # Output layer
        self.weights[1] -= self.dw2 * self.learning_rate
        self.biases[1] -= self.db2 * self.learning_rate

    
    def train(self, X, Y, cross_entropy=False): 
        '''
        Train the network using forward propagation and backward propagation by batches

        Argv:
        X (np.Array): features
        Y (np.Array): labels 
        '''
        # Epochs
        for i in range(self.epochs):
            # Batches
            batches = []
            for idx in range(0, len(X), self.mini_batch_size):
                x = X[idx:idx+self.mini_batch_size]
                y = Y[idx:idx+self.mini_batch_size]

                # Gradients set to zero for every batch
                self.dw2 = 0
                self.db2 = 0
                self.dw1 = 0
                self.db1 = 0
                
                # Process forward propagation and backward propagation of current batch
                for a, b in zip(x, y):
                    a2 = self.forward_propagation(a)
                    cost = self.compute_cost(a2, b, cross_entropy)
                    gradient = self.backward_propagation(a, b)
                    
                # Update weights and biases in the end of each batch
                self.update_weights_biases(self.mini_batch_size)
                

    def test(self, testX, testY):
        '''
        Test the network with given test set

        Argv:
        testX (np.Array): features
        testY (np.Array): labels 

        Return:
        str (correct / size of test set)
        '''
        correct = 0
        for idx, x in enumerate(testX):
            output = self.forward_propagation(x)
            if (np.argmax(output) == int(testY[idx])):
                correct += 1
        return f"{correct}/{len(testX)}"

    def predict(self, X):
        output = self.forward_propagation(X)
        for i in output:
            print((np.argmax(output))


if __name__ == "__main__":
    
    n_samples = 50000
    nInput = int(sys.argv[1])    # 784
    nHidden = int(sys.argv[2])   # 64
    nOutput = int(sys.argv[3])   # 10

    print("\n\nLoading files... Please wait.")
    trainX = np.loadtxt(sys.argv[4], delimiter=',') # "TrainDigitX.csv.gz"
    trainY = np.loadtxt(sys.argv[5], delimiter=',') # TrainDigitY.csv.gz

    testX = np.loadtxt(sys.argv[6], delimiter=',')  # TestDigitX.csv.gz
    testY = np.loadtxt(sys.argv[7], delimiter=',')  # TestDigitY.csv.gz
    print("Loading complete.")

    # predictY = np.loadtxt("TestDigitX2.csv.gz", delimiter=',')
    
    default = input("Process with defalut ephochs(30), mini batch size(20) and learning rate(3.0)? Y/N ")
    if default == "y" or default == "Y":
        # Default
        n_epochs = 30
        learning_rate = 3
        mini_batch_size = 20
    else:
        print("Please enter the following...")
        n_epochs = int(input("Epochs: "))
        learning_rate = float(input("Learning rate (between 0 and 1): "))
        mini_batch_size = int(input("Mini batch size: "))
    
    cost_func = int(input("""Choose the cost function.
    1. Mean Squared Error (quadratic) 
    2. Cross-entropy 
"""))
    
    if cost_func == 1:
        cross_entropy = False
    else:
        cross_entropy = True


    nn = NeuralNetwork(nInput, nHidden, nOutput, n_epochs, learning_rate, mini_batch_size)
    print("\n\nTraining commencing... please wait.")
    nn.train(trainX, trainY, cross_entropy)
    print("Training complete.")

    print("\nTesting commencing... please wait.")
    result = nn.test(testX, testY)
    print(f"Test complete. Accuracy: {result}")
    
