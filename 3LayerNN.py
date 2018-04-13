import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, num_input, num_hidden, num_output):
        self.input_dim = int(num_input)
        self.output_dim = int(num_output)
        self.hidden_dim = int(num_hidden)
        
        np.random.seed(0)
        self.theta1 = np.random.randn(self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)       
        self.bias1 = np.zeros((1, self.hidden_dim))
        self.theta2 = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.bias2 = np.zeros((1, self.output_dim))
        
    def train(self, X, y, num_epochs, learning_rate, rlambda):
        for epoch in range(0, num_epochs):
            #Forward Propagation
            a1 = X #input
            z1 = np.dot(a1, self.theta1) + self.bias1 #input to hidden layer
            a2 = actfunc(z1) #output from hidden layer
            z2 = np.dot(a2,self.theta2) + self.bias2 #input to output layer

            exp_z = np.exp(z2)
            softmax_scores = exp_z/np.sum(exp_z, axis = 1, keepdims = True)
            
            #backpropagation          
            d3 = np.zeros_like(softmax_scores)
            one_hot_y = np.zeros_like(softmax_scores)
            for i in range(X.shape[0]):
            		one_hot_y[i,int(y[i])] = 1
            		
            #error calculation
            d3 = softmax_scores - one_hot_y# + rlambda*np.sum(self.theta2*self.theta2)
            #d2 = np.dot(d3,self.theta2.T)*(sigmoid(z2)*(1-sigmoid(z2))) #For sigmoid
            d2 = np.dot(d3, self.theta2.T) * (1 - np.power(a2, 2))# + rlambda*np.sum(self.theta1*self.theta1)#For tanh

            #gradient computation for tanh
            ddtheta2 = np.dot(a2.T,d3)
            ddtheta1 = np.dot(X.T,d2)
            ddbias2 = np.sum(d3, axis=0)
            ddbias1 = np.sum(d2, axis=0)

            #parameter updates
            self.theta1 -= learning_rate*ddtheta1 + rlambda*np.sum(self.theta1)
            self.theta2 -= learning_rate*ddtheta2 + rlambda*np.sum(self.theta2)
            self.bias1 -= learning_rate*ddbias1
            self.bias2 -= learning_rate*ddbias2
            
    def predict(self, X):
        a1 = X #input
        z1 = np.dot(a1, self.theta1) + self.bias1 #input to hidden layer
        a2 = actfunc(z1) #output from hidden layer
        z2 = np.dot(a2,self.theta2) + self.bias2 #input to output layer
     
        exp_z = np.exp(z2)
        softmax_scores = exp_z / np.sum(exp_z, axis = 1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        
        return predictions

def actfunc(s):
    #return 1/(1+np.exp(-s)) #Sigmoid Function
    #return np.maximum(s, 0, s) #ReLU Function
    return np.tanh(s) #Tanh Function

def plot_decision_boundary(model, X, y):
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

def main():
    learning_rate = 0.01
    num_epochs = 10000
    hidden_dim = 20
    rlambda = 0.01
    X = np.genfromtxt('DATA/NonLinearX.csv', delimiter = ',')
    y = np.genfromtxt('DATA/NonLineary.csv', delimiter = ',')
    #X = np.genfromtxt('DATA/LinearX.csv', delimiter = ',')
    #y = np.genfromtxt('DATA/Lineary.csv', delimiter = ',')

    #from sklearn.cross_validation import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

    input_dim = np.shape(Xtrain)[1]
    output_dim = np.max(ytrain) + 1

    neuralnet = NeuralNetwork(input_dim, hidden_dim, output_dim)
    neuralnet.train(Xtrain, ytrain, num_epochs, learning_rate, rlambda)

    acc = 0
    y_pred = neuralnet.predict(Xtest)
    con_mat = np.zeros((int(output_dim), int(output_dim)))
    for i in range(len(y_pred)):
        con_mat[y_pred[i], int(ytest[i])] += 1
        if ytest[i] == y_pred[i]:
            acc += 1
    acc = acc/len(y_pred)

    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr) #http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
    plt.show()
    plot_decision_boundary(neuralnet, X, y)

    print ('LEARNING RATE: ', learning_rate)
    print ('# OF EPOCHS: ', num_epochs)
    print ('LAMBDA: ' , rlambda)
    print ('# NODES IN HIDDEN LAYER: ', hidden_dim)
    print ('ACCURACY: ', acc)
    print ('CONFUSION MATRIX: \n', con_mat)
    return neuralnet

main()