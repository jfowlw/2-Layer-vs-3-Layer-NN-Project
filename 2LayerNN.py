import numpy as np    # numpy is the fundamental package for scientific computing with Python, such linear algebra, array...
import matplotlib.pyplot as plt      # matplotlib is a Python 2D plotting library which produces publication quality figures.class LogisticRegression:
from sklearn.model_selection import train_test_split
    
class NeuralNetwork:
    """
    This lab implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta = np.random.randn(int(input_dim), int(output_dim)) / np.sqrt(int(input_dim))       
        self.bias = np.zeros((1, int(output_dim)))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        num_examples = np.shape(X)[0]
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        one_hot_y = np.zeros((num_examples,np.max(y)+1))
        logloss = np.zeros((num_examples,))        
        for i in range(np.shape(X)[0]):
            one_hot_y[i,y[i]] = 1
            logloss[i] = -np.sum(np.log(softmax_scores[i,:]) * one_hot_y[i,:])
        data_loss = np.sum(logloss)
        return 1./num_examples * data_loss
    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / (exp_z + 1)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    # TODO: implement logistic regression using gradient descent 
    #--------------------------------------------------------------------------
    def fit(self,X,y,num_epochs,alpha):
        #Learns model parameters to fit the data.
        for epoch in range(0, num_epochs):

            # Forward propagation
            z = np.dot(X,self.theta) + self.bias
            exp_z = np.exp(z)
            #softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            softmax_scores = exp_z / (exp_z + 1)
        
            # Backpropagation
            beta = np.zeros_like(softmax_scores)
            #print (beta)
            one_hot_y = np.zeros_like(softmax_scores)
            for i in range(X.shape[0]):
                one_hot_y[i,int(y[i])] = 1
            beta = softmax_scores - one_hot_y
            # Compute gradients of model parameters
            dtheta = np.dot(X.T,beta)
            dbias = np.sum(beta, axis=0)
    
            # Gradient descent parameter update
            self.theta -= alpha * dtheta
            self.bias -= alpha * dbias   
        return 0

def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    
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
    #X = np.genfromtxt('DATA/NonLinearX.csv', delimiter = ',')
    #y = np.genfromtxt('DATA/NonLineary.csv', delimiter = ',')
    X = np.genfromtxt('DATA/LinearX.csv', delimiter = ',')
    y = np.genfromtxt('DATA/Lineary.csv', delimiter = ',')

    #from sklearn.cross_validation import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

    input_dim = np.shape(Xtrain)[1]
    output_dim = np.max(ytrain) + 1

    neuralnet = NeuralNetwork(input_dim, output_dim)
    neuralnet.fit(Xtrain, ytrain, num_epochs, learning_rate)

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
    print ('ACCURACY: ', acc)
    print ('CONFUSION MATRIX: \n', con_mat)
    return neuralnet

main()