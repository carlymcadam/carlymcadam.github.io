import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        if self.w is None:
            self.w = torch.rand((X.size(1)))
        return X @ self.w 

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        return (self.score(X) > 0).float() 

class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
        y_ = 2*y - 1
        """

        y_ = 2 * y - 1  
        return (self.score(X) * y_ <= 0).float().mean()

    def grad(self, X, y):
        pass 

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        pass

class LogisticRegression(LinearModel):
    def loss(self, X, y):
        """
        Compute the logistic loss (empirical risk) for binary classification.
        """
        if self.w is None:
            self.score(X)
        
        y_ = 2 * y - 1  # convert to {-1, 1}
        s = self.score(X)  
        loss = torch.log(1 + torch.exp(-y_ * s)).mean()
        return loss

    def grad(self, X, y):
        """
        Compute the gradient of the logistic loss function. This function computes the 
        gradient of the empirical risk used for binary classification using logistic 
        regression.

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() = (n,). The possible labels 
            for y are {0, 1}.
        
        RETURNS:
            grad, torch.Tensor: the gradient vector. Shape is (p,).
        """
        if self.w is None:
            self.score(X)  

        y_ = 2 * y - 1  # convert to {-1, 1}
        s = self.score(X) 
        v = y_ * torch.sigmoid(-y_ * s)  
        
        
        grad = - (X.T @ v) / X.size(0) 
        return grad
    
class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model
        self.prev_w = None  # To store w_{k-1}

    def step(self, X, y, alpha, beta):
        """
        Perform one step of gradient descent with momentum. This updates the model 
        parameters using the gradient of the loss function, scaled by a learning rate 
        alpha, and adds a momentum term scaled by beta to smooth out updates.

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() = (n,). The possible labels 
            for y are {0, 1}.

            alpha, float: the learning rate. Determines the step size for the 
            gradient descent update.

            beta, float: the momentum coefficient. Determines how much of the 
            previous update direction is used in the current step.
        """
        # initialize w 
        if self.model.w is None:
            self.model.score(X)

        # get gradient of loss at current w
        grad = self.model.grad(X, y)

        # no momentum if first step
        if self.prev_w is None:
            self.prev_w = self.model.w.clone()

        # momentum term
        momentum = beta * (self.model.w - self.prev_w)

        # gradient descent update
        new_w = self.model.w - alpha * grad + momentum

        # update weights and prev_w
        self.prev_w = self.model.w.clone()
        self.model.w = new_w
