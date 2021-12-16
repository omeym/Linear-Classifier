
import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################
        learning_rate = step_size
        #Updating the values of y and X to suit the perceptron function
        X_new = np.insert(X,0,1,axis =1)
        w_new = np.insert(w,0,b,axis=0)
        y_new = np.where(y==0,-1,1)
        for iter in range(max_iterations+1):
            predicted_value = binary_predict(X_new,w_new,123456789)
            predicted_value = np.where(predicted_value==0,-1,1)
            expected_value =  y_new
            z = expected_value*predicted_value
            z = np.where(z<=0,1,0)
            gradient = (1/N) * np.dot((z*y_new), X_new)
            w_new = w_new + (learning_rate*gradient) 
        
        

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        learning_rate = step_size
        X_new = np.insert(X,0,1,axis =1)
        w_new = np.insert(w,0,b,axis=0)
        y_new = np.where(y==0,-1,1)
        for i in range(max_iterations+1):
            predicted_value = np.dot(X_new, w_new)
            expected_value =  y_new
            z = expected_value*predicted_value
            sigma_val = sigmoid(-z)
            gradient = (1/N) * np.dot((sigma_val*y_new), X_new)
            w_new = w_new + (learning_rate*gradient)


        

    else:
        raise "Undefined loss function."

    b = w_new[0]
    w = w_new[1:]
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    
    value = 1/(1+np.exp(-z))

    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    if b != 123456789:
        X = np.insert(X,0,1,axis = 1)
        w = np.insert(w,0,b,axis = 0)
    
    preds = np.sign(np.dot(X,w))
    preds = np.where(preds==-1,0,1)
    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    x_new = np.append(X, np.ones((N, 1)), axis=1)
    w_new = np.append(w, np.array([b]).T, axis=1)
    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################			
            w_x = np.matmul(w_new, x_new[n].transpose())
            Predictions = softmax(w_x)
            #Dealing with the case k=yn
            Predictions[y[n]] = Predictions[y[n]] - 1
            sgd = np.matmul(np.array([Predictions]).transpose(), np.array([x_new[n]]))
            w_new = w_new - (step_size * sgd)
            

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        y = np.eye(C)[y]
        for i in range(max_iterations):
            w_x = x_new.dot(w_new.transpose())
            Predictions = softmax(w_x, axis1 = None, axis2 = 1, take_transpose= True)
            error = Predictions - y
            gd = np.dot(error.transpose(), x_new)
            w_new = w_new - ((step_size/N) * gd) 

    else:
        raise "Undefined algorithm."
    
    b = w_new[:, -1]
    w = np.delete(w_new, -1, 1)
    
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    x_new = np.insert(X,0,1, axis=1)
    w_new = np.insert(w,0,b, axis=1)
    z = np.dot(x_new,w_new.transpose())
   
    preds = np.argmax(z, axis=1)
    
    assert preds.shape == (N,)
    return preds


def softmax(z, axis1 = None, axis2 = None, take_transpose=False):
    z_new = np.subtract(z, np.amax(z, axis=axis1))

    if(not take_transpose):
        z_softmax = np.exp(z_new)/np.sum(np.exp(z_new), axis = axis2)
    else:
        numerator_term = np.exp(z - np.amax(z))
        denominator_term = np.sum(numerator_term, axis=1)
        z_softmax = (numerator_term.transpose() / denominator_term).transpose()
    
    return z_softmax


        