###### Your ID ######
# ID1: 323842914
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd
import itertools

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    # Mean normalization: (x - mean) / (max - min)
    X = (X - X.mean()) / (X.max() - X.min())
    y = (y - y.mean()) / (y.max() - y.min())

    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1))  # column of ones
    X = np.hstack((ones, X))         # add as first column
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    m = y.shape[0]  # number of training examples
    predictions = X @ theta  # matrix multiplication to get predictions
    errors = predictions - y
    J = (1 / (2 * m)) * np.dot(errors, errors)  # same as sum(errors^2)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = y.shape[0]  # number of training examples
    for i in range(num_iters):
        predictions = X @ theta                    # shape: (m,)
        errors = predictions - y                   # shape: (m,)
        gradient = (1 / m) * (X.T @ errors)        # shape: (n,)
        theta = theta - alpha * gradient           # update rule

        # Record the cost at each step
        J = (1 / (2 * m)) * np.dot(errors, errors) # optional optimization
        J_history.append(J)
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    # Normal Equation: theta = (X^T X)^(-1) X^T y
    XTX = X.T @ X
    XTy = X.T @ y
    pinv_theta = np.linalg.inv(XTX) @ XTy
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = y.shape[0]
    for i in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        # Compute cost early to catch NaNs
        cost = (1 / (2 * m)) * np.dot(errors, errors)
        if np.isnan(cost) or np.isinf(cost):
            break
        J_history.append(cost)
        gradient = (1 / m) * (X.T @ errors)
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            break
        theta -= alpha * gradient
        if i > 0 and abs(J_history[-2] - J_history[-1]) < 1e-8:
            break
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    for alpha in alphas:
        theta_init = np.zeros(X_train.shape[1])
        theta, _ = efficient_gradient_descent(X_train, y_train, theta_init, alpha, iterations)
        val_loss = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = val_loss
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    num_features = X_train.shape[1]
    remaining_features = list(range(num_features))
    
    for _ in range(5):
        best_feature = None
        lowest_cost = float('inf')

        for feature in remaining_features:
            # Temporarily add this feature to the current selection
            candidate_features = selected_features + [feature]
            
            # Slice selected columns from X_train and X_val
            X_train_subset = X_train[:, candidate_features]
            X_val_subset = X_val[:, candidate_features]

            # Apply the bias trick
            X_train_bias = apply_bias_trick(X_train_subset)
            X_val_bias = apply_bias_trick(X_val_subset)

            # Initialize theta
            theta_init = np.zeros(X_train_bias.shape[1])

            # Train model
            theta, _ = efficient_gradient_descent(
                X_train_bias, y_train, theta_init, best_alpha, iterations)

            # Evaluate on validation set
            cost = compute_cost(X_val_bias, y_val, theta)

            # Update best feature if this is the best cost
            if cost < lowest_cost:
                lowest_cost = cost
                best_feature = feature

        # Permanently add the best feature of this round
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    feature_names = df.columns

    new_features = {}

    # Add square terms
    for feature in feature_names:
        new_features[f"{feature}^2"] = df[feature] ** 2

    # Add interaction terms
    for f1, f2 in itertools.combinations(feature_names, 2):
        new_features[f"{f1}*{f2}"] = df[f1] * df[f2]

    # Combine original df with new features in one call
    df_poly = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    
    return df_poly
