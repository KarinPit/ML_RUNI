###### Your ID ######
# ID1: 323842914
# ID2: 208663120
#####################


import numpy as np


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed and initialize weights w and bias b
        np.random.seed(self.random_state)

        # Add bias term to X
        m, n = X.shape
        X = np.c_[np.ones(m), X]
        self.theta = np.random.randn(n + 1)

        for n in range(self.n_iter):
            # 1. Compute linear combination:
            Z = np.dot(X, self.theta)

            # 2. Apply sigmoid function:
            y_hat = 1 / (1 + np.exp(-Z))

            # 3. Compute the log-loss:
            J = (-1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            self.Js.append(J)
            self.thetas.append(self.theta.copy())

            # 4. Compute gradients:
            gradient = (1 / m) * np.dot(X.T, (y_hat - y))

            # 5. Update weights:
            self.theta -= self.eta * gradient

            # 6. Check convergence:
            if n > 0:
                J_diff = abs(self.Js[-2] - self.Js[-1])
                if J_diff < self.eps:
                    print("Converged")
                    break

    def predict(self, X, threshold=0.5):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        # Add bias term to X
        m = X.shape[0]
        X = np.c_[np.ones(m), X]

        # Compute predictions (sigmoid)
        y_hat = 1 / (1 + np.exp(-np.dot(X, self.theta)))

        # Apply threshold (default 0.5)
        preds = (y_hat >= threshold).astype(int)

        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = []

    # set random seed and shuffle data
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # divide data into folds
    X_folds = np.array_split(X_shuffled, folds)
    y_folds = np.array_split(y_shuffled, folds)

    for i in range(folds):
        # create copy of X and y arrays to avoid overide
        train_X = X_folds.copy()
        train_y = y_folds.copy()

        # remove the ith fold array from train data
        test_X = train_X.pop(i)
        test_y = train_y.pop(i)

        # merge the fold arrays to one array
        train_X = np.concatenate(train_X)
        train_y = np.concatenate(train_y)

        # create new instance of model and predict results
        model = LogisticRegressionGD(
            eta=algo.eta, eps=algo.eps, random_state=algo.random_state
        )

        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)

        # calculate percentage of match between predictions and real value
        acc = np.mean(y_pred == test_y)
        cv_accuracy.append(acc)

    # calculate total match percentage for a given modal instance
    cv_accuracy = np.mean(cv_accuracy)

    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.
    """

    p = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)

    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = {}
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        data = data.ravel()  # ensures data is flattened
        batches = np.array_split(data, self.k)
        self.mus = np.array([np.mean(batch) for batch in batches])
        self.sigmas = np.array([np.std(batch) for batch in batches])
        self.weights = np.ones(self.k) / self.k

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        data = data.ravel()
        m = data.shape[0]
        self.responsibilities = np.zeros((m, self.k))

        for j in range(self.k):
            pdf_vals = norm_pdf(data, self.mus[j], self.sigmas[j]).flatten()
            self.responsibilities[:, j] = self.weights[j] * pdf_vals

        sum_responsibilities = np.sum(self.responsibilities, axis=1, keepdims=True)
        self.responsibilities /= sum_responsibilities

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        data = data.ravel()
        m = data.shape[0]

        for j in range(self.k):
            r = self.responsibilities[:, j]
            total_r = np.sum(r)
            self.weights[j] = total_r / m
            self.mus[j] = np.sum(r * data) / total_r
            self.sigmas[j] = np.sqrt(np.sum(r * (data - self.mus[j]) ** 2) / total_r)

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        data = data.ravel()
        self.init_params(data)

        for i in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)

            # compute negative log-likelihood cost
            weighted_pdfs = np.zeros((data.shape[0], self.k))
            for j in range(self.k):
                weighted_pdfs[:, j] = (
                    self.weights[j]
                    * norm_pdf(data, self.mus[j], self.sigmas[j]).flatten()
                )
            likelihoods = np.sum(weighted_pdfs, axis=1)
            cost = -np.sum(np.log(likelihoods + 1e-10))  # Avoid log(0)
            self.costs.append(cost)

            if i > 0 and abs(self.costs[-2] - self.costs[-1]) < self.eps:
                break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.
    """
    pdf = None
    k = len(weights)
    pdf = np.zeros_like(data, dtype=float)
    for j in range(k):
        pdf += weights[j] * norm_pdf(data, mus[j], sigmas[j])
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.models = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        n, d = X.shape

        classes = np.unique(y)
        priors = {c: np.sum(y == c) / len(y) for c in classes}
        self.prior = priors
        self.models = {}

        for c in classes:
            X_c = X[y == c]
            self.models[c] = []
            for f in range(d):
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(X_c[:, f])
                self.models[c].append(em)

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        n, d = X.shape
        classes = list(self.models.keys())
        preds = []

        for i in range(n):
            sample = X[i]
            class_log_likelihoods = {}

            for c in classes:
                # Start with log prior
                log_likelihood = np.log(self.prior[c] + 1e-10)

                # Add log-likelihood for each feature under its GMM
                for f in range(d):
                    em = self.models[c][f]
                    gmm_val = gmm_pdf(sample[f], *em.get_dist_params())
                    log_likelihood += np.log(gmm_val + 1e-10)  # Avoid log(0)

                class_log_likelihoods[c] = log_likelihood

            # Pick the class with highest log-likelihood
            best_class = max(class_log_likelihoods, key=class_log_likelihoods.get)
            preds.append(best_class)

        return np.array(preds)


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    """
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    """

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    lor_train_pred = lor.predict(x_train)
    lor_test_pred = lor.predict(x_test)
    lor_train_acc = np.mean(lor_train_pred == y_train)
    lor_test_acc = np.mean(lor_test_pred == y_test)

    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    train_pred = naive_bayes.predict(x_train)
    bayes_train_acc = np.mean(train_pred == y_train)
    test_pred = naive_bayes.predict(x_test)
    bayes_test_acc = np.mean(test_pred == y_test)

    return {
        "lor_train_acc": lor_train_acc,
        "lor_test_acc": lor_test_acc,
        "bayes_train_acc": bayes_train_acc,
        "bayes_test_acc": bayes_test_acc,
    }


def generate_datasets():
    from scipy.stats import multivariate_normal

    """
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    """
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    n_samples = 1500

    # Dataset A: Suits Naive Bayes (features conditionally independent given class)
    mean0_a = [0, 0, 0]
    cov0_a = np.eye(3)  # Independent features
    mean1_a = [3, 3, 3]
    cov1_a = np.eye(3) * 1.2

    X0_a = multivariate_normal.rvs(mean=mean0_a, cov=cov0_a, size=n_samples)
    X1_a = multivariate_normal.rvs(mean=mean1_a, cov=cov1_a, size=n_samples)

    dataset_a_features = np.vstack([X0_a, X1_a])
    dataset_a_labels = np.array([0] * n_samples + [1] * n_samples)

    # Dataset B: Logistic Regression performs better
    mean_0 = [-3, -1, -1]
    mean_1 = [-1, -1, -1]

    # Much tighter Gaussians (less spread)
    cov = np.array([[1.0, 0.95, 0.95], [0.95, 1.0, 0.95], [0.95, 0.95, 1.0]])

    data_b_0 = np.random.multivariate_normal(mean_0, cov, 500)
    labels_b_0 = np.zeros(500)

    data_b_1 = np.random.multivariate_normal(mean_1, cov, 500)
    labels_b_1 = np.ones(500)

    dataset_b_features = np.vstack([data_b_0, data_b_1])
    dataset_b_labels = np.hstack([labels_b_0, labels_b_1])

    return {
        "dataset_a_features": dataset_a_features,
        "dataset_a_labels": dataset_a_labels,
        "dataset_b_features": dataset_b_features,
        "dataset_b_labels": dataset_b_labels,
    }
