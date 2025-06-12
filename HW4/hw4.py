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
        # set random seed
        np.random.seed(self.random_state)

        m, n = X.shape

        # Add bias term to X
        X = np.c_[np.ones(m), X]

        # Initialize theta
        self.theta = np.random.randn(n + 1)

        for i in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = 1 / (1 + np.exp(-z))  # sigmoid

            # Compute cost
            J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

            self.Js.append(J)
            self.thetas.append(self.theta.copy())

            # Compute gradient and update theta
            gradient = (1 / m) * np.dot(X.T, (h - y))
            self.theta -= self.eta * gradient
            print(self.theta)

            # Check convergence
            if i > 0 and abs(self.Js[-2] - self.Js[-1]) < self.eps:
                print(self.Js[-2] - self.Js[-1])
                break

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        m = X.shape[0]
        X = np.c_[np.ones(m), X]
        h = 1 / (1 + np.exp(-np.dot(X, self.theta)))  # sigmoid
        preds = (h >= 0.5).astype(int)
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

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    indices = np.arange(len(X))
    np.random.shuffle(indices)  # shuffle the data

    fold_size = len(X) // folds  # determine size of each fold
    acc_list = []

    for i in range(folds):
        val_idx = indices[
            i * fold_size : (i + 1) * fold_size
        ]  # current fold (validation)
        train_idx = np.concatenate(
            (indices[: i * fold_size], indices[(i + 1) * fold_size :])
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = LogisticRegressionGD(
            eta=algo.eta,
            n_iter=algo.n_iter,
            eps=algo.eps,
            random_state=algo.random_state,
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracy = np.mean(preds == y_val)
        acc_list.append(accuracy)

    cv_accuracy = np.mean(acc_list)
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
    p = None
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
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
        self.weights = np.random.dirichlet(np.ones(self.k))

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
            cost = -np.sum(np.log(likelihoods))
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
        self.params = {}
        self.em = EM(self.k, random_state=self.random_state)

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
        # Extract unique classes and features
        classes, count = np.unique(y, return_counts=True)
        features = np.arange(X.shape[1])
        dataset = np.hstack((X, y.reshape(-1, 1)))

        for class_value in classes:
            self.params[class_value] = {}
            # Subset of data belonging to the current class
            dataset_per_class = dataset[dataset[:, -1] == class_value]
            prior = len(dataset_per_class) / len(
                dataset
            )  # Calculate prior probability of the class
            self.params[class_value]["prior"] = prior

            means, stds, weights = [], [], []
            for feature in features:
                # Fit EM on the feature column for the current class
                self.em.fit(dataset_per_class[:, feature])
                w, m, s = self.em.get_dist_params()
                means.append(m)
                stds.append(s)
                weights.append(w)

            self.params[class_value]["means"] = means
            self.params[class_value]["stds"] = stds
            self.params[class_value]["weights"] = weights

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        classes = list(self.params.keys())
        features = np.arange(X.shape[1])
        predictions = []

        # Loop over each sample in X
        for i in range(X.shape[0]):
            class_posteriors = {}

            for class_value in classes:
                likelihood = 1

                for feature in features:
                    feature_val = X[i, feature]
                    mean = self.params[class_value]["means"][feature]
                    std = self.params[class_value]["stds"][feature]
                    weight = self.params[class_value]["weights"][feature]

                    # Compute the weighted sum of PDFs
                    gmm_prob = 0
                    for k in range(self.k):
                        gmm_prob += weight[k] * norm_pdf(feature_val, mean[k], std[k])
                    likelihood *= gmm_prob

                posterior = likelihood * self.params[class_value]["prior"]
                class_posteriors[class_value] = posterior

            prediction = max(class_posteriors, key=class_posteriors.get)
            predictions.append(prediction)

        preds = np.array(predictions)

        return preds


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

    # Train models on full dataset
    lor_full = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_full.fit(x_train, y_train)

    nb_full = NaiveBayesGaussian(k=k)
    nb_full.fit(x_train, y_train)

    # Compute accuracies
    lor_train_acc = np.mean(lor_full.predict(x_train) == y_train)
    lor_test_acc = np.mean(lor_full.predict(x_test) == y_test)
    bayes_train_acc = np.mean(nb_full.predict(x_train) == y_train)
    bayes_test_acc = np.mean(nb_full.predict(x_test) == y_test)

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

    # Dataset A: Naive Bayes performs better
    size = [300, 300, 300]
    means_class0 = [[5, 5, 5], [10, 0, 0], [5, -5, 0]]
    cov = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * 3
    data_a_0 = np.vstack(
        [
            np.random.multivariate_normal(mean, cov[i], size[i])
            for i, mean in enumerate(means_class0)
        ]
    )
    labels_a_0 = np.zeros(data_a_0.shape[0])

    means_class1 = [[-5, -5, -5], [-10, 0, 0], [-5, 5, 0]]
    data_a_1 = np.vstack(
        [
            np.random.multivariate_normal(mean, cov[i], size[i])
            for i, mean in enumerate(means_class1)
        ]
    )
    labels_a_1 = np.ones(data_a_1.shape[0])

    dataset_a_features = np.vstack([data_a_0, data_a_1])
    dataset_a_labels = np.hstack([labels_a_0, labels_a_1])

    # Dataset B: Logistic Regression performs better
    mean_0 = [0, 0, 0]
    cov_0 = [[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1]]
    data_b_0 = np.random.multivariate_normal(mean_0, cov_0, 500)
    labels_b_0 = np.zeros(500)

    mean_1 = [3, 3, 3]
    cov_1 = [[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1]]
    data_b_1 = np.random.multivariate_normal(mean_1, cov_1, 500)
    labels_b_1 = np.ones(500)

    dataset_b_features = np.vstack([data_b_0, data_b_1])
    dataset_b_labels = np.hstack([labels_b_0, labels_b_1])

    return {
        "dataset_a_features": dataset_a_features,
        "dataset_a_labels": dataset_a_labels,
        "dataset_b_features": dataset_b_features,
        "dataset_b_labels": dataset_b_labels,
    }
