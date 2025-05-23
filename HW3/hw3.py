###### Your ID ######
# ID1: 323842914
# ID2: 208663120
#####################

import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)
        
        # First, let's define P(X|C) and P(Y|C) to ensure conditional independence
        # X given C: P(X=x|C=c) ->
        #   P(X=0|C=0) = 0.1
        #   P(X=1|C=0) = 0.9
        #   P(X=0|C=1) = 0.5
        #   P(X=1|C=1) = 0.5
        
        # Y given C: P(Y=y|C=c) ->
        #   P(Y=0|C=0) = 0.1
        #   P(Y=1|C=0) = 0.9
        #   P(Y=0|C=1) = 0.5
        #   P(Y=1|C=1) = 0.5

        self.X_Y = {
            (0, 0): 0.13,  # = P(X=0,Y=0,C=0) + P(X=0,Y=0,C=1) = 0.005 + 0.125 = 0.13
            (0, 1): 0.17,  # = P(X=0,Y=1,C=0) + P(X=0,Y=1,C=1) = 0.045 + 0.125 = 0.17
            (1, 0): 0.17,  # = P(X=1,Y=0,C=0) + P(X=1,Y=0,C=1) = 0.045 + 0.125 = 0.17
            (1, 1): 0.53  # = P(X=1,Y=1,C=0) + P(X=1,Y=1,C=1) = 0.405 + 0.125 = 0.53
        } # P(X=x, Y=y) = = P(X=x,Y=y,C=0) + P(X=x,Y=y,C=1)
        
        self.X_C = {
            (0, 0): 0.05,  # = P(X=0|C=0) * C = 0.1 * 0.5 = 0.05
            (0, 1): 0.25,  # = P(X=0|C=1) * P(C=1) = 0.5 * 0.5 = 0.25
            (1, 0): 0.45,  # = P(X=1|C=0) * P(C=0) = 0.9 * 0.5 = 0.45
            (1, 1): 0.25,  # = P(X=1|C=1) * P(C=1) = 0.5 * 0.5 = 0.25
        } # P(X=x, C=y) = P(X=x|C=c) * P(C=c)
        
        self.Y_C = {
            (0, 0): 0.05,  # = P(Y=0|C=0) * P(C=0) = 0.1 * 0.5 = 0.05
            (0, 1): 0.25,  # = P(Y=0|C=1) * P(C=1) = 0.5 * 0.5 = 0.25
            (1, 0): 0.45,  # = P(Y=1|C=0) * P(C=0) = 0.9 * 0.5 = 0.45
            (1, 1): 0.25,  # = P(Y=1|C=1) * P(C=1) = 0.5 * 0.5 = 0.25
        } # P(Y=y, C=c) = P(Y=y|C=c) * P(C=c)
        
        self.X_Y_C = {
            (0, 0, 0): 0.005,  # = P(X=0|C=0) * P(Y=0|C=0) * P(C=0) = 0.1 * 0.1 * 0.5 = 0.005
            (0, 0, 1): 0.125,  # = P(X=0|C=1) * P(Y=0|C=1) * P(C=1) = 0.5 * 0.5 * 0.5 = 0.125
            (0, 1, 0): 0.045,  # = P(X=0|C=0) * P(Y=1|C=0) * P(C=0) = 0.1 * 0.9 * 0.5 = 0.045
            (0, 1, 1): 0.125,  # = P(X=0|C=1) * P(Y=1|C=1) * P(C=1) = 0.5 * 0.5 * 0.5 = 0.125
            (1, 0, 0): 0.045,  # = P(X=1|C=0) * P(Y=0|C=0) * P(C=0) = 0.9 * 0.1 * 0.5 = 0.045
            (1, 0, 1): 0.125,  # = P(X=1|C=1) * P(Y=0|C=1) * P(C=1) = 0.5 * 0.5 * 0.5 = 0.125
            (1, 1, 0): 0.405,  # = P(X=1|C=0) * P(Y=1|C=0) * P(C=0) = 0.9 * 0.9 * 0.5 = 0.405
            (1, 1, 1): 0.125,  # = P(X=1|C=1) * P(Y=1|C=1) * P(C=1) = 0.5 * 0.5 * 0.5 = 0.125
        } # P(X=x, Y=y, C=c) = P(X=x|C=c) * P(Y=y|C=c) * P(C=c)
        

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for (x, y), p_xy in X_Y.items():
            if not np.isclose(p_xy, X[x] * Y[y]):
                return True  # dependency found
        return False  # all joint probs match product of marginals → independent

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for c in C:
            p_c = C[c]
            for x in X:
                for y in Y:
                    p_xyz = X_Y_C[(x, y, c)]        # P(X=x, Y=y, C=c)
                    p_x_given_c = X_C[(x, c)] / p_c # P(X=x | C=c)
                    p_y_given_c = Y_C[(y, c)] / p_c # P(Y=y | C=c)
                    p_xy_given_c = p_xyz / p_c      # P(X=x, Y=y | C=c)
                    if not np.isclose(p_xy_given_c, p_x_given_c * p_y_given_c):
                        return False
        return True

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    log_p = k * np.log(rate) - rate - np.math.lgamma(k + 1)
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.array([
        np.sum([poisson_log_pmf(k, rate) for k in samples]) for rate in rates
    ])
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)
    best_index = np.argmax(likelihoods)
    rate = rates[best_index]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.mean(samples)
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    coef = 1 / (np.sqrt(2 * np.pi) * std)
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    p = coef * exponent
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        
        # Filter only rows that match the class
        self.class_data = dataset[dataset[:, -1] == class_value][:, :-1]  # remove the label column

        # Compute mean and std per feature (column)
        self.means = np.mean(self.class_data, axis=0)
        self.stds = np.std(self.class_data, axis=0, ddof=1)  # Use sample std (ddof=1)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        total_count = self.dataset.shape[0]
        class_count = np.sum(self.dataset[:, -1] == self.class_value)
        prior = class_count / total_count
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1.0
        for i in range(len(x)):
            likelihood *= normal_pdf(x[i], self.means[i], self.stds[i])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        posterior_0 = self.ccd0.get_instance_posterior(x)
        posterior_1 = self.ccd1.get_instance_posterior(x)
        pred = 0 if posterior_0 > posterior_1 else 1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    correct = 0
    for instance in test_set:
        x = instance[:-1]         # features
        true_label = instance[-1] # actual label
        predicted = map_classifier.predict(x)
        if predicted == true_label:
            correct += 1
    acc = correct / len(test_set)
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    d = len(x)
    x = np.array(x)
    mean = np.array(mean)
    cov = np.array(cov)

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1 / (np.sqrt((2 * np.pi) ** d * det_cov))
    exponent = -0.5 * (x - mean).T @ inv_cov @ (x - mean)
    pdf = norm_const * np.exp(exponent)
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value

        # Filter rows matching class_value and exclude label column
        self.class_data = dataset[dataset[:, -1] == class_value][:, :-1]

        # Compute mean vector and covariance matrix
        self.mean = np.mean(self.class_data, axis=0)
        self.cov = np.cov(self.class_data, rowvar=False)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        total_count = self.dataset.shape[0]
        class_count = np.sum(self.dataset[:, -1] == self.class_value)
        prior = class_count / total_count
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        prior_0 = self.ccd0.get_prior()
        prior_1 = self.ccd1.get_prior()
        pred = 0 if prior_0 > prior_1 else 1
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        likelihood_0 = self.ccd0.get_instance_likelihood(x)
        likelihood_1 = self.ccd1.get_instance_likelihood(x)
        pred = 0 if likelihood_0 > likelihood_1 else 1
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value

        # Filter rows by class
        self.class_data = dataset[dataset[:, -1] == class_value]
        self.n_features = dataset.shape[1] - 1

        # Precompute feature value counts per class
        self.feature_value_counts = [{} for _ in range(self.n_features)]
        self.feature_totals = [0 for _ in range(self.n_features)]

        for f in range(self.n_features):
            values, counts = np.unique(self.class_data[:, f], return_counts=True)
            self.feature_value_counts[f] = dict(zip(values, counts))
            self.feature_totals[f] = np.sum(counts)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1.0
        for i in range(self.n_features):
            value = x[i]
            count = self.feature_value_counts[i].get(value, 0)
            total = self.feature_totals[i]
            # Laplace smoothing
            p = (count + 1) / (total + len(self.feature_value_counts[i]))
            likelihood *= p if p > 0 else EPSILLON
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        post_0 = self.ccd0.get_instance_posterior(x)
        post_1 = self.ccd1.get_instance_posterior(x)
        pred = 0 if post_0 > post_1 else 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        correct = 0
        for instance in test_set:
            x = instance[:-1]
            y = instance[-1]
            if self.predict(x) == y:
                correct += 1
        acc = correct / len(test_set)
        return acc


