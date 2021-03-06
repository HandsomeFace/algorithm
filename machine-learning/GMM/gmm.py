from numpy import *

def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet


class GMM:

    def __init__(self, k=4, eps=0.00001):
        self.k = k
        self.eps = eps

    def fit_EM(self, X, max_iters=1000):

        # n = number of data-points, d = dimension of data points
        n, d = X.shape

        # randomly choose the starting centroids/means
        ## as 3 of the points from datasets
        mu = X[np.random.choice(n, self.k, False), :]

        # initialize the covariance matrices for each gaussians
        Sigma = [np.eye(d)] * self.k

        # initialize the probabilities/weights for each gaussians
        w = [1. / self.k] * self.k

        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))

        ### log_likelihoods
        log_likelihoods = []

        P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-d / 2) \
                          * np.exp(-.5 * np.einsum('ij, ij -> i', \
                                                   X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

        # Iterate till max_iters iterations
        while len(log_likelihoods) < max_iters:

            # E - Step

            ## Vectorized implementation of e-step equation to calculate the
            ## membership for each of k -gaussians
            for k in range(self.k):
                a = w[k] * P(mu[k], Sigma[k])
                R[:, k] = a

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis=1)))

            log_likelihoods.append(log_likelihood)

            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis=1)).T

            ## The number of datapoints belonging to each gaussian
            N_ks = np.sum(R, axis=0)

            # M Step
            ## calculate the new mean and covariance for each gaussian by
            ## utilizing the new responsibilities
            for k in range(self.k):
                ## means
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis=1).T
                x_mu = np.matrix(X - mu[k])

                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T, R[:, k]), x_mu))

                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]
            # check for convergence
            if len(log_likelihoods) < 2: continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break

        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])
        self.params.mu = mu
        self.params.Sigma = Sigma
        self.params.w = w
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)

        return self.params

    def plot_log_likelihood(self):
        import pylab as plt
        plt.plot(self.params.log_likelihoods)
        plt.title('Log Likelihood vs iteration plot')
        plt.xlabel('Iterations')
        plt.ylabel('log likelihood')
        plt.show()

    def predict(self, x):
        p = lambda mu, s: np.linalg.det(s) ** - 0.5 * (2 * np.pi) ** \
                                                      (-len(x) / 2) * np.exp(-0.5 * np.dot(x - mu, \
                                                                                           np.dot(np.linalg.inv(s),
                                                                                                  x - mu)))
        probs = np.array([w * p(mu, s) for mu, s, w in \
                          zip(self.params.mu, self.params.Sigma, self.params.w)])
        return probs / np.sum(probs)














