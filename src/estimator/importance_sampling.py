import numpy as np

class ImportanceSampling:
    def __init__(self,):
        self.fs = []
        self.estimators = []
        self.weights = []
        self.variance_estimates = []
        self.qs = []
        self.N = 0

    def append_estimator(self, q, f, index, new_f, mask, *args, **kwargs):
        q = np.array(q)
        means = np.mean(np.array(f) / q[index])
        estimator = np.sum(self.fs) + means
        self.fs.extend(new_f)
        self.estimators.append(estimator)
        self.N += 1
        S = np.sum(mask)
        self.weights.append(1 / S / (S - 1))
        self.qs.append(q.copy())
        vars = np.var(np.array(f) / q[index], ddof=1) / len(f)
        self.variance_estimates.append(vars)


    def estimate(self, weights = None, *args, **kwargs):
        if self.N == 0:
            return
        if weights is None:
            weights = np.array(self.weights)
        estimators = np.array(self.estimators)
        return np.sum(weights * estimators) / np.sum(weights)

    def total_variance(self, weights = None, *args, **kwargs):
        if weights is None:
            weights = np.array(self.weights)
        weights = weights / np.sum(weights)
        variances = np.array(self.variance_estimates)
        return np.sum(np.square(weights) * variances)

