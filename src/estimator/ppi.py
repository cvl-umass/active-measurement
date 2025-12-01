import numpy as np


class PPI:
    def __init__(self, model, ):
        self.model = model
        self.mask = np.ones(self.model.N)
        self.labeled_mask = np.zeros(self.model.N)
        self.round = 0
        self.labels = []

    def step(self,):
        if np.sum(self.mask) == 0:
            return
        id, _, truth = self.model.sample_index(mask=self.mask)
        self.mask[id] = 0
        self.labels.append(self.model._g[id] - truth)
        g = np.array(self.model._g)
        self.estimate = np.mean(g) - np.mean(self.labels)
        self.round += 1

    def train(self):
        res = self.model.train(self.labeled_mask)

    def get_count(self, ):
        return self.estimate * self.model.N
