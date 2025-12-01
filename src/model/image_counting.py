import numpy as np
import pandas as pd


class ImageCounting:
    def __init__(self, method='finetune', seed = 0, uniform = False, active_testing = False, path = f'../data/DSC5214_data.csv'):
        assert(method in ['finetune', 'famnet', 'rand', 'rand2'])
        if method == 'finetune':
            columns = ['finetune_1', 'finetune_10', 'finetune_20', 'finetune_30', 'finetune_40', 'finetune_50', 'finetune_60', 'finetune_70',]
        elif method == 'famnet':
            columns = ['famnet_1', 'famnet_2', 'famnet_3', 'famnet_4', 'famnet_5', 'famnet_6', 'famnet_7', ]
        elif method == 'rand':
            columns = ['rand_1', 'rand_10', 'rand_20', 'rand_30', 'rand_40', 'rand_50', 'rand_60', 'rand_70',]
        elif method == 'rand2':
            columns = ['rand2_1', 'rand2_10', 'rand2_20', 'rand2_30', 'rand2_40', 'rand2_50', 'rand2_60', 'rand2_70',]
        else:
            raise ValueError(method)
        self.columns = columns
        self.max_round = len(columns)
        self.f = []
        self.gs = pd.read_csv(path)
        self.f = self.gs['ground_truth'].to_numpy()
        self.g = self.gs[columns[0]].to_numpy()
        self.t = 0
        self.N = len(self.f)
        self.random_state = np.random.RandomState(seed)
        self.uniform = uniform
        self.active_testing = active_testing
        self._g = self.g.copy()
        if active_testing:
            self.g = np.abs(self.f-self.g)
        if uniform:
            self.g = np.ones_like(self.f)


    def evaluate(self, index):
        return self.g[index]

    def train(self, mask = None):
        if self.t < self.max_round - 1:
            self.t = self.t + 1
            self.g = self.gs[self.columns[self.t]].to_numpy()
            self._g = self.g.copy()
            if self.active_testing:
                self.g = np.abs(self.f - self.g)
            if self.uniform:
                self.g = np.ones_like(self.f)
            return True
        return False

    def sample_index(self, mask):
        probs = mask * np.maximum(self.g, 1)
        probs = probs / np.sum(probs)
        index = self.random_state.choice(np.arange(self.N), p=probs)
        return index, probs, self.f[index]

    def ground_truth(self):
        return np.sum(self.f)
