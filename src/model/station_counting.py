import numpy as np
import pandas as pd

def dist(f,g):
    f = np.array(f)/np.sum(f)
    g = np.array(g)/np.sum(g)
    return np.sqrt(np.sum(np.square(np.sqrt(f) - np.sqrt(g)))) / np.sqrt(2)

class StationCounting:
    def __init__(self, station='KAPX', seed = 0, uniform = False, active_testing = False):
        self.f = []
        finetunes = [10, 20, 30, 40]
        self.gs = [[] for _ in range(len(finetunes)+1)]
        for year in [2015, 2016, 2017, 2018, 2019]:
            csvfile = f'../data/roost_counts/ground_truth/day_counts/day_counts_{station}_{year}0601_{year}1031.csv'
            csvfile2 = f'../data/roost_counts/init/day_counts_with_ui_filter/day_counts_{station}_{year}0601_{year}1031.txt'
            finetunefiles = [f'../data/roost_counts/{station}_{ft}/day_counts_with_ui_filter/day_counts_{station}_{year}0601_{year}1031.txt' for ft in finetunes]
            fs = pd.read_csv(csvfile)
            fs2 = pd.read_csv(csvfile2)
            fss = [pd.read_csv(csvfile3) for csvfile3 in finetunefiles]
            for month in ['06','07','08','09','10']:
                for day in range(1, 32):
                    if day == 31 and month in ['06', '09']:
                        continue
                    id = f'{station}{year}{month}{day}'
                    if id in fs["station_day"].values:
                        self.f.append(fs.loc[fs["station_day"] == id, "n_animals"].iloc[0])
                    else:
                        self.f.append(0)
                    if id in fs2["station_day"].values:
                        self.gs[0].append(fs2.loc[fs2["station_day"] == id, "n_animals"].iloc[0])
                    else:
                        self.gs[0].append(0)
                    for i, fs3 in enumerate(fss):
                        if id in fs3["station_day"].values:
                            self.gs[i+1].append(fs3.loc[fs3["station_day"] == id, "n_animals"].iloc[0])
                        else:
                            self.gs[i+1].append(0)

        self.f = np.array(self.f)
        self.g = np.array(self.gs[0])
        self.max_round = len(self.gs)
        self.t = 0
        self.N = len(self.f)
        self.random_state = np.random.RandomState(seed)
        self.uniform = uniform
        self.active_testing = active_testing
        if active_testing:
            self.g = np.abs(self.f-self.g)
        if uniform:
            self.g = np.ones_like(self.f)

    def evaluate(self, index):
        return self.g[index]

    def train(self, mask = None):
        if self.t < self.max_round - 1 and not self.uniform:
            self.t = self.t + 1
            self.g = np.array(self.gs[self.t])
            if self.active_testing:
                self.g = np.abs(self.f - self.g)
            return True
        return False

    def sample_index(self, mask):
        probs = mask * (self.g + 1000)
        probs = probs / np.sum(probs)
        index = self.random_state.choice(np.arange(self.N), p=probs)
        return index, probs, self.f[index]

    def ground_truth(self):
        return np.sum(self.f)
