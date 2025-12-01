import numpy as np
import os
from PIL import Image
import pandas as pd

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
import copy

from dataset_utils import get_finetune_cfg, get_image_dicts

class DetectorCounting:
    def __init__(self, seed = 0, path = f'DSC5295_data.csv', dataset_path='tiles/tiles_DSC5295', lr=0.001, output_dir='finetune', min_count=1):
        self.output_dir = output_dir
        self.f = []
        self.gs = pd.read_csv(path)
        self.f = self.gs['ground_truth'].to_numpy()
        self.max_round = 7
        self.g = np.ones_like(self.f) # Initialize with Uniform Proposal
        self.t = 0
        self.N = len(self.f)
        self.random_state = np.random.RandomState(seed)
        self.cfg = get_finetune_cfg(output_dir=output_dir, lr=lr)
        self.dataset = get_image_dicts(dataset_path=dataset_path)
        self.min_count = min_count


    def evaluate(self, index):
        return self.g[index]

    def train(self, mask = None):
        if self.t < self.max_round - 1:
            self.t = self.t + 1
            data = [self.dataset[x] for x in np.where(mask)[0]]
            if len(data) == 0:
                return
            
            # Finetune Model
            DatasetCatalog.register(f"bird_dataset_{self.t}", lambda: data)
            MetadataCatalog.get(f"bird_dataset_{self.t}").set(thing_classes=["bird"])
            self.cfg.DATASETS.TRAIN = (f"bird_dataset_{self.t}",)
            trainer = DefaultTrainer(self.cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()

            # Update Predictions
            test_cfg = copy.deepcopy(self.cfg)
            test_cfg.MODEL.WEIGHTS = os.path.join(self.output_dir, 'model_final.pth')
            predictor = DefaultPredictor(test_cfg)
            g = []
            for data in self.dataset:
                image = Image.open(data['file_name'])
                image.load()
                pred = len(predictor(np.array(image))['instances'])
                g.append(pred)
            self.g = np.array(g)

    def sample_index(self, mask):
        probs = mask * np.maximum(self.g, self.min_count)
        probs = probs / np.sum(probs)
        index = self.random_state.choice(np.arange(self.N), p=probs)
        return index, probs, self.f[index]

    def ground_truth(self):
        return np.sum(self.f)