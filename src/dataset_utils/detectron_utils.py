from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode

import json
from collections import defaultdict
import skimage as ski
import numpy as np
import os
from PIL import Image

def get_finetune_cfg(output_dir='finetune', lr=0.001):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("bird_dataset_train",)
    cfg.DATASETS.TEST = ("bird_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.SOLVER.IMS_PER_BATCH = 8  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = lr # pick a good LR
    cfg.SOLVER.MAX_ITER = 50#400
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    cfg.TEST.EVAL_PERIOD = 0
    cfg.OUTPUT_DIR = output_dir
    cfg.SOLVER.CHECKPOINT_PERIOD = 400
    cfg.SOLVER.WARMUP_ITERS = 0

    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    return cfg

def get_annotations(task,batch):
    # Simple code to estimate the size of the bounding boxes
    imagepath = 'tiles/{}'.format(task)
    json_file = 'tiles/tasks/{}/labels/{}.json'.format(task,batch)

    with open(json_file, "r") as f:
        data = json.load(f)

    annotations = defaultdict(list)
    for key, value in data.items():
        filename = value["filename"]
        name = filename.split('/')[-1]
        batch = filename.split('/')[-2]
        imagename = '{}/{}/{}'.format(imagepath,batch,name)
        img = ski.io.imread(imagename)
        regions = value["regions"]

        # Calculate the rough size based on the area of the foreground pixels
        gray_image = ski.color.rgb2gray(img)
        H, W = gray_image.shape
        threshold_otsu = ski.filters.threshold_otsu(gray_image)
        thresholded_image = gray_image < threshold_otsu
        total_area = sum(sum(thresholded_image))
        total_regions = len(regions)
        mask = np.zeros((H,W))

        if total_regions > 0:
            #Approximate each bird using a square of radius r
            #Include a multiplicative factor as the bird occupies a faction of the box
            r = np.sqrt(total_area/total_regions)/2*1.5
            r_buffer = r*2
            for region in regions:
                cx = int(region["shape_attributes"]["cx"])
                cy = int(region["shape_attributes"]["cy"])
                minx = int(max(0,cx-r_buffer))
                maxx = int(min(W-1,cx+r_buffer))
                miny = int(max(0,cy-r_buffer))
                maxy = int(min(H-1, cy+r_buffer))
                mask[miny:maxy, minx:maxx] = 1

            total_area = sum(sum(mask*thresholded_image))
            r = np.sqrt(total_area/total_regions)/2*1.5
            for region in regions:
                cx = int(region["shape_attributes"]["cx"])
                cy = int(region["shape_attributes"]["cy"])
                annotations[name].append({"bbox": [cx-r,cy-r,cx+r,cy+r],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0})
    return annotations

# data registering
def get_image_dicts(dataset_path='tiles/tiles_DSC5295'):
    dataset_dicts = []
    task = dataset_path.split('/')[-1]
    annotation_data = {}
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if len(dirnames) > 0:
            continue
        batch = dirpath.split('/')[-1]
        if batch not in annotation_data:
            annotation_data[batch] = get_annotations(task, batch)
        for f in filenames:
            data = {}
            data['file_name'] = os.path.join(dirpath, f)
            data['width'], data['height'] = Image.open(data['file_name']).size
            data['annotations'] = annotation_data[batch][f]
            dataset_dicts.append(data)

    for i, d in enumerate(dataset_dicts):
        d['image_id'] = i
    return dataset_dicts