import os, time, tqdm
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import sys
from visualize import Visualization

VAR_LAYER_CNT = 50	# COCO dataset (in case of rcnn_R_50_FPN)
VAR_NUM_CLASSES = 6	# number of classes
VAR_RES_DIR = './result'
VAR_OUTPUT_DIR = './output'

from detectron2 import model_zoo

def setup_cfg(path):
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    if (VAR_LAYER_CNT == 50):
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    else:
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = VAR_NUM_CLASSES  # only has one class (chicken)
    cfg.MODEL.WEIGHTS = os.path.join(path, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    cfg.freeze()
    return cfg

def test_main():
    setup_logger(name="fvcore")

    r_path = VAR_RES_DIR
    m_path = VAR_OUTPUT_DIR
    cfg = setup_cfg(m_path)
    os.makedirs(r_path, exist_ok=True)

    print('meta: ', cfg.DATASETS.TEST[0], cfg.DATASETS)
    vis = Visualization(cfg)

    idx=0
    while True:
        img = cv2.imread('/home/parksy1314/colcon_ws/test.jpg')

        num_instances, v_output = vis.run_on_image(img)

        if num_instances == -1:
            continue

        fname = 'img_' + str(idx) + '.png'
        out_filename = os.path.join(r_path, fname)
        idx += 1
        v_output.save(out_filename)

    rs.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_main()
