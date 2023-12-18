import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data as data
from loguru import logger

import utils.config as config
from engine.engine import test
from model import build_segmenter
from utils.dataset_cod import TestDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='config/CODdataset/codclip_vit_L14@336.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


if __name__ == '__main__':

    args = get_parser()
    args.output_dir = os.path.join(args.map_save_path, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # build model
    model, _ = build_segmenter(args)
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx in [448, 449, 450, 451, 452, 453, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476]:
            print(f"Index: {idx}, Name: {name}, Size: {param.size()}")