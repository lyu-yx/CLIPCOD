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
                        default='path to xxx.yaml',
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


@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    args.model_dir = os.path.join(args.output_dir, "Net_epoch_best.pth")
    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    # load data
    for cur_dataset in args.test_dataset:
        test_root = os.path.join(args.test_root, cur_dataset)
        print(f"Loading {cur_dataset}...")
        test_data = TestDataset(image_root=test_root + 'Imgs/',
                                gt_root=test_root + 'GT/',
                                desc_root=test_root + 'Desc/',
                                testsize=args.input_size,
                                word_length=args.word_len)
        
        test_sampler = data.distributed.DistributedSampler(test_data, shuffle=False)
        
        test_loader = data.DataLoader(test_data,
                                    batch_size=args.batch_size_val,
                                    shuffle=False,
                                    num_workers=args.workers_val,
                                    pin_memory=True,
                                    sampler=test_sampler,
                                    drop_last=False)
        print(f"Loading {cur_dataset} done.")
    

    # inference
    results = test(test_loader, model, cur_dataset, args)
    print(results)

if __name__ == '__main__':
    main()
