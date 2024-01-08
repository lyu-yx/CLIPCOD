import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import logging
from loguru import logger
from utils.dataset import tokenize
import utils.metrics as Measure
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    total_loss_meter = AverageMeter('Loss', ':2.4f')
    fix_loss_meter = AverageMeter('Fix Loss', ':2.4f')
    kl_loss_meter = AverageMeter('KL Loss', ':2.4f')
    cc_loss_meter = AverageMeter('CC Loss', ':2.4f')
    mask_loss_meter = AverageMeter('mask Loss', ':2.4f')
    consistency_loss_meter = AverageMeter('Consistency Loss', ':2.4f')
    attr_loss_meter = AverageMeter('Attr Loss', ':2.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, total_loss_meter, mask_loss_meter, fix_loss_meter, kl_loss_meter, cc_loss_meter, consistency_loss_meter, attr_loss_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(0.5)
    end = time.time()

    for i, (img, img_gt, fix_gt, overall_desc, camo_desc, attr) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        img = img.cuda(non_blocking=True)
        img_gt = img_gt.cuda(non_blocking=True)
        overall_desc = overall_desc.cuda(non_blocking=True)
        camo_desc = camo_desc.cuda(non_blocking=True)
        fix_gt = fix_gt.cuda(non_blocking=True)
        attr = attr.cuda(non_blocking=True)
        # forward
        with amp.autocast():
            pred, fix_out, total_loss, fix_loss, kl_loss, cc_loss, mask_loss, consistency_loss, attr_loss = model(img, img_gt, overall_desc, camo_desc, attr, fix_gt)

        # backward
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        dist.all_reduce(total_loss.detach())
        total_loss = total_loss / dist.get_world_size()
        total_loss_meter.update(total_loss.item(), img.size(0))

        dist.all_reduce(fix_loss.detach())
        fix_loss = fix_loss / dist.get_world_size()
        fix_loss_meter.update(fix_loss.item(), img.size(0))

        dist.all_reduce(kl_loss.detach())
        kl_loss = kl_loss / dist.get_world_size()
        kl_loss_meter.update(kl_loss.item(), img.size(0))

        dist.all_reduce(cc_loss.detach())
        cc_loss = cc_loss / dist.get_world_size()
        cc_loss_meter.update(cc_loss.item(), img.size(0))

        dist.all_reduce(mask_loss.detach())
        mask_loss = mask_loss / dist.get_world_size()
        mask_loss_meter.update(mask_loss.item(), img.size(0))

        dist.all_reduce(consistency_loss.detach())
        consistency_loss = consistency_loss / dist.get_world_size()
        consistency_loss_meter.update(consistency_loss.item(), img.size(0))

        dist.all_reduce(attr_loss.detach())
        attr_loss = attr_loss / dist.get_world_size()
        attr_loss_meter.update(attr_loss.item(), img.size(0))   


        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0]:
                wandb.log(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/total loss": total_loss_meter.val,
                        "training/fix loss": fix_loss_meter.val,
                        "training/kl loss": kl_loss_meter.val,
                        "training/cc loss": cc_loss_meter.val,
                        "training/mask loss": mask_loss_meter.val,
                        "training/consistency loss": consistency_loss_meter.val,
                        "training/attr_loss": attr_loss_meter.val
                    },
                    step=epoch * len(train_loader) + (i + 1))


def val(test_loader, model, epoch, args, shared_vars):
    """
    validation function
    """
    
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()
    metrics_dict = dict()

    model.eval()
    with torch.no_grad():
        for i, (image, gt, _, shape) in enumerate(test_loader):
            shape = (shape[1], shape[0])

            gt = F.upsample(gt, size=shape, mode='bilinear', align_corners=False)
            gt = gt.numpy().astype(np.float32).squeeze()
            gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
            
            image = image.cuda(non_blocking=True)
            res = model(image, gt)

            res = F.upsample(res, size=shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            WFM.step(pred=res*255, gt=gt*255)
            SM.step(pred=res*255, gt=gt*255)
            EM.step(pred=res*255, gt=gt*255)
            MAE.step(pred=res*255, gt=gt*255)
            
        metrics_dict.update(sm=SM.get_results()['sm'].round(3))
        metrics_dict.update(em=EM.get_results()['em']['adp'].round(3))
        metrics_dict.update(wfm=WFM.get_results()['wfm'].round(3))
        metrics_dict.update(mae=MAE.get_results()['mae'].round(3))


        cur_score = metrics_dict['sm'] + metrics_dict['em'] + metrics_dict['wfm'] - 0.5 * metrics_dict['mae']

        if epoch == 1:
            if not os.path.exists(args.model_save_path):
                os.mkdir(args.model_save_path)
            shared_vars['best_score'] = cur_score
            shared_vars['best_epoch'] = epoch
            shared_vars['best_metric_dict'] = metrics_dict.copy()
            print('[Cur Epoch: {}] Metrics (Sm={}, Em={}, Wfm={}, MAE={})'.format(
                epoch, metrics_dict['sm'], metrics_dict['em'], metrics_dict['wfm'], metrics_dict['mae']))
            logging.info('[Cur Epoch: {}] Metrics (Sm={}, Em={}, Wfm={}, MAE={})'.format(
                epoch, metrics_dict['sm'], metrics_dict['em'], metrics_dict['wfm'], metrics_dict['mae']))
            
        else:
            if cur_score > shared_vars['best_score']:
                shared_vars['best_score'] = cur_score
                shared_vars['best_epoch'] = epoch
                shared_vars['best_metric_dict'] = metrics_dict.copy()
                torch.save(model.state_dict(), args.model_save_path + 'Net_epoch_best.pth')
                print('>>> Save successfully! cur score: {}, best score: {}'.format(cur_score, shared_vars['best_score']))
                dist.all_reduce(shared_vars['best_score'], op=dist.ReduceOp.MAX)
                dist.all_reduce(shared_vars['best_epoch'])
            else:
                print('>>> Continue -> cur score: {}, best score: {}'.format(cur_score, shared_vars['best_score']))

            print('[Cur Epoch: {}] Metrics (Sm={}, Em={}, Wfm={}, MAE={})\n[Best Epoch: {}] Metrics (Sm={}, Em={}, Wfm={}, MAE={})'.format(
            epoch, metrics_dict['sm'], metrics_dict['em'], metrics_dict['wfm'], metrics_dict['mae'],
            shared_vars['best_epoch'], shared_vars['best_metric_dict']['sm'], shared_vars['best_metric_dict']['em'], 
            shared_vars['best_metric_dict']['wfm'], shared_vars['best_metric_dict']['mae']))

            logging.info('[Cur Epoch: {}] Metrics (Sm={}, Em={}, Wfm={}, MAE={})\n[Best Epoch: {}] Metrics (Sm={}, Em={}, Wfm={}, MAE={})'.format(
            epoch, metrics_dict['sm'], metrics_dict['em'], metrics_dict['wfm'], metrics_dict['mae'],
            shared_vars['best_epoch'], shared_vars['best_metric_dict']['sm'], shared_vars['best_metric_dict']['em'], 
            shared_vars['best_metric_dict']['wfm'], shared_vars['best_metric_dict']['mae']))

def test(test_loader, model, cur_dataset, args):
    """
    validation function
    """

    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()
    
    model.eval()
    with torch.no_grad():
        for i, (image, gt, name, shape) in tqdm(enumerate(test_loader)):
            
            shape = (shape[1], shape[0])
            gt = F.upsample(gt, size=shape, mode='bilinear', align_corners=False)
            gt = gt.numpy().astype(np.float32).squeeze()
            gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
            
            image = image.cuda(non_blocking=True)
            res = model(image, gt)

            res = F.upsample(res, size=shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            if args.visualize:
                cv2.imwrite(os.path.join(args.vis_dir, name[0]), res*255)
            
            WFM.step(pred=res*255, gt=gt*255)
            SM.step(pred=res*255, gt=gt*255)
            EM.step(pred=res*255, gt=gt*255)
            MAE.step(pred=res*255, gt=gt*255)
            

        sm = SM.get_results()['sm'].round(3)
        adpem = EM.get_results()['em']['adp'].round(3)
        wfm = WFM.get_results()['wfm'].round(3)
        mae = MAE.get_results()['mae'].round(3)
    
    print(f'{cur_dataset} done.')

    return {'Sm':sm, 'adpE':adpem, 'wF':wfm, 'M':mae}
