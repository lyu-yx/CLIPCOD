import torch
import torch.nn.functional as F

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def kl_div_loss(fix_pred, gt):
    '''
    KullbackLeibler divergence (KL) 
    '''
    # gt = gt.squeeze()
    # batch_size = fix_pred.size(0)
    # w = fix_pred.size(1)
    # h = fix_pred.size(2)

    kl_loss = torch.nn.KLDivLoss(size_average=False, reduce=False)
    fix_pred = fix_pred.squeeze(1)
    fix_pred = fix_pred.cuda(non_blocking=True)
    fix_pred = F.log_softmax(fix_pred, dim=1)
    fix_pred = fix_pred.unsqueeze(1).float()
    
    kl = kl_loss(fix_pred, gt).mean()
    
    return kl


    # sum_s_map = torch.sum(fix_pred.view(batch_size, -1), 1)
    # expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    # assert expand_s_map.size() == fix_pred.size()


    # sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    # expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    # assert expand_gt.size() == gt.size()

    # fix_pred = fix_pred / (expand_s_map * 1.0)
    # gt = gt / (expand_gt * 1.0)

    # fix_pred = fix_pred.view(batch_size, -1)
    # gt = gt.view(batch_size, -1)

    # eps = 2.2204e-16
    # result = gt * torch.log(eps + gt / (fix_pred + eps))
    # # print(torch.log(eps + gt/(s_map + eps))   )
    # return torch.mean(torch.sum(result, 1))



def correlation_coefficient_loss(pred, gt):
    '''
    Pearson’s Correlation Coefficient (CC)
    '''
    # Reshape tensors to [batch_size, -1] to flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)

    # Compute mean of each tensor along the batch dimension
    mean_pred = torch.mean(pred_flat, dim=1, keepdim=True)
    mean_gt = torch.mean(gt_flat, dim=1, keepdim=True)

    # Center the tensors by subtracting the mean
    centered_pred = pred_flat - mean_pred
    centered_gt = gt_flat - mean_gt

    # Compute the dot product of the centered tensors
    dot_product = torch.sum(centered_pred * centered_gt, dim=1)

    # Compute the product of standard deviations
    std_pred = torch.std(pred_flat, dim=1, keepdim=True)
    std_gt = torch.std(gt_flat, dim=1, keepdim=True)

    # Compute Pearson's correlation coefficient
    correlation_coefficient = dot_product / (std_pred * std_gt + 1e-8)  # Add small epsilon to avoid division by zero

    return correlation_coefficient.mean().item()

    # batch_size = s_map.size(0)
    # w = s_map.size(1)
    # h = s_map.size(2)

    # mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    # std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    # mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    # std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    # s_map = (s_map - mean_s_map) / std_s_map
    # gt = (gt - mean_gt) / std_gt

    # ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    # aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    # bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    # return torch.mean(ab / (torch.sqrt(aa*bb)))