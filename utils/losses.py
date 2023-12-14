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
    # Reshape tensors to [batch_size, -1] to flatten spatial dimensions
    pred_flat = fix_pred.view(fix_pred.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)

    # Apply softmax to obtain probabilities
    pred_probs = F.softmax(pred_flat, dim=1)
    gt_probs = F.softmax(gt_flat, dim=1)

    # Compute KL Divergence
    kl_loss = F.kl_div(torch.log(pred_probs + 1e-8), gt_probs, reduction='mean')

    return kl_loss


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

    return correlation_coefficient.mean()
