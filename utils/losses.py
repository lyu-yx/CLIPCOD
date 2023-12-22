import torch
import torch.nn as nn
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

def attribuion_loss(pred_attr, target_attr):
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    attr_ce_loss = criterion_ce(pred_attr, target_attr)
    log_probs = F.log_softmax(pred_attr, dim=1)
    attr_kl_loss = criterion_kl(log_probs, target_attr)
    return attr_ce_loss, attr_kl_loss

def kl_div_loss(s_map, gt):
    '''
    KullbackLeibler divergence (KL) 
    '''
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    log_s_map = F.log_softmax(s_map, dim=1)  # Applying log-softmax to s_map
    kl_loss = criterion_kl(log_s_map, gt)

    return kl_loss

def correlation_coefficient_loss(s_map, gt):
    '''
    Pearson’s Correlation Coefficient (CC)
    '''
    s_map = s_map.squeeze()
    gt = gt.squeeze()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    return -torch.mean(ab / (torch.sqrt(aa*bb)))

def cosine_similarity_loss(text_features, visual_features):
    '''
    Cosine Similarity loss
    '''
    # Normalize the features to have unit norm
    text_features = F.normalize(text_features, p=2, dim=1)
    visual_features = F.normalize(visual_features, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(text_features * visual_features, dim=1)
    
    # Since we want to maximize similarity, we minimize the negative similarity
    return -cosine_sim.mean()