import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from utils.losses import structure_loss, kl_div_loss, correlation_coefficient_loss, cosine_similarity_loss
from .layers import FPN, Projector, TransformerDecoder, FixationEstimation, FeatureFusionModule, ProjectionNetwork, pool_visual_features, d3_to_d4


class CLIPCODBLANK(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.feats_layer_num).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim , 3)
        

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        ''' 
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: list: 3 x [b, 576, 768]
        # word: b, 77, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)           # list: 3 x [b, 576, 768]
        word, state = self.backbone.encode_text(word)   # [b, 77, 768] [b, 768]

        # b, c, 24, 24
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)  # [b, c, 24, 24]

        pred = self.proj(fq, state) # [b, c, 96, 96]

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = structure_loss(pred, mask)
            return pred.detach(), mask, loss
        else:
            return pred.detach()


class CLIPCOD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
# para init
        self.fixation_weight = cfg.fixation_weight
        self.kl_weight = cfg.kl_weight
        self.cc_weight = cfg.cc_weight
        self.consistency_weight = cfg.consistency_weight
        
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.feats_layer_num).float()
        
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        
        # fixation prediction
        self.fix_encoder = FixationEstimation(cfg.fix_embed_dim, 
                                              cfg.fix_num_head,
                                              cfg.fix_num_layers,
                                              cfg.fix_dim_ffn,
                                              cfg.fix_out_size)
        
        # visual modal fusion
        self.visual_fusion = FeatureFusionModule(embed_dim = 768)

        # projector for consistency loss
        self.word_proj = ProjectionNetwork(input_dim=cfg.word_dim, proj_dim=cfg.projector_dim)
        self.vis_proj = ProjectionNetwork(input_dim=cfg.vis_dim, proj_dim=cfg.projector_dim)

        # multimodal decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim, 3)
         
    def forward(self, img, desc, img_gt, fix_gt=None):
        '''
            img: b, 3, h, w
            desc: b, worddim, words
            state: b, words
            img_gt: b, 1, h, w
            fix_gt: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(desc).masked_fill_(desc == 0, 1).bool()

        # vis: list: 3 x [b, 576, 768]
        # word: b, 77, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)           # list: 3 x [b, 576, 768]
        desc, state = self.backbone.encode_text(desc)   # [b, 77, 768] [b, 768]

        # vis branch
        fix_out, fix_tensor = self.fix_encoder(vis)  # [b, 1, 96, 96]  [b, 576, 768]
        vis_feats = self.visual_fusion(vis, fix_tensor) # [b, 576, 768]
        
        # for consistency loss
        vis_proj = pool_visual_features(vis_feats, pooling_type='max') # [b, 576, 768] -> [b, 768]
        vis_proj = self.vis_proj(vis_proj) # [b, 768] -> [b, 512]
        word_proj = self.word_proj(state)   # [b, 768] -> [b, 512]

        # multimodal branch 
        multimodal_feats = d3_to_d4(self, vis_feats)
        b, c, h, w = multimodal_feats.size() # [b, out_channels[1], 24, 24]
        multimodal_feats = self.decoder(multimodal_feats, desc, pad_mask)  # desc should change to while img description
        multimodal_feats = multimodal_feats.reshape(b, c, h, w)  # [b, c, 24, 24]

        
        pred = self.proj(multimodal_feats, state) # [b, c, 96, 96]

        if self.training:
            # resize mask
            if pred.shape[-2:] != img_gt.shape[-2:]:
                img_gt = F.interpolate(img_gt, pred.shape[-2:], mode='nearest').detach()
                fix_gt = F.interpolate(fix_gt, fix_out.shape[-2:], mode='nearest').detach()
            
            # normalization 
            img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-8)
            fix_gt = (fix_gt - fix_gt.min()) / (fix_gt.max() - fix_gt.min() + 1e-8)
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            fix_out = (fix_out - fix_out.min()) / (fix_out.max() - fix_out.min() + 1e-8)

            mask_loss = structure_loss(pred, img_gt)
            kl_loss = kl_div_loss(fix_out, fix_gt)
            cc_loss = correlation_coefficient_loss(fix_out, fix_gt)
            fix_loss = kl_loss * self.kl_weight + cc_loss * self.cc_weight
            consistency_loss = cosine_similarity_loss(vis_proj, word_proj)
            total_loss = mask_loss + fix_loss * self.fixation_weight + consistency_loss * self.consistency_weight
            return pred.detach(), fix_out.detach(), total_loss, fix_loss, kl_loss, cc_loss, mask_loss, consistency_loss
        else:
            return pred.detach()
