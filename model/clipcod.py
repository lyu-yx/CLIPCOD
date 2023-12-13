import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from utils.losses import structure_loss, kl_div_loss, correlation_coefficient_loss
from .layers import FPN, Projector, TransformerDecoder, FixationEstimation


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
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.feats_layer_num).float()
        # Multi-Modal FPN
        self.fix_encoder = FixationEstimation()
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
        self.fixation_weight = cfg.fixation_weight

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
        fix = self.fix_encoder(vis)  # [b, 1, 24, 24]

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
            mask_loss = structure_loss(pred, mask)
            kl_loss = kl_div_loss(pred, mask)
            cc_loss = correlation_coefficient_loss(pred, mask)
            fix_loss = kl_loss + cc_loss
            total_loss = fix_loss * self.fixation_weight + mask_loss
            return pred.detach(), mask, total_loss, fix_loss, kl_loss, cc_loss
        else:
            return pred.detach()