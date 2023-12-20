import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

def pool_visual_features(visual_features, pooling_type='max'):
    """
    Pool the 3D visual features to 2D.
    visual_features: Tensor of shape [b, 576, 768]
    pooling_type: 'max' or 'avg'
    """
    if pooling_type == 'max':
        pooled, _ = torch.max(visual_features, dim=1)
    elif pooling_type == 'avg':
        pooled = torch.mean(visual_features, dim=1)
    else:
        raise ValueError("Unsupported pooling type. Choose 'max' or 'avg'.")
    return pooled
# def create_graph(fixation_pred, vit_features):
#     batch_size, _, num_nodes = fixation_pred.shape
#     _, num_features, _ = vit_features.shape

#     # Node features
#     fixation_nodes = fixation_pred.view(batch_size * num_nodes, -1)
#     vit_nodes = vit_features.view(batch_size * num_nodes, num_features)
#     x = torch.cat([fixation_nodes, vit_nodes], dim=0)

#     # Edge Index
#     src = torch.arange(batch_size * num_nodes).repeat_interleave(num_nodes)
#     dest = torch.arange(batch_size * num_nodes, 2 * batch_size * num_nodes).repeat(num_nodes)
#     edge_index = torch.stack([src, dest], dim=0)

#     return Data(x=x, edge_index=edge_index)

class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            conv_layer(in_channel, out_channel, 3, padding=1),
            conv_layer(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


# class GNNFusionModel(nn.Module):
#     def __init__(self, num_features, hidden_dim=2048, output_dim=576*768):  # Example output_dim for classification
#         super(GNNFusionModel, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = global_mean_pool(x, batch)
#         return self.fc(x)

# class FixationEstimation(nn.Module):
#     def __init__(self, in_channels):
#         super(FixationEstimation, self).__init__()
#         self.reduce0 = DimensionalReduction(in_channels[0], 256) #  x0 -> x2 shallower to deeper
#         self.reduce1 = DimensionalReduction(in_channels[1], 256) #  1024/768 worddim
#         self.reduce2 = DimensionalReduction(in_channels[2], 256)
#         self.shallow_fusion = nn.Sequential(conv_layer(in_channels[0] + 256, 256, 3, padding=1))
#         self.deep_fusion = nn.Sequential(
#             conv_layer(in_channels[1] + 256, 256, 3, padding=1),
#             conv_layer(256, 128, 3, padding=1))
#         self.out = nn.Conv2d(128, 1, 1)
#         self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
    
#     def forward(self, x):
#         # size = x[0].size()[2:]   # x: 3*[b, 576, 768]
#         x0 = d3_to_d4(self, x[0])
#         x0 = self.reduce0(x0)      # [b, 768, 24, 24] -> [b, 256, 24, 24]
#         x1 = d3_to_d4(self, x[1])  # [b, 768, 24, 24]
#         out = self.shallow_fusion(torch.cat((x0, x1), dim=1)) # [b, 768+256, 24, 24] -> [b, 256, 24, 24]
#         # x1 = self.reduce1(x1)
#         x2 = d3_to_d4(self, x[2])  # [b, 768, 24, 24]
#         mid_f = self.deep_fusion(torch.cat((out, x2), dim=1))   # [b, 768+256, 24, 24] -> [b, 256, 24, 24]
#         fix_pred = self.out(mid_f)  # [b, 256, 24, 24] -> [b, 1, 24, 24]
#         fix_pred = self.upsample(fix_pred)
#         return fix_pred


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, features):
        kv = torch.cat(features, dim=0)
        query = features[2]
        attn_output, _ = self.attention(query=query, key=kv, value=kv)
        return attn_output

class FixationEstimation(nn.Module):
    def __init__(self, embed_dim, num_heads, num_decoder_layers, dim_feedforward, output_map_size):
        super(FixationEstimation, self).__init__()
        self.fusion = CrossAttentionFusion(embed_dim, num_heads)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.intermediate_linear = nn.Linear(embed_dim, output_map_size * output_map_size)
        self.reshape_size = output_map_size

    def forward(self, feature_list):
        fused_features = self.fusion(feature_list)
        fused_features = fused_features.permute(1, 0, 2)  # Shape: [sequence_length, batch_size, feature_size]

        memory = torch.zeros(fused_features.size()).to(fused_features.device)
        output = self.transformer_decoder(fused_features, memory)

        # Reshape and project the output to the desired fixation map size
        output = self.intermediate_linear(output)
        output = output.view(-1, 1, self.reshape_size, self.reshape_size)  # Shape: [b, 1, 96, 96]

        return output

class VisualGateFusion(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Convolutional layers to process the fixation map for different feature levels
        self.conv_layers1 = self._create_conv_layers()  # for vis_features[0]
        self.conv_layers2 = self._create_conv_layers()  # for vis_features[1]
        self.conv_layers3 = self._create_conv_layers()  # for vis_features[2]

    def _create_conv_layers(self):
        # Function to create convolutional layers for gating
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*12*12, 768),
            nn.Sigmoid()
        )

    def forward(self, vis_features, fixation_map):
        b, _, _, _ = fixation_map.shape

        # Generate different gating values for each feature level
        gating_values1 = self.conv_layers1(fixation_map).view(b, 1, 768)
        gating_values2 = self.conv_layers2(fixation_map).view(b, 1, 768)
        gating_values3 = self.conv_layers3(fixation_map).view(b, 1, 768)

        # Apply the gating values to enhance features
        enhanced_feature1 = vis_features[0] * gating_values1
        enhanced_feature2 = vis_features[1] * gating_values2
        enhanced_feature3 = vis_features[2] * gating_values3

        # Combine the enhanced features with more weight on the deeper layers
        combined_feature = (enhanced_feature1 + 2 * enhanced_feature2 + 4 * enhanced_feature3) / 7

        return combined_feature


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim=None):
        super(ProjectionNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + proj_dim) // 2  # A heuristic for hidden dimension size

        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # spatical resolution times 4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim , in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        # self.c_adj = conv_layer(768, word_dim, 3, padding=1)
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 24, 24
            word: b, 768
        '''
        x = self.vis(x) # b, 128, 96, 96
        B, C, H, W = x.size()
        x = x.reshape(1, B * C, H, W) # 1, b*128, 96, 96
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b 
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 96, 96
        return out


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 24*24, b, 512
            txt: L, b, 512
            vis_pos: 24*24, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

class FPN(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection [b, 768] --> [b, 1024]
        self.txt_proj = linear_layer(in_channels[2], out_channels[2]) # linear + batch norm + relu

        # fusion 1: v5 & seq -> f_5: b, 1024, 24, 24
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)  # CBR
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 24, 24
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 24, 24
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 24, 24
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # imgs: 3 x [b, 576, 768] state: [b, 768]
        v3, v4, v5 = imgs
        v3 = d3_to_d4(self, v3) # [b, 768, 24, 24]
        v4 = d3_to_d4(self, v4)
        v5 = d3_to_d4(self, v5) 
        # fusion 1: b, 768, 24, 24
        # text projection: b, 768 -> b, 1024
        # out: b, 1024, 24, 24
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 768, 24, 24
        # out: b, 512, 24, 24
        f4 = self.f2_v_proj(v4)
        f4 = self.f2_cat(torch.cat([f4, f5], dim=1))
        # fusion 3: b, 768, 24, 24
        # out: b, 256, 24, 24
        f3 = self.f3_v_proj(v3)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: 3 * [b, 768, 24, 24]
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, out_channels[1], 24, 24
        return fq
