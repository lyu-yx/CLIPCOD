import torch


clip_model = torch.jit.load("pretrain/ViT-L-14-336px.pt", map_location="cpu").eval()


for name, module in clip_model.named_modules():
    print(name)
