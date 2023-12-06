
from glob import glob
import os
from tqdm import tqdm

data_root = "dataset/TestDataset/NC4K/Imgs"
target_pth = "dataset/TestDataset/NC4K/Desc"
if not os.path.exists(target_pth):
        os.makedirs(target_pth)
desc = "Camouflaged object."

file_list = glob(os.path.join(data_root, "*.jpg"))
for file in tqdm(file_list):
    filename = os.path.basename(file)
    with open(os.path.join(target_pth, filename + ".txt"), 'a') as file:
        file.write(desc)

    # with open(os.path.join(target_pth, filename + ".txt"), 'w') as file:
    #     file.write(filename)

