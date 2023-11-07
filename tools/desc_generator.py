
from glob import glob
import os
from tqdm import tqdm

data_root = "dataset/TrainDataset/Imgs"
target_pth = "dataset/TrainDataset/Desc"

attr_root = "dataset/TrainDataset/COD_Attr/COD_train"


attr_list = os.listdir(attr_root)
attr_exp = ["Camouflaged object and the surrounding background share same color or texture. ",
            "Camouflaged object conceal in complex background. ",
            "Camouflaged object locate far from image center. ",
            "Camouflaged object have disruptive patterns around the object edges. ",
            "Camouflaged instance mimics certain species or non-living objects in their habitat. ",
            "Camouflaged object is partially occluded by other objects. ",
            "Easier camouflaged instance. ",
            "Camouflaged object occupy small part in image. "
            ]


file_list = glob(os.path.join(data_root, "*.jpg"))
occ_num = [0] * 10
max_list = []
max_cnt = 0


for file in tqdm(file_list):
    cnt = 0
    flg = 0
    filename = os.path.basename(file)
    for attr in attr_list:
        search_list = glob(os.path.join(attr_root, attr, "*.jpg"))
        contains = [s for s in search_list if filename in s]
        if contains:
            with open(os.path.join(target_pth, filename + ".txt"), 'a') as file:
                file.write(attr_exp[attr_list.index(attr)])
            flg = 1
            cnt += 1
    occ_num[cnt] += 1
    if flg==0:
        with open(os.path.join(target_pth, filename + ".txt"), 'a') as file:
            file.write("Normal Camouflaged object.")
    if cnt > max_cnt:
        max_cnt = cnt 
        max_list.append(file)
print(max_cnt)
print(max_list)    
print(occ_num)
    # with open(os.path.join(target_pth, filename + ".txt"), 'w') as file:
    #     file.write(filename)

