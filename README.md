CLIPCOD

TodoList
- [x]  Prepare image description containing attribution informations from `tools\desc_generator.py`  

- [x]  extract multiple feature maps from Vit.
- [x]  textual encoding done.
- [x]  neck design for vis and text feature fusion
- [ ]  loss adjustment.
- [ ]  blank version without fixation done.
- [ ]  adding fixation

Dataset setting

    dataset
      --TestDataset
          --CAMO
          --CHAMELEON
          --COD10K
          --NC4k
      --TrainDataset
          --Desc
          --Fix
          --GT
          --Imgs

put `ViT-L-14-336px.pt` in 

    pretrain
        --ViT-L-14-336px.pt