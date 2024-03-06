****** repo will be further modified... ******



[![Static Badge](https://img.shields.io/badge/Project_Page-https%3A%2F%2Fpardistaghavi.github.io%2FSwinMTL.html-green?style=flat)](https://pardistaghavi.github.io/SwinMTL.html)


# SwinMTL

This readme file provides an overview of the SwinMTL project, a multi-task learning framework for simultaneous depth estimation and semantic segmentation using the Swin Transformer architecture.
Project website: 
Paper: 

![qualititative](https://github.com/PardisTaghavi/SwinMTL/blob/main/results/qualititativeResults.png)


### Installation
Clone the repository: 

```git clone https://github.com/PardisTaghavi/SwinMTL.git```

Install dependencies: 

Refer to the requirements.txt file for required libraries.

### Training

```
python3 trainMulti.py --dataset cityscapes --data_path ../datasets/cityscapes --max_depth 80.0 --max_depth_eval 80.0 --backbone swin_base_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 22 22 22 11 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 2 --pretrained weights/swin_v2_base_simmim.pth --save_model --crop_h 480 --crop_w 480 --layer_decay 0.9 --drop_path_rate 0.3 --log_dir logs/
```



### Contributions:
Introduces a multi-task learning approach for joint depth estimation and semantic segmentation.
Achieves state-of-the-art performance on Cityscapes dataset.
Utilizes efficient shared encoder-decoder architecture.
Integrates novel techniques like logarithmic depth scaling and MixUp augmentation.
Enables 3D scene reconstruction through voxel map generation.

We welcome feedback and contributions to the SwinMTL project. Feel free to contact taghavi.pardis@gmail.com



----------------------------------------------------------------------------------------------
### Acknowledgments
Special thanks to the authors of the following projects for laying the foundation of this work.
Our code relies on GLPDepth([Link](https://github.com/vinvino02/GLPDepth)) and MIM-Depth-Estimation([Link](https://github.com/SwinTransformer/MIM-Depth-Estimation?tab=readme-ov-file)) with the model code sourced from SwinTransformer([Link](https://github.com/microsoft/Swin-Transformer)).


