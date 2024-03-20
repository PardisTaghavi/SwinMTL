

[![Static Badge](https://img.shields.io/badge/Project_Page-SwinMTL-green?style=flat)](https://pardistaghavi.github.io/SwinMTL.html) [![Static Badge](https://img.shields.io/badge/Paper-SwinMTL-blue?style=flat)]((https://arxiv.org/abs/2403.10662))

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swinmtl-a-shared-architecture-for/multi-task-learning-on-nyuv2)](https://paperswithcode.com/sota/multi-task-learning-on-nyuv2?p=swinmtl-a-shared-architecture-for)

<!--
<p>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=PardisTaghavi.SwinMTL&left_color=gray&right_color=red">
  </a>
</p>
-->


# SwinMTL

This readme file provides an overview of the SwinMTL project, a multi-task learning framework for simultaneous depth estimation and semantic segmentation using the Swin Transformer architecture.

<!--
Project website: [SwinMTL Project Page](https://pardistaghavi.github.io/SwinMTL.html)

Paper: [SwinMTL](https://arxiv.org/abs/2403.10662)
-->

 <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/results/qualititativeResults2.png" alt="qualititative" width="800"/>

### Installation
Clone the repository: 

```git clone https://github.com/PardisTaghavi/SwinMTL.git```

```cd SwinMTL```

Create conda environment and activate it:

```conda env create --file environment.yml```

```conda activate prc```


### Testing

Refer to testLive.ipynb for testing.
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1P91LEB4PXPomcAcdYzXRw4_9TVdFpYTA?usp=sharing).

### <ins>Zero-shot</ins> results on the Kitti Dataset

<img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/KittiZeroShotDemo.gif" width="350">



### Contributions:
- Introduces a multi-task learning approach for joint depth estimation and semantic segmentation.
- Achieves state-of-the-art performance on Cityscapes and NYUv2 dataset.
- Utilizes an efficient shared encoder-decoder architecture coupled with novel techniques to enhance accuracy.

We welcome feedback and contributions to the SwinMTL project. Feel free to contact taghavi.pardis@gmail.com



----------------------------------------------------------------------------------------------
### Acknowledgments
Special thanks to the authors of the following projects for laying the foundation of this work.
Our code relies on:
- GLPDepth([Link](https://github.com/vinvino02/GLPDepth))
-  MIM-Depth-Estimation([Link](https://github.com/SwinTransformer/MIM-Depth-Estimation?tab=readme-ov-file))
-  SwinTransformer([Link](https://github.com/microsoft/Swin-Transformer)).


