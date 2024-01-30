# SwinMTL

This readme file provides an overview of the SwinMTL project, a multi-task learning framework for simultaneous depth estimation and semantic segmentation using the Swin Transformer architecture.


### Installation
Clone the repository: 

```git clone https://github.com/PardisTaghavi/SwinMTL.git```

Install dependencies: 

Refer to the requirements.txt file for required libraries.

### Contributions:
Introduces a multi-task learning approach for joint depth estimation and semantic segmentation.
Achieves state-of-the-art performance on Cityscapes dataset.
Utilizes efficient shared encoder-decoder architecture.
Integrates novel techniques like logarithmic depth scaling and MixUp augmentation.
Enables 3D scene reconstruction through voxel map generation.

We welcome feedback and contributions to the SwinMTL project. Feel free to contact taghavi.pardis@gmail.com


![qualititative](https://github.com/PardisTaghavi/SwinMTL/blob/main/results/qualititativeResults.png)

----------------------------------------------------------------------------------------------
### Acknowledgments
Special thanks to the authors of the following projects for laying the foundation of this work.
Our code relies on GLPDepth([Link](https://github.com/vinvino02/GLPDepth)) and MIM-Depth-Estimation([Link](https://github.com/SwinTransformer/MIM-Depth-Estimation?tab=readme-ov-file)) with the model code sourced from SwinTransformer([Link](https://github.com/microsoft/Swin-Transformer)).


