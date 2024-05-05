# SwinMTL: Multi-Task Learning with Swin Transformer

Welcome to the SwinMTL project, a multi-task learning framework designed for simultaneous depth estimation and semantic segmentation using the Swin Transformer architecture.

[![Project Page](https://img.shields.io/badge/Project_Page-SwinMTL-green?style=flat)](https://pardistaghavi.github.io/SwinMTL.html) [![Paper](https://img.shields.io/badge/Paper-SwinMTL-blue?style=flat)](https://arxiv.org/abs/2403.10662) [![Papers with Code](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swinmtl-a-shared-architecture-for/multi-task-learning-on-nyuv2)](https://paperswithcode.com/sota/multi-task-learning-on-nyuv2?p=swinmtl-a-shared-architecture-for)

<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/results/qualititativeResults2.png" alt="Qualitative Results" width="800"/>
</div>

### Installation

To get started, follow these steps:

0. Only for ROS installation (otherwise skip this part)
   ```bash
   cd catkin_ws/src
   catkin_create_pkg SwinMTL_ROS std_msgs rospy
   cd ..
   catkin_make
   source devel/setup.bash
   cd src/SwinMTL_ROS/src
   git clone https://github.com/PardisTaghavi/SwinMTL.git
   mv ./Launch/ ./..  
   ```
1. Clone the repository:
    ```bash
    git clone https://github.com/PardisTaghavi/SwinMTL.git
    cd SwinMTL
    ```

2. Create a conda environment and activate it:
    ```bash
    conda env create --file environment.yml
    conda activate prc
    ```

### Testing

To run the testing for the project, follow the below steps:

1. Download Pretrained Models:
    - Click [here](https://drive.google.com/drive/folders/1P91LEB4PXPomcAcdYzXRw4_9TVdFpYTA?usp=sharing) to access the pretrained models.
    - Download the pretrained models you need.
    - Create a new folder named model_zoo in the project directory.

2. Move Pretrained Models:
    - Create a new folder named `model_zoo `
    - After downloading, move the pretrained models into the model_zoo folder you created in the project directory.
    - Refer to `testLive.ipynb` for testing.
  
### ROS Launch
```bash
roslauch SwinMTL_ROS swinmtl_launch.launch
```
### 3D Mapping


<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/voxelmapDemo-ezgif.com-video-to-gif-converter.gif" alt="3D Mapping Results" width="350">
</div>



### Zero-shot Results on the Kitti Dataset

<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/KittiZeroShotDemo.gif" alt="Zero-shot Results" width="350">
</div>

### Contributions

- Introduction of a multi-task learning approach for joint depth estimation and semantic segmentation.
- Achievement of state-of-the-art performance on Cityscapes and NYUv2 datasets.
- Utilization of an efficient shared encoder-decoder architecture coupled with novel techniques to enhance accuracy.

We welcome feedback and contributions to the SwinMTL project. Feel free to contact taghavi.pardis@gmail.com.

---

### Acknowledgments

Special thanks to the authors of the following projects for laying the foundation of this work. Our code relies on:

- [GLPDepth](https://github.com/vinvino02/GLPDepth)
- [MIM-Depth-Estimation](https://github.com/SwinTransformer/MIM-Depth-Estimation?tab=readme-ov-file)
- [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
