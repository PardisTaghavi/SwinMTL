# [IROS24] SwinMTL: Multi-Task Learning with Swin Transformer

A multi-task learning framework designed for simultaneous depth estimation and semantic segmentation using the Swin Transformer architecture.

[![Project Page](https://img.shields.io/badge/Project_Page-SwinMTL-green?style=flat)](https://pardistaghavi.github.io/SwinMTL.html) [![Paper](https://img.shields.io/badge/Paper-SwinMTL-blue?style=flat)](https://arxiv.org/abs/2403.10662) [![Papers with Code](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swinmtl-a-shared-architecture-for/multi-task-learning-on-nyuv2)](https://paperswithcode.com/sota/multi-task-learning-on-nyuv2?p=swinmtl-a-shared-architecture-for)

### News
- **[30th June] Paper Accepted at the IROS 2024 Conference 🔥🔥🔥** 


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
   chmod +x inference_ros.py
   mv ./launch/ ./..  
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
    - [here](https://drive.google.com/drive/folders/1P91LEB4PXPomcAcdYzXRw4_9TVdFpYTA?usp=sharing) access the pretrained models.
    - Download the pretrained models you need.

2. Move Pretrained Models:
    - Create a new folder named `model_zoo`  and  move the pretrained models into the model_zoo folder you created in the project directory.
    - Refer to `testLive.ipynb` for testing.
  
### ROS Launch
```bash
roslaunch SwinMTL_ROS swinmtl_launch.launch
```
<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/ros3nodes.gif" alt="Zero-shot Results" width="300">
</div>

### 3D Mapping

<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/voxelmapDemo-ezgif.com-video-to-gif-converter.gif" alt="3D Mapping Results" width="400">
</div>





### Zero-shot Results on the Kitti Dataset

<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/KittiZeroShotDemo.gif" alt="Zero-shot Results" width="300">
</div>

### Citation

If you find our project useful, please consider citing:
```bibtex
@inproceedings{taghavi2024swinmtl,
  title={SwinMTL: A shared architecture for simultaneous depth estimation and semantic segmentation from monocular camera images},
  author={Taghavi, Pardis and Langari, Reza and Pandey, Gaurav},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4957--4964},
  year={2024},
  organization={IEEE}
}
