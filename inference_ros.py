import rospy
from sensor_msgs.msg import Image
import torch
from transformations import *
from models.modelMulti import GLPDepth
from labels import labels
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import ros_numpy
from labels import labels
import argparse

    

ckpt_dir='/home/avalocal/thesis23/MIM-Depth-Estimation/logs/2024-02-10_12-10-16_cityscapes_4_swin_v2_base_simmim_deconv3_32_2_480_480_00005_3e-05_09_005_200_22_22_22_11_2_2_18_2/epoch_200_model.ckpt'

# Function to load the GLPDepth model
def load_glp_depth_model(self, args):
    model = GLPDepth(args=args).to(self.device)
    load_model(ckpt_dir, model)  # Load weights
    model.eval()
    return model

#load weights
def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')

    state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value
    #print(weights.keys(), 'loaded...')
    model.load_state_dict(weights)
    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)
    #print(ckpt, 'loaded....')

def parse_opt():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=float, default=80, help="Maximum depth value")
    parser.add_argument("--backbone", type=str, default="swin_base_v2", help="Backbone model")
    parser.add_argument("--depths", type=list, default=[2, 2, 18, 2], help="Number of layers in each stage")
    parser.add_argument("--num_filters", type=list, default=[32, 32, 32], help="Number of filters in each stage")
    parser.add_argument("--deconv_kernels", type=list, default=[2, 2, 2], help="Kernel size for deconvolution")
    parser.add_argument("--window_size", type=list, default=[22, 22, 22, 11], help="Window size for MIM")
    parser.add_argument("--pretrain_window_size", type=list, default=[12, 12, 12, 6], help="Window size for pretraining")
    parser.add_argument("--use_shift", type=list, default=[True, True, False, False], help="Use shift operation")
    parser.add_argument("--shift_size", type=int, default=16, help="Shift size")
    parser.add_argument("--save_visualization", type=bool, default=False, help="Save visualization")
    parser.add_argument("--flip_test", type=bool, default=False, help="Flip test")
    parser.add_argument("--shift_window_test", type=bool, default=False, help="Shift window test")
    parser.add_argument("--num_classes", type=int, default=20, help="Number of classes")
    parser.add_argument("--drop_path_rate", type=float, default=0.3, help="Drop path rate")
    parser.add_argument("--pretrained", type=str, default="/home/avalocal/thesis23/MIM-Depth-Estimation/weights/swin_v2_base_simmim.pth", help="Pretrained weights")
    parser.add_argument("--save_model", type=bool, default=False, help="Save model")
    parser.add_argument("--crop_h", type=int, default=224, help="Crop height")
    parser.add_argument("--crop_w", type=int, default=224, help="Crop width")
    parser.add_argument("--layer_decay", type=float, default=0.9, help="Layer decay")
    parser.add_argument("--use_checkpoint", type=bool, default=True, help="Use checkpoint")
    parser.add_argument("--num_deconv", type=int, default=3, help="Number of deconvolution layers")
    return parser.parse_args()


class PerceptionNode:

    def __init__(self, args):

        self.sub_image = rospy.Subscriber("/resized/camera_fl/image_color", Image, self.callback)
        self.pub_depth = rospy.Publisher("/perception/depth", Image, queue_size=10)
        self.pub_seg = rospy.Publisher("/perception/segmentation", Image, queue_size=10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        self.model = None
        self.model = load_glp_depth_model(self, args)
        #wait for the model to load

    def callback(self, image):
        
        img = ros_numpy.numpify(image)  # w, h, 3
        img = cv2.resize(img, (640, 640))  #
        im = img.transpose(2, 0, 1)
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255
        if not self.model:
            #pass if the model is not loaded yet
            return
        
        
        with torch.no_grad():
            pred = self.model(im.unsqueeze(0).to(self.device))
            
            pred_depth = pred['pred_d'].squeeze(0)
            pred_seg = pred['pred_seg']#.squeeze(0)
            pred_seg = torch.argmax(pred_seg, dim=1)

        pred_depth  = pred_depth.permute(1,2,0)
        pred_seg = pred_seg.permute(1,2,0)
        pred_seg = pred_seg.cpu().numpy()
        pred_seg = get_color_mask(pred_seg, labels, id_type='trainId')
        pred_depth = pred_depth.cpu().numpy()
        
        pred_depth = ros_numpy.msgify(Image, pred_depth, encoding="32FC1")
        pred_seg = ros_numpy.msgify(Image, pred_seg.astype(np.uint8)
                                    , encoding="rgb8")

        self.pub_depth.publish(pred_depth) 
        self.pub_seg.publish(pred_seg)



def get_color_mask(mask, labels, id_type='id'):
    try:
        h, w = mask.shape
    except ValueError:
        mask = mask.squeeze(-1)
        h, w = mask.shape

    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    if id_type == 'id':
        for lbl in labels:
            color_mask[mask == lbl.id] = lbl.color
    elif id_type == 'trainId':
        for lbl in labels:
            if (lbl.trainId != 255) | (lbl.trainId != -1):
                color_mask[mask == lbl.trainId] = lbl.color
    return color_mask



if __name__ == "__main__":
    args = parse_opt()
    rospy.init_node("perception_node", anonymous=True)
    perception_node = PerceptionNode(args)
    rospy.spin()