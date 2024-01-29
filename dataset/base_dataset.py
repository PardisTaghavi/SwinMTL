# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import cv2
import matplotlib.pyplot as plt

def get_dataset(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    dataset_lib = importlib.import_module(
        '.' + dataset_name, package='dataset')
     
    print(dataset_name, dataset_lib)
    dataset_abs = getattr(dataset_lib, dataset_name)
    return dataset_abs(**kwargs)


class BaseDataset(Dataset):
    def __init__(self, crop_size):
        
        self.count = 0
        
        basic_transform = [
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]

        physical_transform = [
            A.HorizontalFlip(),
            A.RandomCrop(crop_size[0], crop_size[1])]
        
        self.basic_transform = basic_transform   
        self.physical_transform = physical_transform
        self.to_tensor = transforms.ToTensor() 

    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT

    def augment_training_data(self, image, depth, seg):
        H, W, C = image.shape
        #print(image.shape, depth.shape, seg.shape,"data augmentation")
        #print(self.count ,"==============count")

        '''if self.count % 8 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75
            gamma = random.randint(0,1)      #gamma random 0 or 1

            #print(alpha, beta, gamma, "========alfa beta gama==========")
            l = int(alpha * W) 
            w = int(max((W - alpha * W) * beta * p, 1))
            #print(l, w, "=======l w===========")

            if gamma == 0:

                image[:, l:l+w, 0] = depth[:, l:l+w]
                image[:, l:l+w, 1] = depth[:, l:l+w]
                image[:, l:l+w, 2] = depth[:, l:l+w]
                #seg[:, l:l+w] = depth[:, l:l+w]
           
            elif gamma == 1:
                image[:, l:l+w, 0] = seg[:, l:l+w]
                image[:, l:l+w, 1] = seg[:, l:l+w]
                image[:, l:l+w, 2] = seg[:, l:l+w]
                #depth[:, l:l+w] = seg[:, l:l+w]'''
        #later add vertical cut and horizontal cut
        
        #additional_targets = { 'depth': 'image'} #, 'seg': 'image'}
        aug = A.Compose(transforms=self.basic_transform)#,  additional_targets=additional_targets)
        augmented = aug(image=image)#, depth=depth)#, seg=seg)
        image = augmented['image']
       
        #crop segmentation mask same as image
        additional_targets2 = { 'depth': 'image', 'seg': 'image'}
        aug2=A.Compose(transforms=self.physical_transform, additional_targets=additional_targets2)
        augmented2 = aug2(image=image, depth=depth, seg=seg)  
        
        image = augmented2['image']
        depth = augmented2['depth']
        seg = augmented2['seg']
        # show image, depth, seg in subplots and pause for 1 second
        '''plt.subplot(131)
        plt.imshow(image)
        plt.subplot(132)
        plt.imshow(depth)
        plt.subplot(133)
        plt.imshow(seg)
        plt.show()
        plt.pause(1000)
        plt.close()'''

        #print(image.shape, depth.shape, seg.shape,"data augmentation")
        #print(seg,"================1")

        image = self.to_tensor(image)
        #depth = self.to_tensor(depth).squeeze()
        depth= torch.from_numpy(depth).squeeze()
        seg=torch.from_numpy(seg).squeeze()
        #seg = self.to_tensor(seg).squeeze()  #segmentation labels should not be normalized!!! 
        
        self.count += 1

        return image, depth, seg

    def augment_test_data(self, image, depth, seg):
        #to_tensor = transforms.ToTensor()
    
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()
        seg = torch.from_numpy(seg).squeeze() #so as to stop normalization of labels
        #seg = self.to_tensor(seg).squeeze()
        
        return image, depth, seg

