# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
#  moddified by Pardis Taghavi (taghavi.pardis@gmail.com)
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np

from dataset.base_dataset import BaseDataset
from dataset.labels import labels
import torch
class cityscapes(BaseDataset):
    def __init__(self, data_path, filenames_path='./dataset/filenames/',
                 split='test', label_map='trainId', crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)

        self.scale_size = scale_size
        self.data_path = data_path 
        self.label_map = label_map # 'id', 'trainId', 'categoryId'
        self.split=split

        self.image_path_list = []
        self.depth_path_list = []
        self.seg_path_list = []

        #get label mappping from cityscapes
        self.id_2_train = {}
        self.id_2_cat = {}
        self.train_2_id = {}
        self.id_2_name = {-1 : 'unlabeled'}
        self.trainid_2_name = {19 : 'unlabeled'} #{255 : 'unlabeled', -1 : 'unlabeled'}

        for lbl in labels:
            self.id_2_train.update({lbl.id : lbl.trainId})
            self.id_2_cat.update({lbl.id : lbl.categoryId})
            if lbl.trainId != 19: #(lbl.trainId > 0) and (lbl.trainId != 255):
                self.trainid_2_name.update({lbl.trainId : lbl.name})
                self.train_2_id.update({lbl.trainId : lbl.id})
            if lbl.id > 0:
                self.id_2_name.update({lbl.id : lbl.name})


        txt_path = os.path.join(filenames_path, 'cityscapes')
        if self.split=='train':
            txt_path += '/cityscapes_train.txt'
        elif self.split =='val':
            txt_path += '/cityscapes_val.txt'
        elif self.split=="test":
            txt_path += '/cityscapes_test.txt'

        self.filenames_list = self.readTXT(txt_path)
        if self.split=='train' : phase = 'train' 
        if self.split=='test'  : phase =' test' 
        if self.split=='val'   : phase= 'val' 
        print("Dataset: cityscapes Depth")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path =   self.data_path + self.filenames_list[idx].split(' ')[0] #left
        seg_path=    self.data_path + self.filenames_list[idx].split(' ')[1] #mask
        depth_path = self.data_path + self.filenames_list[idx].split(' ')[2] #depth
        
        filename = depth_path.split("/")[-1]
        filename = filename.split("_crestereo_depth.png")[0]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype('float32') 

        dmin= 0.0001
        dmax= 80
        #normalized in log space
        depth = np.log(depth.clip(dmin,dmax)/dmin)/np.log(dmax/dmin)                
        seg   = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED) 

        # get label id
        if self.label_map == 'id':
            seg[seg==-1] == 0
        elif self.label_map == 'trainId':
            #print("id to trainid")
            for _id, train_id in self.id_2_train.items():
                seg[seg==_id] = train_id
             
        elif self.label_map == 'categoryId':
            for _id, train_id in self.id_2_cat.items():
                seg[seg==_id] = train_id

        if self.split=='train':
            #print("adding train aug")
            image, depth, seg = self.augment_training_data(image, depth, seg)
            
        elif self.split=='val' or self.split=='test':
            #print("test aug")
            image, depth, seg = self.augment_test_data(image, depth, seg)
            
        return {'image': image, 'depth': depth, 'seg': seg, 'filename' : filename}
        
        
        
        
