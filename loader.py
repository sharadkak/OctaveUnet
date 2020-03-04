import torch
import os 
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

class CustomDataset(Dataset):
    '''This class is meant to read the images from all the directories and return the whole dataset for 
    the purpose of Dataloader in pytorch.'
    
    params:
    mode: train or test or validate
    new_H: height fot the resized image
    new_W: width for the resized image
    transforms: boolean value to apply transforms
    '''

    def __init__(self, dir = ROOT, mode = 'train', new_H =1536, new_W =768 ,tranforms = False):
        self.dir = os.path.join(ROOT, 'dataset', mode)
        self.mode = mode
        self.images = list()
        self.target_im = list()
        self.new_W = new_W
        self.new_H = new_H
        self.classes = 4
        
        self.mapping = {
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 0,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 0,  # building
            12: 0,  # wall
            13: 0,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 0,  # pole
            18: 0,  # polegroup
            19: 0,  # traffic light
            20: 0,  # traffic sign
            21: 0,  # vegetation
            22: 0,  # terrain
            23: 2,  # sky
            24: 0,  # person
            25: 0,  # rider
            26: 3,  # car
            27: 0,  # truck
            28: 0,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 0,  # train
            32: 0,  # motorcycle
            33: 0,  # bicycle
            -1: 0  # licenseplate
        }

        self.mappingrgb = {
            0: (255, 0, 0),  # unlabeled
            1: (255, 0, 0),  # ego vehicle
            2: (255, 0, 0),  # rect border
            3: (255, 0, 0),  # out of roi
            4: (255, 0, 0),  # static
            5: (255, 0, 0),  # dynamic
            6: (255, 0, 0),  # ground
            7: (0, 255, 0),  # road
            8: (255, 0, 0),  # sidewalk
            9: (255, 0, 0),  # parking
            10: (255, 0, 0),  # rail track
            11: (255, 0, 0),  # building
            12: (255, 0, 0),  # wall
            13: (255, 0, 0),  # fence
            14: (255, 0, 0),  # guard rail
            15: (255, 0, 0),  # bridge
            16: (255, 0, 0),  # tunnel
            17: (255, 0, 0),  # pole
            18: (255, 0, 0),  # polegroup
            19: (255, 0, 0),  # traffic light
            20: (255, 0, 0),  # traffic sign
            21: (255, 0, 0),  # vegetation
            22: (255, 0, 0),  # terrain
            23: (0, 0, 255),  # sky
            24: (255, 0, 0),  # person
            25: (255, 0, 0),  # rider
            26: (255, 255, 0),  # car
            27: (255, 0, 0),  # truck
            28: (255, 0, 0),  # bus
            29: (255, 0, 0),  # caravan
            30: (255, 0, 0),  # trailer
            31: (255, 0, 0),  # train
            32: (255, 0, 0),  # motorcycle
            33: (255, 0, 0),  # bicycle
            -1: (255, 0, 0)  # licenseplate
        }

        # read all the images from directory 
        for city in sorted(os.listdir(self.dir)):
            path = os.path.join(self.dir , city)
    
            for file_name in sorted(os.listdir(path)):
                if 'gtFine_color' in file_name: self.images.append(os.path.join(path, file_name))
                elif 'gtFine_labelIds' in file_name: self.target_im.append(os.path.join(path, file_name))
                else: continue 

        assert len(self.images) == len(self.target_im), 'Images are not same as masks available'
                
    def __repr__(self):
        '''String return when class object is called '''

        string = 'Class {}\n'.format(self.__class__.__name__)
        string += ' Number of datapoints: {}\n'.format(len(self.images))
        string += ' Root location {}\n'.format(self.dir)
        string += ' Split ' + self.mode 
        string += ' \nNumber of classes ' + str(self.classes)
        return string
    
    
    def ids_to_class(self, mask):
        '''This function takes in the target_mask and convert its pixel values to custom ids as per need.
        Make sure these new ids are mapped properly in the dict self.mapping '''

        target_im = torch.zeros((mask.shape[0], mask.shape[1]), dtype = torch.uint8)
        for k in self.mapping:
            target_im[mask == k] = self.mapping[k]

        return target_im


    def mask_to_rgb(self,mask):
        """ This function encodes mask pixels as colors correspond to the ids.
        This masks could be useful incase recontructed image is to be compared against the rgb mask."""

        target_im = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype = torch.uint8)

        for k in self.mappingrgb:
            target_im[0][mask == k] = self.mappingrgb[k][0]
            target_im[1][mask == k] = self.mappingrgb[k][1]
            target_im[2][mask == k] = self.mappingrgb[k][2]

        return target_im
    
    def classes_to_rgb(self, mask):
        '''Converts a class ids encoded mask to rgb encoded mask'''
        
        class_to_ids = {v:self.mappingrgb[k] for k,v in (self.mapping).items()} 
        target_im = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype = torch.uint8)
        
        for k in class_to_ids:
            target_im[0][mask == k] = class_to_ids[k][0]
            target_im[1][mask == k] = class_to_ids[k][1]
            target_im[2][mask == k] = class_to_ids[k][2]
            
        return target_im


    def __getitem__(self, index):
        '''This function takes in the index and returns the image and masks corresponding to that. 
        Returned images and masks are resized as per specified. '''

        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.target_im[index]).convert('L')

        #resize the image 
        img = F.resize(img, size = (self.new_W, self.new_H), interpolation = Image.BILINEAR)
        mask = F.resize(mask, size = (self.new_W, self.new_H), interpolation = Image.BILINEAR)

        mask = torch.from_numpy(np.array(mask, dtype = np.uint8))
        img = F.to_tensor(img)

        # prepare masks for training purpose
        target_mask = self.ids_to_class(mask)
        maskrgb = self.mask_to_rgb(mask)
        target_mask = target_mask.long()
        maskrgb = maskrgb.long()

        # return the im, color encoded mask and class ids encoded mask
        return img, maskrgb, target_mask

    def __len__(self):
        '''Calculate length of the whole dataset in this mode.'''

        return len(self.images) 

    def tranforms(self):
    	'''Apply transformations to the training images if specified.'''

    	pass


