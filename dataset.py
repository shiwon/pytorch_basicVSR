import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image
import glob
import numpy as np
import os
import random

def generate_segment_indices(videopath1,videopath2,num_input_frames=10,filename_tmpl='{:08d}.png'):
    """generate segment function
    Args:
        videopath1,2 (str): input directory which contains sequential frames
        filename_tmpl (str): template which represents sequential frames
    Returns:
        Tensor, Tensor: Output sequence with shape (t, c, h, w)
    """
    seq_length=len(glob.glob(f'{videopath1}/*.png'))
    seq_length2=len(glob.glob(f'{videopath2}/*.png'))

    if seq_length!=seq_length2:
        raise ValueError(f'videopath1 and videopath2 must have same number of frames\nbut they have {seq_length} and {seq_length2}')
    if num_input_frames > seq_length:
        raise ValueError(f'num_input_frames{num_input_frames} must be greater than frames in {videopath1} \n and {videopath2}')
    
    start_frame_idx = np.random.randint(0, seq_length - num_input_frames)
    end_frame_idx = start_frame_idx + num_input_frames
    segment1=[read_image(os.path.join(videopath1,filename_tmpl.format(i))) / 255. for i in range(start_frame_idx,end_frame_idx)]
    segment2=[read_image(os.path.join(videopath2,filename_tmpl.format(i))) / 255. for i in range(start_frame_idx,end_frame_idx)]
    return torch.stack(segment1),torch.stack(segment2)

def pair_random_crop(hr,lr,patch_size,scale_factor=4):
    """crop image pair for data augment
    Args:
        hr (Tensor): hr image with shape (c, 4h, 4w).
        lr (Tensor): lr image with shape (c, h, w).
        patch_size (int): the size of cropped image
    Returns:
        Tensor, Tensor: cropped images(hr,lr)
    """
    i,j,h,w=T.RandomCrop.get_params(lr,output_size=(patch_size,patch_size))
    hr=T.functional.crop(hr,i*scale_factor,j*scale_factor,h*scale_factor,w*scale_factor)
    lr=T.functional.crop(lr,i,j,h,w)
    return hr,lr

def pair_random_flip(image1,image2,p=0.5,horizontal=True,vertical=True):
    """flip image pair for data augment
    Args:
        image1 (Tensor): image.
        image2 (Tensor): image.
        p (float): probability of the image being flipped.
            Default: 0.5
        horizontal (bool): Store `False` when don't flip horizontal
            Default: `True`.
        vertical (bool): Store `False` when don't flip vertical
            Default: `True`.
    Returns:
        Tensor, Tensor: cropped images
    """
    # Random horizontal flipping
    if horizontal and random.random() > 0.5:
        image1 = T.functional.hflip(image1)
        image2 = T.functional.hflip(image2)

    # Random vertical flipping
    if vertical and random.random() > 0.5:
        image1 = T.functional.vflip(image1)
        image2 = T.functional.vflip(image2)

    return image1,image2

def pair_random_transposeHW(image1,image2,p=0.5):
    """crop image pair for data augment
    Args:
        image1 (Tensor): image.
        image2 (Tensor): image.
        p (float): probability of the image being cropped.
            Default: 0.5
    Returns:
        Tensor, Tensor: cropped images
    """
    if random.random() > 0.5:
        image1=torch.transpose(image1,1,2)
        image2=torch.transpose(image2,1,2)

    return image1,image2

class REDSDataset(Dataset):
    """REDS dataset for video super resolution.
    Args:
        gt_dir (str): Path to a gt folder.
        lq_dir (str): Path to a lq folder.
        patch_size (int): the size of training image
            Default: 256
        is_test (bool): Store `True` when building test dataset.
            Default: `False`.
        max_keys (int): clip names(make keys '000' to 'max_keys:03d')
            Default: 270(make keys '000' to '270')
    """
    def __init__(self, gt_dir, lq_dir,scale_factor=4, patch_size=256, num_input_frames=10, is_test=False,max_keys=270):
        val_keys=['000', '011', '015', '020']
        if is_test:
            self.keys = [f'{i:03d}' for i in range(0, max_keys) if f'{i:03d}' in val_keys]
        else:
            self.keys = [f'{i:03d}' for i in range(0, max_keys) if f'{i:03d}' not in val_keys]
        self.gt_dir=gt_dir
        self.lq_dir=lq_dir
        self.scale_factor=scale_factor
        self.patch_size=patch_size
        self.num_input_frames=num_input_frames
        self.is_test=is_test
        self.gt_seq_paths=[os.path.join(self.gt_dir,k) for k in self.keys]
        self.lq_seq_paths=[os.path.join(self.lq_dir,k) for k in self.keys]

        gt_sequence, lq_sequence = generate_segment_indices(self.gt_seq_paths[0],self.lq_seq_paths[0],num_input_frames=self.num_input_frames)
        gt_sequence, lq_sequence = self.transform(gt_sequence,lq_sequence)
    
    def transform(self,gt_seq,lq_seq):
        seq_lenght=self.num_input_frames
        gt_transformed=torch.empty(seq_lenght,3,self.patch_size*self.scale_factor,self.patch_size*self.scale_factor)
        lq_transformed=torch.empty(seq_lenght,3,self.patch_size,self.patch_size)
        for t in range(0,seq_lenght):
            gt,lq=pair_random_crop(gt_seq[t],lq_seq[t],patch_size=self.patch_size)
            gt,lq=pair_random_flip(gt,lq,p=0.5)
            gt,lq=pair_random_transposeHW(gt,lq,p=0.5) 
            gt_transformed[t]=gt
            lq_transformed[t]=lq
        return gt_transformed,lq_transformed

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self,idx):
        gt_sequence, lq_sequence = generate_segment_indices(self.gt_seq_paths[idx],self.lq_seq_paths[idx],num_input_frames=self.num_input_frames)
        gt_sequence, lq_sequence = self.transform(gt_sequence,lq_sequence)
        return gt_sequence,lq_sequence
