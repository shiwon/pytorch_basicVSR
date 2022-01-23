"""
This code is based on Open-MMLab's one.
https://github.com/open-mmlab/mmediting
"""

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

def pair_random_crop_seq(hr_seq,lr_seq,patch_size,scale_factor=4):
    """crop image pair for data augment
    Args:
        hr (Tensor): hr images with shape (t, c, 4h, 4w).
        lr (Tensor): lr images with shape (t, c, h, w).
        patch_size (int): the size of cropped image
    Returns:
        Tensor, Tensor: cropped images(hr,lr)
    """
    seq_lenght=lr_seq.size(dim=0)
    gt_transformed=torch.empty(seq_lenght,3,patch_size*scale_factor,patch_size*scale_factor)
    lq_transformed=torch.empty(seq_lenght,3,patch_size,patch_size)
    i,j,h,w=T.RandomCrop.get_params(lr_seq[0],output_size=(patch_size,patch_size))
    gt_transformed=T.functional.crop(hr_seq,i*scale_factor,j*scale_factor,h*scale_factor,w*scale_factor)
    lq_transformed=T.functional.crop(lr_seq,i,j,h,w)
    return gt_transformed,lq_transformed

def pair_random_flip_seq(sequence1,sequence2,p=0.5,horizontal=True,vertical=True):
    """flip image pair for data augment
    Args:
        sequence1 (Tensor): images with shape (t, c, h, w).
        sequence2 (Tensor): images with shape (t, c, h, w).
        p (float): probability of the image being flipped.
            Default: 0.5
        horizontal (bool): Store `False` when don't flip horizontal
            Default: `True`.
        vertical (bool): Store `False` when don't flip vertical
            Default: `True`.
    Returns:
        Tensor, Tensor: cropped images
    """
    T_length=sequence1.size(dim=0)
    # Random horizontal flipping
    hfliped1=sequence1.clone()
    hfliped2=sequence2.clone()
    if horizontal and random.random() > 0.5:
        hfliped1 = T.functional.hflip(sequence1)
        hfliped2 = T.functional.hflip(sequence2)

    # Random vertical flipping
    vfliped1=hfliped1.clone()
    vfliped2=hfliped2.clone()
    if vertical and random.random() > 0.5:
        vfliped1 = T.functional.vflip(hfliped1)
        vfliped2 = T.functional.vflip(hfliped2)
    return vfliped1,vfliped2

def pair_random_transposeHW_seq(sequence1,sequence2,p=0.5):
    """crop image pair for data augment
    Args:
        sequence1 (Tensor): images with shape (t, c, h, w).
        sequence2 (Tensor): images with shape (t, c, h, w).
        p (float): probability of the image being cropped.
            Default: 0.5
    Returns:
        Tensor, Tensor: cropped images
    """
    T_length=sequence1.size(dim=0)
    transformed1=sequence1.clone()
    transformed2=sequence2.clone()
    if random.random() > 0.5:
        transformed1=torch.transpose(sequence1,2,3)
        transformed2=torch.transpose(sequence2,2,3)
    return transformed1,transformed2

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
    def __init__(self, gt_dir, lq_dir,scale_factor=4, patch_size=256, num_input_frames=10, is_test=False,max_keys=270,filename_tmpl='{:08d}.png'):
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
        self.filename_tmpl=filename_tmpl
    
    def transform(self,gt_seq,lq_seq):
        gt_transformed,lq_transformed=pair_random_crop_seq(gt_seq,lq_seq,patch_size=self.patch_size)
        gt_transformed,lq_transformed=pair_random_flip_seq(gt_transformed,lq_transformed,p=0.5)
        gt_transformed,lq_transformed=pair_random_transposeHW_seq(gt_transformed,lq_transformed,p=0.5) 
        return gt_transformed,lq_transformed

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self,idx):
        gt_sequence, lq_sequence = generate_segment_indices(self.gt_seq_paths[idx],self.lq_seq_paths[idx],num_input_frames=self.num_input_frames,filename_tmpl=self.filename_tmpl)
        if not self.is_test:
            gt_sequence, lq_sequence = self.transform(gt_sequence,lq_sequence)
        return gt_sequence,lq_sequence
