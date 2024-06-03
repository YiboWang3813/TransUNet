import nibabel as nib 
import numpy as np 
import scipy.io as io 
import torch 
from networks.blocks import SpatialTransformerBlock
from src import utils

vol_shape = (192, 160, 192)

# generate transformer 
transformer = SpatialTransformerBlock(vol_shape)

# 保存fixed和moving的灰度图和分割图
fixed = io.loadmat('../LPBA40_T1/test/sample_0/norm.mat')
fixed = fixed['norm']
io.savemat('./wap/fixed.mat', {'wap': fixed})

fixed_seg = io.loadmat('../LPBA40_T1/test/sample_0/seg.mat')
fixed_seg = fixed_seg['seg']
io.savemat('./seg/fixed.mat', {'seg': fixed_seg})

moving = io.loadmat('../LPBA40_T1/test/sample_1/norm.mat')
moving = moving['norm']
io.savemat('./wap/moving.mat', {'wap': moving})

moving_seg = io.loadmat('../LPBA40_T1/test/sample_1/seg.mat')
moving_seg = moving_seg['seg']
io.savemat('./seg/moving.mat', {'seg': moving_seg})

# 保存ANTs的灰度图和分割图 
warped = nib.load('../data/lpba/sample_1/outWarped.nii.gz')
warped = warped.dataobj 
warped = np.array(warped, dtype='float32')
io.savemat('./wap/ants.mat', {'wap': warped})

moving = nib.load('../LPBA40_T1/test/sample_1/seg.nii')
flow = nib.load('../data/lpba/sample_1/out1Warp.nii.gz')
moving, flow = np.array(moving.dataobj, dtype='float32'), np.array(flow.dataobj, dtype='float32')
moving, flow = torch.from_numpy(moving), torch.from_numpy(flow)
flow = flow.permute(3, 4, 0, 1, 2)
moving = moving.reshape((1, 1, *vol_shape))
warped = transformer(moving, flow)
warped = np.round(warped)
warped = utils.to_array(warped, vol_shape)
io.savemat('./seg/ants.mat', {'seg': warped})

