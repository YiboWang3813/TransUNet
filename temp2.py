import nibabel as nib 
import numpy as np 
import cv2 
from src import utils 
import scipy.io as io 


# flow = nib.load('../data/oasis/sample_1/out1Warp.nii.gz') 
# flow = np.array(flow.dataobj, dtype='float32') 
# flow = np.transpose(flow, [3, 4, 0, 1, 2])
# flowX, flowY, flowZ = flow[0, 0, ...], flow[0, 1, ...], flow[0, 2, ...] 

# idx = 100 
# sliceX, sliceY, sliceZ = flowX[:, :, idx], flowY[:, :, idx], flowZ[:, :, idx] 

# sliceX, sliceY, sliceZ = utils.normal(sliceX), utils.normal(sliceY), utils.normal(sliceZ) 
# sliceX = (sliceX * 255).astype(np.uint8) 
# sliceY = (sliceY * 255).astype(np.uint8) 
# sliceZ = (sliceZ * 255).astype(np.uint8) 

# sliceXYZ = cv2.merge([sliceX, sliceY, sliceZ]) 
# cv2.imwrite('./fig/flow1.png', sliceXYZ) 
 
# X = nib.load('../data/oasis/sample_1/outWarped.nii.gz')
# X = X.dataobj 
# print(X.shape) 

idx = 120 
fixed = io.loadmat('../OASIS1_T1/test/sample_0/norm.mat')['norm']
# fixed = utils.normal(fixed) 
fixed = (fixed.astype(np.float32) * 255.0).astype(np.uint8)
fixed_seg = io.loadmat('../OASIS1_T1/test/sample_0/seg.mat')['seg'] 
fixed_seg = fixed_seg.astype(np.uint8) 

moving = io.loadmat('../OASIS1_T1/test/sample_1/norm.mat')['norm']
moving = (moving.astype(np.float32) * 255.0).astype(np.uint8)
moving_seg = io.loadmat('../OASIS1_T1/test/sample_1/seg.mat')['seg'] 
moving_seg = moving_seg.astype(np.uint8) 

fixed = fixed[:, :, idx] 
fixed_seg = fixed_seg[:, :, idx] 
moving = moving[:, :, idx] 
moving_seg = moving_seg[:, :, idx] 

cv2.imwrite('./fig/img_fixed.png', fixed) 
cv2.imwrite('./fig/mask_fixed.png', fixed_seg) 
cv2.imwrite('./fig/img_moving.png', moving) 
cv2.imwrite('./fig/mask_moving.png', moving_seg)  