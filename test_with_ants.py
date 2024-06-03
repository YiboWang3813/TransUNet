from datetime import datetime

import cv2 
from src import utils, metrics 
import nibabel as nib 
import numpy as np 
from networks.blocks import SpatialTransformerBlock 
import torch 

vol_shape = (192, 160, 192)

# generate transformer 
transformer = SpatialTransformerBlock(vol_shape)

# load labels 
# labels = utils.get_lpba40_labels() 
labels = utils.get_oasis1_labels() 

# load fixed volume 
# fixed = nib.load('../LPBA40_T1/test/sample_0/seg.nii')
fixed = nib.load('../data/oasis/sample_0/seg.nii.gz')
fixed = np.array(fixed.dataobj, dtype='float32')

# file_name = './log/lpba_ants.txt'
file_name = './log/oasis_ants.txt'

# generate score mat 
# num_samples = 9 
# num_areas = 12 
num_samples = 40
num_areas = 16 
scores = np.zeros((num_samples, num_areas))

print('-------------------- ANTs Test Start --------------------') 
for i in range(1, num_samples + 1): 
    print(i)

    # generate warped volume
    # moving = nib.load('../LPBA40_T1/test/sample_' + str(i) + '/seg.nii')
    # flow = nib.load('../data/lpba/sample_' + str(i) + '/out1Warp.nii.gz')
    moving = nib.load('../data/oasis/sample_' + str(i) + '/seg.nii.gz')
    flow = nib.load('../data/oasis/sample_' + str(i) + '/out1Warp.nii.gz')
    moving, flow = np.array(moving.dataobj, dtype='float32'), np.array(flow.dataobj, dtype='float32')
    moving, flow = torch.from_numpy(moving), torch.from_numpy(flow)
    flow = flow.permute(3, 4, 0, 1, 2)
    moving = moving.reshape((1, 1, *vol_shape))
    warped = transformer(moving, flow)
    warped = np.round(warped)
    warped = utils.to_array(warped, vol_shape)

    # cal dice 
    dice = metrics.dice(warped, fixed, labels)

    # record 
    scores[i - 1, :] = dice 
    
    # save 1st sample 100 idx z and flow
    if i == 1: 
        # img and mask 
        outWarped = nib.load('../data/oasis/sample_1/outWarped.nii.gz') 
        outWarped = np.array(outWarped.dataobj, dtype='float32') 
        outWarped = utils.normal(outWarped) 
        outWarped = (outWarped * 255).astype(np.uint8) 
        warped = warped.astype(np.uint8) 
        idx = 120 
        img_ants = outWarped[:, :, idx] 
        mask_ants = warped[:, :, idx] 
        cv2.imwrite('./fig/img_ants.png', img_ants) 
        cv2.imwrite('./fig/mask_ants.png', mask_ants) 
        # flow 
        flow = flow.numpy().reshape((3, *vol_shape)) 
        flowX, flowY, flowZ = flow[0, ...], flow[1, ...], flow[2, ...] 
        sliceX, sliceY, sliceZ = flowX[:, :, idx], flowY[:, :, idx], flowZ[:, :, idx] 
        sliceX, sliceY, sliceZ = utils.normal(sliceX), utils.normal(sliceY), utils.normal(sliceZ) 
        sliceX = (sliceX * 255).astype(np.uint8) 
        sliceY = (sliceY * 255).astype(np.uint8) 
        sliceZ = (sliceZ * 255).astype(np.uint8) 
        sliceXYZ = cv2.merge([sliceX, sliceY, sliceZ]) 
        cv2.imwrite('./fig/flow_ants.png', sliceXYZ) 
print('-------------------- ANTs Test End --------------------')

# save into local disk 
with open(file_name, 'a') as f: 
    for j in range(num_areas): 
        for i in range(num_samples): 
            message = '%.4f' % scores[i, j] + '\n'
            f.write(message)
        f.write('\n')
f.close() 


