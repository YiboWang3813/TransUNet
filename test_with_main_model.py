import time 
from datetime import datetime 
import torch
import numpy as np
from scipy import io 
from options.general_options import GeneralOptions
from models.main_model import MainModel
from src import metrics, utils
from networks.blocks import SpatialTransformerBlock
import cv2 

vol_shape = (192, 160, 192) 

opt = GeneralOptions().parse()
model = MainModel(opt)
transformer = SpatialTransformerBlock(opt.test_size)


# load fixed volume and seg template 
fixed_path = opt.dataset_dir + '/' + 'test' + '/' + 'sample_0' + '/' + 'norm.mat'
fixed_seg_path = opt.dataset_dir + '/' + 'test' + '/' + 'sample_0' + '/' + 'seg.mat'
fixed = io.loadmat(fixed_path)['norm']
fixed_seg = io.loadmat(fixed_seg_path)['seg']
fixed = utils.normal(fixed)
fixed = torch.from_numpy(fixed)
fixed = fixed.reshape((1, 1, *opt.test_size))

# load lables 
# labels = utils.get_lpba40_labels() 
labels = utils.get_oasis1_labels() 

# get time 
# file_name = './log/lpba_main_notf.txt' 
file_name = './log/oasis_main.txt'

# generate score mat 
# num_samples = 9 
# num_areas = 12  
num_samples = 40 
num_areas = 16  
scores = np.zeros((num_samples, num_areas))

print('-------------------- Main Test Start --------------------')  
for i in range(1, num_samples + 1):
    print(i)

    # load 
    sample_dir = opt.dataset_dir + '/' + 'test' + '/' + 'sample_' + str(i) 
    moving_path = sample_dir + '/' + 'norm.mat'
    moving_seg_path = sample_dir + '/' + 'seg.mat'
    moving = io.loadmat(moving_path)['norm']
    moving_seg = io.loadmat(moving_seg_path)['seg']
    moving = utils.normal(moving)

    # to tensor 
    moving = torch.from_numpy(moving)
    moving_seg = torch.from_numpy(moving_seg)

    # reshape 
    moving = moving.reshape((1, 1, *opt.test_size))
    moving_seg = moving_seg.reshape((1, 1, *opt.test_size))

    # test 
    data_in = {'moving': moving, 'fixed': fixed}
    model.set_input_batch(data_in)
    test_start_time = time.time() 
    model.test_on_batch() 
    test_end_time = time.time() 
    print('Test Time Cost: %s s' % str(test_end_time - test_start_time))
    flow = model.get_test_results()['flow']
    
    # deform 
    warped_seg = transformer(moving_seg, flow)

    # cal dice 
    warped_seg = warped_seg.numpy().reshape(opt.test_size)
    warped_seg = np.round(warped_seg)
    dice = metrics.dice(warped_seg, fixed_seg, labels)    

    # record 
    scores[i - 1, :] = dice 
    
    # save 1st sample 100 idx z and flow
    if i == 1: 
        # img and mask 
        outWarped = transformer(moving, flow) 
        outWarped = outWarped.reshape(vol_shape).numpy() 
        outWarped = utils.normal(outWarped) 
        outWarped = (outWarped * 255).astype(np.uint8) 
        warped_seg = warped_seg.astype(np.uint8) 
        idx = 80   
        img_main = outWarped[:, :, idx] 
        mask_main = warped_seg[:, :, idx] 
        cv2.imwrite('./fig/img_main.png', img_main) 
        cv2.imwrite('./fig/mask_main.png', mask_main) 
        # flow 
        flow = flow.numpy().reshape((3, *vol_shape)) 
        flowX, flowY, flowZ = flow[0, ...], flow[1, ...], flow[2, ...] 
        sliceX, sliceY, sliceZ = flowX[:, :, idx], flowY[:, :, idx], flowZ[:, :, idx] 
        sliceX, sliceY, sliceZ = utils.normal(sliceX), utils.normal(sliceY), utils.normal(sliceZ) 
        sliceX = (sliceX * 255).astype(np.uint8) 
        sliceY = (sliceY * 255).astype(np.uint8) 
        sliceZ = (sliceZ * 255).astype(np.uint8) 
        sliceXYZ = cv2.merge([sliceX, sliceY, sliceZ]) 
        cv2.imwrite('./fig/flow_main.png', sliceXYZ) 
print('-------------------- Main Test End --------------------')  

# save into local disk 
with open(file_name, 'a') as f: 
    for j in range(num_areas): 
        for i in range(num_samples): 
            message = '%.4f' % scores[i, j] + '\n'
            f.write(message)
        f.write('\n')
f.close() 

