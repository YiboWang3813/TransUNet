import torch 
from scipy import io 
from src import utils 
import cv2 as cv
import numpy as np 
from models import main_model, morph_model, cyclemorph_model, vtn_model 
from options.general_options import GeneralOptions 
from networks.blocks import SpatialTransformerBlock 


opt = GeneralOptions().parse() 
model = morph_model.MorphModel(opt)
transformer = SpatialTransformerBlock(opt.test_size) 

# load fixed and moving volume  
fixed_path = opt.dataset_dir + '/' + 'test' + '/' + 'sample_0' + '/' + 'norm.mat'
fixed = io.loadmat(fixed_path)['norm']
fixed = utils.normal(fixed)
fixed = torch.from_numpy(fixed)
fixed = fixed.reshape((1, 1, *opt.test_size))

moving_path = opt.dataset_dir + '/' + 'test' + '/' + 'sample_1' + '/' + 'norm.mat'
moving = io.loadmat(moving_path)['norm']
moving = utils.normal(moving)
moving = torch.from_numpy(moving)
moving = moving.reshape((1, 1, *opt.test_size))

# model.set_input_batch({'moving': moving, 'fixed': fixed})
model.set_input_batch({'moving': moving, 'fixed': fixed})
model.test_on_batch() 
flow = model.get_test_results()['flow']

warped = transformer(moving, flow)
[moving, fixed, warped] = [utils.to_array(x, opt.test_size) for x in [moving, fixed, warped]]

# save figs 
idx = 100 
fixed = np.array(fixed * 255, dtype='int32')
moving = np.array(moving * 255, dtype='int32')
warped = np.array(warped * 255, dtype='int32')

cv.imwrite('./fig/fixed.jpg', fixed[:, :, idx])
cv.imwrite('./fig/moving.jpg', moving[:, :, idx])
cv.imwrite('./fig/warped.jpg', warped[:, :, idx])