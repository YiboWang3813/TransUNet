import torch
import numpy as np
import scipy.io as io 
from src import utils 
from options.general_options import GeneralOptions 
from models import main_model, morph_model, vtn_model, cyclemorph_model 
from networks.blocks import SpatialTransformerBlock 


opt = GeneralOptions().parse() 
# model = main_model.MainModel(opt) 
# model = morph_model.MorphModel(opt)
model = vtn_model.VTNModel(opt)
# model = cyclemorph_model.CycleMorphModel(opt)
transformer = SpatialTransformerBlock(opt.test_size)

# load fixed volume and seg template 
fixed_path = opt.dataset_dir + '/' + 'test' + '/' + 'sample_0' + '/' + 'norm.mat'
fixed = io.loadmat(fixed_path)['norm']
fixed = utils.normal(fixed)
fixed = torch.from_numpy(fixed)
fixed = fixed.reshape((1, 1, *opt.test_size))

# load moving volume and seg template 
moving_path = opt.dataset_dir + '/' + 'test' + '/' + 'sample_1' + '/' + 'norm.mat'
moving = io.loadmat(moving_path)['norm']
moving = utils.normal(moving)
moving = torch.from_numpy(moving)
moving = moving.reshape((1, 1, *opt.test_size)) 

moving_seg_path = opt.dataset_dir + '/' + 'test' + '/' + 'sample_1' + '/' + 'seg.mat'
moving_seg = io.loadmat(moving_seg_path)['seg']
moving_seg = torch.from_numpy(moving_seg)
moving_seg = moving_seg.reshape((1, 1, *opt.test_size))

# test 
data_in = {'moving': moving, 'fixed': fixed}
model.set_input_batch(data_in)
model.test_on_batch() 
flow = model.get_test_results()['flow']

# warp 
warped = transformer(moving, flow)
warped_seg = transformer(moving_seg, flow)
warped_seg = np.round(warped_seg)

# save 
warped, warped_seg = utils.to_array(warped, opt.test_size), utils.to_array(warped_seg, opt.test_size)
io.savemat('./wap/%s.mat' % 'vtn', {'wap': warped})
io.savemat('./seg/%s.mat' % 'vtn', {'seg': warped_seg})
