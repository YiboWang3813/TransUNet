import torch 
from scipy import io 
from src import utils 
import cv2 as cv
import numpy as np 
from models import main_model 
from options.general_options import GeneralOptions 
from networks.blocks import SpatialTransformerBlock 
import matplotlib.pyplot as plt 


opt = GeneralOptions().parse() 
model =  main_model.MainModel(opt)
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

flow = flow.numpy().reshape((3, *opt.test_size))
flowX, flowY, flowZ = flow[0, ...], flow[1, ...], flow[2, ...]

io.savemat('./mat/flowX.mat', {'flow': flowX})
io.savemat('./mat/flowY.mat', {'flow': flowY})
io.savemat('./mat/flowZ.mat', {'flow': flowZ})
