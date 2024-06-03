import numpy as np
from scipy import io
from src import utils
import torch 
from datasets.base_loader import BaseDataLoader3D


class DataLoaderIR3D(BaseDataLoader3D):
    def __init__(self, opt):
        super().__init__(opt)
        """
        Data Loader for Image Registration (3D)
        Atlas-based Mode: choose one sample as fixed, the other samples are moving 
        Fixed Volume: OASIS1_T1/test/sample_0/norm.mat 
        Parameters: 
            opt: options 
        Author: Wang Yibo 
        Time: 2021/12/11 
        """

        self.dataset_dir = opt.dataset_dir 
        self.num_samples = {'train': opt.num_train, 'valid': opt.num_valid, 'test': opt.num_test}

    def next_train_batch(self, batch_size):
        # load the fixed volume 
        fixed_path = self.dataset_dir + '/' + 'test' + '/' + 'sample_0' + '/' + 'norm.mat'
        fixed = io.loadmat(fixed_path)['norm']
        fixed = np.array(fixed, dtype='float32')
        fixed = utils.normal(fixed)
        # load the moving volume 
        for i in range(batch_size):
            sample_idx = np.random.randint(0, self.num_samples['train'])
            moving_path = self.dataset_dir + '/' + 'train' + '/' + 'sample_' + str(sample_idx) + '/' + 'norm.mat'
            moving = io.loadmat(moving_path)['norm']
            moving = np.array(moving, dtype='float32')
            moving = utils.normal(moving)

            rand_idx = [np.random.randint(0, end) for end in self.nums]
            fixedSliced = self._slice(fixed, self.pos, rand_idx)
            movingSliced = self._slice(moving, self.pos, rand_idx)

            op = np.random.randint(0, 4)
            fixedAuged = np.rot90(fixedSliced, op)
            movingAuged = np.rot90(movingSliced, op)

            fixedOne = fixedAuged.reshape((1, 1, self.train_size[0], self.train_size[1], self.train_size[2]))
            movingOne = movingAuged.reshape((1, 1, self.train_size[0], self.train_size[1], self.train_size[2]))

            if i == 0:
                fixedList = fixedOne
                movingList = movingOne
            else:
                fixedList = np.concatenate([fixedList, fixedOne], axis=0)
                movingList = np.concatenate([movingList, movingOne], axis=0)
        fixedTensor = utils.to_tensor(fixedList)
        movingTensor = utils.to_tensor(movingList)
        return movingTensor, fixedTensor

    def next_valid_batch(self, batch_size):
        # load the fixed volume 
        fixed_path = self.dataset_dir + '/' + 'test' + '/' + 'sample_0' + '/' + 'norm.mat'
        fixed = io.loadmat(fixed_path)['norm']
        fixed = np.array(fixed, dtype='float32')
        fixed = utils.normal(fixed)
        # load the moving volume 
        for i in range(batch_size):
            sample_idx = np.random.randint(0, self.num_samples['valid'])
            moving_path = self.dataset_dir + '/' + 'valid' + '/' + 'sample_' + str(sample_idx) + '/' + 'norm.mat'
            moving = io.loadmat(moving_path)['norm']
            moving = np.array(moving, dtype='float32')
            moving = utils.normal(moving)

            rand_idx = [np.random.randint(0, end) for end in self.nums]
            fixedSliced = self._slice(fixed, self.pos, rand_idx)
            movingSliced = self._slice(moving, self.pos, rand_idx)

            fixedOne = fixedSliced.reshape((1, 1, self.train_size[0], self.train_size[1], self.train_size[2]))
            movingOne = movingSliced.reshape((1, 1, self.train_size[0], self.train_size[1], self.train_size[2]))

            if i == 0:
                fixedList = fixedOne
                movingList = movingOne
            else:
                fixedList = np.concatenate([fixedList, fixedOne], axis=0)
                movingList = np.concatenate([movingList, movingOne], axis=0)
        fixedTensor = utils.to_tensor(fixedList)
        movingTensor = utils.to_tensor(movingList)
        return movingTensor, fixedTensor


class DataLoader(object):
    def __init__(self, dataset_dir, num_samples, input_size):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.num_samples = num_samples
        self.input_size = input_size 

    def _augment(self, vol, op1, op2): 
        vol = np.rot90(vol, op1)
        if op2 < 2: 
            vol = np.flip(vol, op2)
        return vol 

    def next_train_batch(self):
        # load the fixed volume 
        fixed_path = self.dataset_dir + '/' + 'test' + '/' + 'sample_0' + '/' + 'norm.mat'
        fixed = io.loadmat(fixed_path)['norm']
        fixed = np.array(fixed, dtype='float32')
        fixed = utils.normal(fixed)
        
        # load the moving volume 
        sample_idx = np.random.randint(0, self.num_samples['train'])
        moving_path = self.dataset_dir + '/' + 'train' + '/' + 'sample_' + str(sample_idx) + '/' + 'norm.mat'
        moving = io.loadmat(moving_path)['norm']
        moving = np.array(moving, dtype='float32')
        moving = utils.normal(moving)

        # data augmentation 
        op1, op2 = np.random.randint(0, 4), np.random.randint(0, 3)
        fixed, moving = self._augment(fixed, op1, op2), self._augment(moving, op1, op2)

        # reshape 
        fixed = fixed.reshape(1, 1, *self.input_size)
        moving = moving.reshape(1, 1, *self.input_size)

        # to tensor 
        fixedTensor = utils.to_tensor(fixed)
        movingTensor = utils.to_tensor(moving)
        return movingTensor, fixedTensor

    def next_valid_batch(self):
        # load the fixed volume 
        fixed_path = self.dataset_dir + '/' + 'test' + '/' + 'sample_0' + '/' + 'norm.mat'
        fixed = io.loadmat(fixed_path)['norm']
        fixed = np.array(fixed, dtype='float32')
        fixed = utils.normal(fixed)
        
        # load the moving volume 
        sample_idx = np.random.randint(0, self.num_samples['train'])
        moving_path = self.dataset_dir + '/' + 'train' + '/' + 'sample_' + str(sample_idx) + '/' + 'norm.mat'
        moving = io.loadmat(moving_path)['norm']
        moving = np.array(moving, dtype='float32')
        moving = utils.normal(moving)

        # reshape 
        fixed = fixed.reshape(1, 1, *self.input_size)
        moving = moving.reshape(1, 1, *self.input_size)

        # to tensor 
        fixedTensor = utils.to_tensor(fixed)
        movingTensor = utils.to_tensor(moving)
        return movingTensor, fixedTensor
