import torch 
import numpy as np 


class BaseDataLoader3D(object):
    def __init__(self, opt):
        super().__init__()
        """
        Base Data Loader for Image Processing (3D)
        Parameters: 
            opt: options consist of [input_size, train_size]
        Author: Wang Yibo 
        Time: 2021/12/10 
        """

        self.input_size = opt.input_size 
        self.train_size = opt.train_size 

        self.steps = [int(self.train_size[0] // 4), int(self.train_size[1] // 4), int(self.train_size[2] // 2)]
        self.pos = [self._cal_pos(self.steps[i], self.train_size[i], self.input_size[i]) for i in range(3)]
        self.nums = [int((self.input_size[0] // self.steps[0]) - 4 + 1), 
                     int((self.input_size[1] // self.steps[1]) - 4 + 1),
                     int((self.input_size[2] // self.steps[2]) - 2 + 1)]

        # rand_idx = [np.random.randint(0, end) for end in self.nums]
        # op1 = np.random.randint(0, 4)
        # op2 = np.random.randint(0, 3)

    def _slice(self, x, position, rand_idx):
        """
        Slice volume 
        Parameters:
            x: input volume 
            position: position lists (a list consist of 3 axis position list)
            rand_idx: rand ids (a list consist of 3 axis rand idx list)
        Returns: 
            x: sliced volume 
        """
        return x[position[0][rand_idx[0] * 2]:position[0][rand_idx[0] * 2 + 1], 
                 position[1][rand_idx[1] * 2]:position[1][rand_idx[1] * 2 + 1],
                 position[2][rand_idx[2] * 2]:position[2][rand_idx[2] * 2 + 1]]

    def _cal_pos(self, step, len, end):
        """
        Calculate positons of start and end points 
        Paramters: 
            step: step size 
            len: length size 
            end: end point 
        Returns:
            pos: position list 
        """
        pos = []
        left, right = 0, 0 
        while 1:
            pos.append(left)
            right += len 
            pos.append(right)
            if right == end:
                break
            left += step 
            right = left
        return pos

    def _to_tensor(self, x):
        """
        Convert numpy ndarray to torch tensor 
        Parameters:
            x: input volume 
        Returns:
            x: torch tensor 
        """
        return torch.from_numpy(x.copy())
    
    def _normal(self, x): 
        """
        Data normalization 
        Parameters: 
            x: input volume 
        Returns:
            x: normalized volume 
        """
        max_val, min_val = np.max(x), np.min(x)
        return (x - min_val) / (max_val - min_val)

    def _augment(self, x, op1, op2):
        """
        Data augmentation 
        Parameters: 
            x: input volume 
            op1: option of rotation [0, 3]
            op2: option of flip [0, 2]
        Returns: 
            x: augmented volume 
        """
        x = np.rot90(x, op1)
        if op2 < 2: 
            x = np.flip(x, op2)
        return x