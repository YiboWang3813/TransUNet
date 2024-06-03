import torch
import torch.nn as nn
from models.base_model import BaseModel
from networks.morph_net import MorphNet 
from networks.losses import Gradient3D, CrossCorrelation3D
from src.utils import weights_init_normal
from networks.blocks import SpatialTransformerBlock


class VTNModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.net = MorphNet().apply(weights_init_normal)

        if opt.is_train == 0: 
            self.net.to(self.device)
            self.load_network('vtn', opt.which_epoch, self.net)

            self.moving = torch.zeros((opt.batch_size, 1, *opt.test_size))
            self.fixed = torch.zeros((opt.batch_size, 1, *opt.test_size))
            self.flow = torch.zeros((opt.batch_size, 3, *opt.test_size))
        else: 
            if opt.is_continue_train == 1: 
                self.net.to(self.device)
                self.load_network('vtn', opt.which_epoch, self.net)
            else: 
                self.net.to(self.device)

                self.moving = torch.zeros((opt.batch_size, 1, *opt.train_size))
                self.fixed = torch.zeros((opt.batch_size, 1, *opt.train_size))

                self.alpha = opt.alpha 

                self.criterion_cc = CrossCorrelation3D()
                self.criterion_l2 = Gradient3D()

                self.optim = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
        
                self.transform = SpatialTransformerBlock(opt.train_size)
                self.transform.to(self.device)  
        
        print('-------------------- Model Initialized --------------------')
        print('Experiment Name: %s' % self.exp_name)
        print('Train: Yes' if opt.is_train else 'Train: No')
        print('Continue Train: Yes' if opt.is_continue_train else 'Continue Train: No')
        print('Test: No' if opt.is_train else 'Test: Yes')
        print('-----------------------------------------------------------')

    def set_input_batch(self, inputs):
        self.moving, self.fixed = inputs['moving'].to(self.device), inputs['fixed'].to(self.device)

    def test_on_batch(self):
        with torch.no_grad():
            flow = self.net(torch.cat([self.moving, self.fixed], dim=1))
            self.flow = flow.cpu() 

    def get_test_results(self):
        return {'flow': self.flow}

    def forward(self): 
        flow1 = self.net(torch.cat([self.moving, self.fixed], dim=1))
        warped1 = self.transform(self.moving, flow1)
        flow2 = self.net(torch.cat([warped1, self.fixed], dim=1))
        warped2 = self.transform(warped1, flow2)
        flow3 = self.net(torch.cat([warped2, self.fixed], dim=1))
        warped3 = self.transform(warped2, flow3)

        self.flow1 = flow1 
        self.flow2 = flow2 
        self.flow3 = flow3 
        self.warped1 = warped1
        self.warped2 = warped2
        self.warped3 = warped3

    def backward(self): 
        loss_cc1 = self.criterion_cc(self.warped1, self.fixed)
        loss_cc2 = self.criterion_cc(self.warped2, self.fixed)
        loss_cc3 = self.criterion_cc(self.warped3, self.fixed)
        loss_cc = (loss_cc1 + loss_cc2 + loss_cc3) / 3
         
        loss_l2 = self.criterion_l2(self.flow1) + self.criterion_l2(self.flow2) + self.criterion_l2(self.flow3)
        loss_l2 = loss_l2 / 3 

        loss = loss_cc + self.alpha * loss_l2 

        loss.backward() 

        self.loss = loss 
        self.loss_cc = loss_cc
        self.loss_l2 = loss_l2 
    
    def train_on_batch(self):
        self.optim.zero_grad()
        self.forward()
        self.backward() 
        self.optim.step()
    
    def get_train_results(self):
        return {'loss': self.loss, 'loss_cc': self.loss_cc, 'loss_l2': self.loss_l2}

    def save(self, epoch_label):
        self.save_network('vtn', epoch_label, self.net)