import torch
import torch.nn as nn
from models.base_model import BaseModel
from networks.morph_net import MorphNet
from networks.losses import Gradient3D, CrossCorrelation3D
from src.utils import weights_init_normal
from networks.blocks import SpatialTransformerBlock


class MorphModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.net = MorphNet(opt.enc_nc_vm, opt.dec_nc_vm1)
        self.net.apply(weights_init_normal)

        if opt.is_train == 0:
            self.net.to(self.device)
            self.load_network('vm', opt.which_epoch, self.net)

            self.moving = torch.zeros((opt.batch_size, 1, *opt.test_size))
            self.fixed = torch.zeros((opt.batch_size, 1, *opt.test_size))
            self.flow = torch.zeros((opt.batch_size, 3, *opt.test_size))
        else:
            if opt.is_continue_train == 1:
                self.net.to(self.device)
                self.load_network('vm', opt.which_epoch, self.net)
            else: 
                self.net.to(self.device)

            self.moving = torch.zeros((opt.batch_size, 1, *opt.train_size))
            self.fixed = torch.zeros((opt.batch_size, 1, *opt.train_size))

            self.alpha = opt.alpha 

            self.criterion_cc = CrossCorrelation3D()
            self.criterion_l2 = Gradient3D()

            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_step_freq, gamma=opt.lr_step_gamma)

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
    
    def valid_on_batch(self):
        with torch.no_grad():
            flow = self.net(torch.cat([self.moving, self.fixed], dim=1))
            warped = self.transform(self.moving, flow)

            loss_cc = self.criterion_cc(warped, self.fixed)
            loss_l2 = self.criterion_l2(flow)

            loss = loss_cc + loss_l2 * self.alpha

            self.loss, self.loss_cc, self.loss_l2 = loss, loss_cc, loss_l2
    
    def get_valid_results(self):
        return {'loss': self.loss, 'loss_cc': self.loss_cc, 'loss_l2': self.loss_l2}
    
    def train_on_batch(self):
        self.optimizer.zero_grad()

        flow = self.net(torch.cat([self.moving, self.fixed], dim=1))
        warped = self.transform(self.moving, flow)

        loss_cc = self.criterion_cc(warped, self.fixed)
        loss_l2 = self.criterion_l2(flow)

        loss = loss_cc + loss_l2 * self.alpha

        loss.backward()

        self.optimizer.step()

        self.loss, self.loss_cc, self.loss_l2 = loss, loss_cc, loss_l2

    def get_train_results(self):
        return {'loss': self.loss, 'loss_cc': self.loss_cc, 'loss_l2': self.loss_l2}

    def update_lr(self):
        self.lr_scheduler.step()
        print('learning rate = %s' % self.optimizer.param_groups[0]['lr'])

    def save(self, network_label, epoch_label):
        self.save_network(network_label, epoch_label, self.net)
