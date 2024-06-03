import torch
import torch.nn as nn
from models.base_model import BaseModel
from networks.morph_net import MorphNet 
from networks.losses import Gradient3D, CrossCorrelation3D
from src.utils import weights_init_normal
from networks.blocks import SpatialTransformerBlock
from torch.autograd import Variable


class CycleMorphModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.netG_A = MorphNet() # default vm-1 
        self.netG_B = MorphNet()
        self.netG_A.apply(weights_init_normal)
        self.netG_B.apply(weights_init_normal)

        if opt.is_train == 0: 

            self.netG_A.to(self.device)
            self.load_network('G_A', opt.which_epoch, self.netG_A)

            self.input_A = torch.zeros((opt.batch_size, 1, *opt.test_size))
            self.input_B = torch.zeros((opt.batch_size, 1, *opt.test_size))
            self.flow = torch.zeros((opt.batch_size, 3, *opt.test_size))

        else:

            if opt.is_continue_train == 1: 
                self.netG_A.to(self.device)
                self.netG_B.to(self.device)
                self.load_network('G_A', opt.which_epoch, self.netG_A)
                self.load_network('G_B', opt.which_epoch, self.netG_B)

            else: 
                self.netG_A.to(self.device)
                self.netG_B.to(self.device)

            self.input_A = torch.zeros((opt.batch_size, 1, *opt.train_size))
            self.input_B = torch.zeros((opt.batch_size, 1, *opt.train_size))

            self.alpha = 0.1 
            self.beta = 0.5 
            self.lambda_ = 1 

            self.criterionL2 = Gradient3D() 
            self.criterionCC = CrossCorrelation3D()
            self.criterionCy = nn.L1Loss() 
            self.criterionId = CrossCorrelation3D()

            self.optimG_A = torch.optim.Adam(self.netG_A.parameters(), lr=2e-4)
            self.optimG_B = torch.optim.Adam(self.netG_B.parameters(), lr=2e-4)

            self.transform = SpatialTransformerBlock(opt.train_size)
            self.transform.to(self.device)

        print('-------------------- Model Initialized --------------------')
        print('Experiment Name: %s' % self.exp_name)
        print('Train: Yes' if opt.is_train else 'Train: No')
        print('Continue Train: Yes' if opt.is_continue_train else 'Continue Train: No')
        print('Test: No' if opt.is_train else 'Test: Yes')
        print('-----------------------------------------------------------')

    def set_input_batch(self, inputs):
        self.input_A = inputs['A'].to(self.device)
        self.input_B = inputs['B'].to(self.device)
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
    
    def test_on_batch(self):
        with torch.no_grad():
            flow = self.netG_A(torch.cat([self.input_A, self.input_B], dim=1))
            self.flow = flow.cpu() 
        
    def get_test_results(self):
        return {'flow': self.flow}
    
    def forward(self):

        # register 
        flow_A = self.netG_A(torch.cat([self.real_A, self.real_B], dim=1))
        fake_B = self.transform(self.real_A, flow_A)
        flow_B = self.netG_B(torch.cat([self.real_B, self.real_A], dim=1))
        fake_A = self.transform(self.real_B, flow_B)

        # cycle     
        bflow_A = self.netG_B(torch.cat([fake_B, fake_A], dim=1))
        back_A = self.transform(fake_B, bflow_A)
        bflow_B = self.netG_A(torch.cat([fake_A, fake_B], dim=1))
        back_B = self.transform(fake_A, bflow_B)

        # identity 
        iflow_A = self.netG_A(torch.cat([self.real_B, self.real_B], dim=1))
        idt_A = self.transform(self.real_B, iflow_A)
        iflow_B = self.netG_B(torch.cat([self.real_A, self.real_A], dim=1))
        idt_B = self.transform(self.real_A, iflow_B)

        self.fake_B = fake_B
        self.fake_A = fake_A
        self.flow_A = flow_A
        self.flow_B = flow_B
        self.back_A = back_A
        self.back_B = back_B
        self.idt_A = idt_A
        self.idt_B = idt_B 

    def backward(self):
        
        # register loss 
        lossA_RC = self.criterionCC(self.fake_B, self.real_B)
        lossA_RL = self.criterionL2(self.flow_A) * self.lambda_
        lossB_RC = self.criterionCC(self.fake_A, self.real_A)
        lossB_RL = self.criterionL2(self.flow_B) * self.lambda_

        # cycle loss  
        lossA_CY = self.criterionCy(self.back_A, self.real_A) * self.alpha
        lossB_CY = self.criterionCy(self.back_B, self.real_B) * self.alpha

        # identity 
        lossA_ID = self.criterionId(self.idt_A, self.real_B) * self.beta
        lossB_ID = self.criterionId(self.idt_B, self.real_A) * self.beta

        loss = lossA_RC + lossA_RL + lossB_RC + lossB_RL + lossA_CY + lossB_CY + lossA_ID + lossB_ID
        loss.backward()

        self.lossA_RC = lossA_RC.item()
        self.lossA_RL = lossA_RL.item()
        self.lossB_RC = lossB_RC.item()
        self.lossB_RL = lossB_RL.item()
        self.lossA_CY = lossA_CY.item()
        self.lossB_CY = lossB_CY.item()
        self.lossA_ID = lossA_ID.item()
        self.lossB_ID = lossB_ID.item()

        self.loss = loss.item()

    def train_on_batch(self):
        self.optimG_A.zero_grad()
        self.optimG_B.zero_grad()
        self.forward()
        self.backward() 
        self.optimG_A.step()
        self.optimG_B.step()

    def get_train_results(self):
        dict_A = {'lossA_RC': self.lossA_RC, 'lossA_RL': self.lossA_RL, 'lossA_CY': self.lossA_CY, 'lossA_ID': self.lossA_ID}
        dict_B = {'lossB_RC': self.lossB_RC, 'lossB_RL': self.lossB_RL, 'lossB_CY': self.lossB_CY, 'lossB_ID': self.lossB_ID}
        return dict_A, dict_B 

    def save(self, epoch_label):
        self.save_network('G_A', epoch_label, self.netG_A)
        self.save_network('G_B', epoch_label, self.netG_B)

    # def valid_on_batch(self):
    #     with torch.no_grad():
    #         flow = self.net(torch.cat([self.moving, self.fixed], dim=1))
    #         warped = self.transform(self.moving, flow)

    #         loss_mse = self.criterion_mse(warped, self.fixed)
    #         loss_l2 = self.criterion_l2(flow)

    #         loss = loss_mse + loss_l2 * self.alpha

    #         self.loss, self.loss_mse, self.loss_l2 = loss, loss_mse, loss_l2
    
    # def get_valid_results(self):
    #     return {'loss': self.loss, 'loss_mse': self.loss_mse, 'loss_l2': self.loss_l2}
    
    # def update_lr(self):
    #     self.lr_scheduler.step()
    #     print('learning rate = %s' % self.optimizer.param_groups[0]['lr'])
