import os 
import argparse
from datetime import datetime 


class GeneralOptions(object):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser()
        self.is_initialized = False

    def initialize(self):
        # arguments of data loader 
        self.parser.add_argument('--dataset_dir', type=str, default='../OASIS1_T1', help='dataset dir')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--input_size', type=str, default='192,160,192', help='size of input volume')
        self.parser.add_argument('--train_size', type=str, default='128,128,64', help='size of volume when train')
        self.parser.add_argument('--test_size', type=str, default='192,160,192', help='size of volume when test')
        self.parser.add_argument('--num_train', type=int, default=320, help='number of train samples')
        self.parser.add_argument('--num_valid', type=int, default=5, help='number of valid samples')
        self.parser.add_argument('--num_test', type=int, default=40, help='number of test samples')

        # arguments of models 
        self.parser.add_argument('--is_train', type=int, default=1, help='train or test')
        self.parser.add_argument('--is_continue_train', type=int, default=0, help='train startly of continuely')
        self.parser.add_argument('--alpha', type=float, default=0.1, help='alpha weight of sub loss')
        self.parser.add_argument('--beta', type=float, default=0.1, help='beta weight of sub loss')
        self.parser.add_argument('--gpu_idx', type=str, default='0', help='gpu index for torch device')
        self.parser.add_argument('--exp_name', type=str, default='test_exp', help='experiment name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='checkpoints dir')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='one epoch weight will be loaded')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--max_epoch', type=int, default=10, help='max epoch of train stage')
        self.parser.add_argument('--lr_step_freq', type=int, default=10, help='frequency of lr scheduler step')
        self.parser.add_argument('--lr_step_gamma', type=float, default=0.5, help='gamma rate of lr scheduler step')
        self.parser.add_argument('--print_loss_freq', type=int, default=1, help='frequency of printing loss on iteration')
        self.parser.add_argument('--save_net_freq', type=int, default=5, help='frequency of saving net on epoch')
        self.parser.add_argument('--valid_model_freq', type=int, default=50, help='frequency of validing model')

        # arguments of networks
        # arguments of voxelmorph  
        self.parser.add_argument('--enc_nc_vm', type=str, default='16,32,32,32,32', help='encoder channels of VM1 and VM2')
        self.parser.add_argument('--dec_nc_vm1', type=str, default='32,32,32,8,8', help='decoder channels of VM1')
        self.parser.add_argument('--dec_nc_vm2', type=str, default='32,32,32,32,16,16', help='decoder channels of VM2')
        # arguments of main network 
        self.parser.add_argument('--enc_nc_main', type=str, default='16,32,32', help='encoder channels of main net')
        self.parser.add_argument('--dec_nc_main', type=str, default='32,32,32,16', help='decoder channels of main net')
        # arguments of transformer block of main net 
        self.parser.add_argument('--patch_size', type=int, default=4, help='patch size of the first image feature')
        self.parser.add_argument('--num_heads', type=int, default=4, help='number of transformer attention block s head')
        
        # arguments of the other 
        self.parser.add_argument('--is_save_opt', type=bool, default=True, help='is to save options to local disk')

    def parse(self):
        if not self.is_initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # preprocess the str arguments 
        input_size = self.opt.input_size.split(',')
        self.opt.input_size = (int(input_size[0]), int(input_size[1]), int(input_size[2]))

        train_size = self.opt.train_size.split(',')
        self.opt.train_size = (int(train_size[0]), int(train_size[1]), int(train_size[2]))

        test_size = self.opt.test_size.split(',')
        self.opt.test_size = (int(test_size[0]), int(test_size[1]), int(test_size[2]))

        enc_nc_vm = self.opt.enc_nc_vm.split(',')
        self.opt.enc_nc_vm = []
        for nc in enc_nc_vm:
            self.opt.enc_nc_vm.append(int(nc))

        dec_nc_vm1 = self.opt.dec_nc_vm1.split(',')
        self.opt.dec_nc_vm1 = []
        for nc in dec_nc_vm1:
            self.opt.dec_nc_vm1.append(int(nc))

        dec_nc_vm2 = self.opt.dec_nc_vm2.split(',')
        self.opt.dec_nc_vm2 = []
        for nc in dec_nc_vm2:
            self.opt.dec_nc_vm2.append(int(nc))

        enc_nc_main = self.opt.enc_nc_main.split(',')
        self.opt.enc_nc_main = []
        for nc in enc_nc_main:
            self.opt.enc_nc_main.append(int(nc))

        dec_nc_main = self.opt.dec_nc_main.split(',')
        self.opt.dec_nc_main = []
        for nc in dec_nc_main:
            self.opt.dec_nc_main.append(int(nc))

        # set cuda device 
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_idx

        # print the setted arguments
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save options into the local disk 
        now = datetime.now() 
        year, month, day = str(now.year), str(now.month), str(now.day)
        hour, minute, second = str(now.hour), str(now.minute), str(now.second)
        file_name = 'opt_%s_%s_%s_%s_%s_%s.txt' % (year, month, day, hour, minute, second)
        exp_dir = os.path.join(self.opt.checkpoints_dir, self.opt.exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        if self.opt.is_save_opt:
            file_path = os.path.join(exp_dir, file_name)
            with open(file_path, 'w') as f:
                f.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    f.write('%s: %s\n' % (str(k), str(v)))
                f.write('-------------- End ----------------\n')
            f.close()
        return self.opt
