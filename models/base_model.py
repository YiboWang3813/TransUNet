import os
import torch


class BaseModel(object):
    def __init__(self, opt):
        super().__init__()

        self.is_train = opt.is_train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.exp_name = opt.exp_name
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.exp_name)

    def get_name(self):
        return self.exp_name

    def set_input_batch(self, inputs):
        pass

    def train_on_batch(self):
        pass

    def test_on_batch(self):
        pass

    def valid_on_batch(self):
        pass

    def get_train_results(self):
        pass

    def get_test_results(self):
        pass

    def get_valid_results(self):
        pass

    def update_lr(self):
        pass

    # Both load and save operation are done on GPU 
    def save_network(self, network_label, epoch_label, network):
        save_filename = "net_%s_epoch_%s.pth" % (network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def load_network(self, network_label, epoch_label, network):
        save_filename = "net_%s_epoch_%s.pth" % (network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path, map_location=self.device))