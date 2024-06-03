import os


class Visualizer(object):
    def __init__(self, opt):
        super(Visualizer, self).__init__()
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.exp_name)
        self.train_log = os.path.join(self.save_dir, 'train_loss.txt')
        self.valid_log = os.path.join(self.save_dir, 'valid_loss.txt')
    
    def print_current_loss(self, mode, epoch, i, iters, loss):
        message = 'Epoch: %d Iters: %d/%d ' % (epoch, i, iters)
        for k, v in loss.items():
            message += '%s: %4f ' % (k, v)
        print(message)
        if mode == 'train':
            with open(self.train_log, 'a') as f:
                f.write('%s\n' % message)
        if mode == 'valid':
            with open(self.valid_log, 'a') as f:
                f.write('%s\n' % message)
