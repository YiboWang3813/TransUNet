from options.general_options import GeneralOptions
from datasets.data_loader import DataLoaderIR3D
from models.main_model import MainModel
from src.visualize import Visualizer
import time 
import setproctitle 
setproctitle.setproctitle('wyb_train_main')


opt = GeneralOptions().parse()
data_loader = DataLoaderIR3D(opt)
model = MainModel(opt)
visual = Visualizer(opt)

# num_iterations = 100  # 25 * 4       
num_iterations = 500  # 320 * 4       

for epoch in range(opt.max_epoch): 
    epoch_start_time = time.time()

    for i in range(num_iterations):
        moving, fixed = data_loader.next_train_batch(opt.batch_size)
        model.set_input_batch({'moving': moving, 'fixed': fixed})
        model.train_on_batch()

        if (i + 1) % opt.print_loss_freq == 0:
            loss = model.get_train_results()
            visual.print_current_loss('train', epoch + 1, i + 1, num_iterations, loss)

        if (i + 1) % opt.valid_model_freq == 0:
            moving, fixed = data_loader.next_valid_batch(opt.batch_size)
            model.set_input_batch({'moving': moving, 'fixed': fixed})
            model.valid_on_batch()
            loss = model.get_valid_results()
            visual.print_current_loss('valid', epoch + 1, i + 1, num_iterations, loss)

    model.save('main', 'latest')
    model.update_lr()

    if (epoch + 1) % opt.save_net_freq == 0:
        print('Epoch: %d, Save Net' % (epoch + 1))
        model.save('main', str(epoch + 1))

    epoch_end_time = time.time()
    epoch_during_time = epoch_end_time - epoch_start_time
    print('Epoch: %d, End, Time Cost: %4f s' % (epoch + 1, epoch_during_time))
