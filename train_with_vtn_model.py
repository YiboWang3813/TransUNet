from options.general_options import GeneralOptions
from datasets.data_loader import DataLoaderIR3D
from models.vtn_model import VTNModel
import time 
import setproctitle 
setproctitle.setproctitle('wyb_train_vtn')


opt = GeneralOptions().parse()
data_loader = DataLoaderIR3D(opt)
model = VTNModel(opt)

num_iterations = 500 

for epoch in range(opt.max_epoch):
    epoch_start_time = time.time()

    for i in range(num_iterations):
        moving, fixed = data_loader.next_train_batch(opt.batch_size)
        model.set_input_batch({'moving': moving, 'fixed': fixed})
        model.train_on_batch()

        if (i + 1) % opt.print_loss_freq == 0:
            loss_dict = model.get_train_results()
            message = 'Epoch: %d Iters: %d/%d ' % (epoch + 1, i, num_iterations)
            for k, v in loss_dict.items():
                message += '%s: %4f ' % (k, v)
            print(message)

    model.save('latest')

    if (epoch + 1) % opt.save_net_freq == 0:
        print('Epoch: %d, Save Net' % (epoch + 1))
        model.save(str(epoch + 1))

    epoch_end_time = time.time()
    epoch_during_time = epoch_end_time - epoch_start_time
    print('Epoch: %d, End, Time Cost: %4f s' % (epoch + 1, epoch_during_time))
