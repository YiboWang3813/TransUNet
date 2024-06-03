from options.general_options import GeneralOptions
from datasets.data_loader import DataLoaderIR3D
from models.cyclemorph_model import CycleMorphModel
import time 
import setproctitle 
setproctitle.setproctitle('wyb_train_cycle')


opt = GeneralOptions().parse()
data_loader = DataLoaderIR3D(opt)
model = CycleMorphModel(opt)

chunk_sum = data_loader.nums[0] * data_loader.nums[1] * data_loader.nums[2]
num_iterations = int(opt.num_train * chunk_sum / opt.batch_size)

for epoch in range(opt.max_epoch):
    epoch_start_time = time.time()

    for i in range(num_iterations):
        input_A, input_B = data_loader.next_train_batch(opt.batch_size)
        model.set_input_batch({'A': input_A, 'B': input_B})
        model.train_on_batch()

        if (i + 1) % opt.print_loss_freq == 0:
            dict_A, dict_B = model.get_train_results()
            message = 'Epoch: %d Iters: %d/%d ' % (epoch + 1, i + 1, num_iterations)
            message += '\n'
            for k, v in dict_A.items():
                message += '%s: %4f ' % (k, v)
            message += '\n'
            for k, v in dict_B.items():
                message += '%s: %4f ' % (k, v)
            print(message)

    model.save('latest')

    if (epoch + 1) % opt.save_net_freq == 0:
        print('Epoch: %d, Save Net' % (epoch + 1))
        model.save(str(epoch + 1))

    epoch_end_time = time.time()
    epoch_during_time = epoch_end_time - epoch_start_time
    print('Epoch: %d, End, Time Cost: %4f s' % (epoch + 1, epoch_during_time))
