from utils.data_io import getGenerator
from utils.args import Args, list_of_param_dicts
from DKSRmodels import SDPANet
from DKSRmodel_runner import ModelRunner
import torch
import gc
torch.backends.cudnn.enabled = False
param_dict = dict(
    data = ['./cainiao/1.json'],
    main_taskpoint= [2],
    task_span = [5],
    train_share = [(0.9, 0.05)],
    input_T = [56],
    kernel_size = [3],
    hidCNN = [56],
    hidRNN = [128],
    dropout = [0.2],
    highway_window = [7],
    clip = [10.],
    epochs = [500],
    batch_size =[128],
    seed = [54321],
    gpu = [0],
    cuda = [True],
    optim = ['adam'],
    lr = [0.001],
    L1Loss = [False],
    skip_size=[32],
    dilation_cycles=[2],
    dilation_depth=[3]
)

if __name__ == '__main__':
    params = list_of_param_dicts(param_dict)
    for param in params:
        cur_args = Args(param)
        generator = getGenerator(cur_args.data)
        data_gen = generator(cur_args.data,
                             train_share=cur_args.train_share, input_T=cur_args.input_T,
                             main_taskpoint=cur_args.main_taskpoint, task_span=cur_args.task_span,
                             cuda=cur_args.cuda)
        runner = ModelRunner(cur_args, data_gen, None)
        runner.model = SDPANet(cur_args, data_gen)
        runner.run()
        runner.getMetrics()
        del runner
        gc.collect()
