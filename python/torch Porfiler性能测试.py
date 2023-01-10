import argparse
import logging
import torch
from collections import OrderedDict

from unet import UNet

def get_model(n_channels=3, n_classes=1, bilinear=False):
    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    # 加载权重
    logging.info(f'Using device {device}')
    net.to(device=device)

    state_dicts = torch.load(args.model, map_location=device)
    if args.multi_distributed:
        state_dicts = newCheckpoint(state_dict=state_dicts)

    net.load_state_dict(state_dicts)
    logging.info("Model loaded !")

    return net

def newCheckpoint(state_dict):
    '''
    输入为多GPU下权重文件内容
    '''
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        new_key = key[key.find(".")+1:]
        new_state_dict[new_key] = state_dict[key]
    
    return new_state_dict


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/exp1/unet_epoch120.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--multi_distributed', action='store_false', default=True,
                        help="whether use multi gpus for training.")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default=None,
                        help='filenames of input images', required=False)
    return parser.parse_args()

args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analysis(Net, input_size:list, log_path='./logs'):
    from torch.profiler import profile, tensorboard_trace_handler
    import time
 
    model = Net
    model.eval()
    inputs = torch.randn(*input_size).to(device)
 
    with profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
             ],
            on_trace_ready=tensorboard_trace_handler(log_path),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
    ) as profiler:
        start = time.time()
        outputs = model(inputs)
        cost = time.time() - start
        print(f"predict_cost = {cost}")
        profiler.step()



if __name__ == "__main__":
    net = get_model(n_classes=1)
    analysis(net, [1, 3, 640, 960])


        

