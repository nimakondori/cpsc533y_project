from __future__ import print_function
import argparse
from distutils.util import strtobool
import random
import logging 
import os
import yaml
import argparse
import math
import numpy as np
import torch
import torch.optim as optim
import logging
import colorlog
import sys
from subprocess import call
from colorlog import ColoredFormatter
from tqdm import tqdm
import torch.nn as nn
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def compute_embeddings(loader, model, embed_dim):
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    total_embeddings = np.zeros((len(loader)*loader.batch_size, embed_dim))
    total_labels = np.zeros(len(loader)*loader.batch_size)
    for idx, (cine, labels,_) in enumerate(tqdm(loader)):
        cine = cine.cuda()
        bsz = labels.shape[0]

        embed = model(cine)
        total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
        total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()

        del cine, labels, embed
        torch.cuda.empty_cache()

    return np.float32(total_embeddings), total_labels.astype(int)


def validation_constructive(model,config):
    valid_loader = get_as_dataloader(config, split='val', mode='val')
    config_c = config.copy()
    config_c['cotrastive_method'] = 'CE'
    config_c['batch_size'] = 2
    train_loader = get_as_dataloader(config_c, split='train', mode='train')
    calculator = AccuracyCalculator(k=1)
    model.eval()
    
    query_embeddings, query_labels = compute_embeddings(valid_loader, model,config['feature_dim'])
    print('Done')
    reference_embeddings, reference_labels = compute_embeddings(train_loader, model,config['feature_dim'])
    print('Done_2')
    acc_dict = calculator.get_accuracy(
        query_embeddings,
        reference_embeddings,
        query_labels,
        reference_labels,
        embeddings_come_from_same_source=False
    )

    del query_embeddings, query_labels, reference_embeddings, reference_labels
    torch.cuda.empty_cache()
    model.train()

    return acc_dict

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x


def updated_config():
    # creating an initial parser to read the config.yml file.
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument('--config_path', default="",
                                help="Path to a config")
    initial_parser.add_argument('--save_dir', default="",
                                help='Path to dir to save train dirs')
    initial_parser.add_argument("--eval_only", type=lambda x: bool(strtobool(x)), default=False,
                                help="evaluate only if it is true")
    initial_parser.add_argument('--eval_data_type', default='val',
                                help='data split for evaluation. either val or test')
    args, unknown = initial_parser.parse_known_args()
    config = load_config(args.config_path)
    config['config_path'] = args.config_path
    config['save_dir'] = args.save_dir
    config['eval_only'] = args.eval_only
    config['eval_data_type'] = args.eval_data_type

    def get_type_v(v):
        """
        for boolean configs, return a lambda type for argparser so string input can be converted to boolean
        """
        if type(v) == bool:
            return lambda x: bool(strtobool(x))
        else:
            return type(v)
    
    # creating a final parser with arguments relevant to the config.yml file
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        if type(v) is not dict:
            parser.add_argument(f'--{k}', type=get_type_v(v), default=None)
        else:
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    parser.add_argument(f'--{k}.{k2}', type=get_type_v(v2), default=None)
                else:
                    for k3, v3 in v2.items():
                        parser.add_argument(f'--{k}.{k2}.{k3}', type=get_type_v(v3), default=None)
    args, unknown = parser.parse_known_args()

    # Update the configuration with the python input arguments
    for k, v in config.items():
        if type(v) is not dict:
            if args.__dict__[k] is not None:
                config[k] = args.__dict__[k]
        else:
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    if args.__dict__[f'{k}.{k2}'] is not None:
                        config[k][k2] = args.__dict__[f'{k}.{k2}']
                else:
                    for k3, v3 in v2.items():
                        if args.__dict__[f'{k}.{k2}.{k3}'] is not None:
                            config[k][k2][k3] = args.__dict__[f'{k}.{k2}.{k3}']

    return config

def load_config(config_path) -> dict:
    """
    This functions reads an input config file and returns a dictionary of configurations.
    args:
        config_path (string): path to config file
    returns:
        config (dict)
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config





# Logging
# =======

def load_log(name):
    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOV':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')
    logging.Logger.infov = _infov
    return log


# General utils
# =============

def load_config(config_path) -> dict:
    """
    This functions reads an input config file and returns a dictionary of configurations.
    args:
        config_path (string): path to config file
    returns:
        config (dict)
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# Path utils
# ==========

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
    return path


# # MultiGPU
# # ========

# class DataParallel(torch.nn.DataParallel):
#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             return getattr(self.module, name)


# Data
# ====

def normalization_params():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return (mean, std)


def to_device(dict, device):
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict


def reset_evaluators(evaluators):
    """
    Calls the reset() method of evaluators in input dict

    :param evaluators: dict, dictionary of evaluators
    """

    for evaluator in evaluators.keys():
        evaluators[evaluator].reset()

def apply_logger_configs(save_dir):
    # Create the logger
    logging.basicConfig(
        filename=os.path.join(save_dir, "log.log"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    # Add a StreamHandler to output logs to the console
    formatter = ColoredFormatter(
	"%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
	datefmt=None,
	reset=True,
	log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},
	secondary_log_colors={},
	style='%'
)
    # Add colors using colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def print_cuda_statistics(logger):
    # logger = logging.getLogger("Cuda Statistics")
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    # call(["nvcc", "--version"])
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))