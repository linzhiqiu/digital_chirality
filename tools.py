import torch
from torch import nn
from torch.optim import lr_scheduler
import os

def get_optimizer(model,
                  optim,
                  learning_rate,
                  momentum,
                  weight_decay,
                  amsgrad=False,):
    if optim == 'sgd':
        optim_module = torch.optim.SGD
        optim_param = {"lr" : learning_rate,
                       "momentum": momentum}
        if weight_decay != None:
            optim_param["weight_decay"] = weight_decay
    elif optim == "adam":
        optim_module = torch.optim.Adam
        optim_param = {"lr": learning_rate,
                       "weight_decay": weight_decay,
                       "amsgrad": amsgrad}
    else:
        print("Not supported")
    
    optimizer = optim_module(
                    filter(lambda x : x.requires_grad, model.parameters()), 
                    **optim_param
                )
    return optimizer

def get_scheduler(optimizer, decay_step, gamma=0.1):
    scheduler = lr_scheduler.StepLR(
                    optimizer, 
                    step_size=decay_step, 
                    gamma=gamma # Decay ratio fixed to 0.1
                )
    return scheduler

def get_dir_name(out_dir, image_pattern, image_type, image_size, demosaic_algo, bayer_pattern, crop, crop_size, log_name):
    path = os.path.join(out_dir, "_".join((image_pattern, str(image_size), "_".join([demosaic_algo, bayer_pattern]))))
    return os.path.join(path, image_type, get_crop_name(crop, crop_size), log_name)

def get_crop_name(crop, crop_size):
    crop_details = [crop]
    if crop != 'none': crop_details += [str(crop_size)]
    return "_".join(crop_details)

def get_log_name(args):
    logging_details = ['arch', args.model_architecture,
                       'optim', args.optimizer,
                       'lr', str(args.learning_rate),
                       'decaystep', str(args.decay_step),
                       'wd', str(args.weight_decay),
                       'batch', str(args.batch_size)]
    return "_".join(logging_details)
