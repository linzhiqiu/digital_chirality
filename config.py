import argparse
import global_setting

arg_lists = []
parser = argparse.ArgumentParser(description='Config for digital imaging pipeline+chirality training.')

def str2bool(v):
    return v.lower() in ('true', '1')

def str2lower(s):
    return s.lower()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

parser = argparse.ArgumentParser()

parser.add_argument("--out_dir",
                    default='./result',
                    help='The folder that will store the datasets') 
parser.add_argument("--train_size", default=100000, type=int)
parser.add_argument("--val_size", default=5000, type=int)
parser.add_argument('--image_pattern', default='gaussian_rgb')
parser.add_argument("--demosaic_algo", default='Malvar2004', choices=list(global_setting.demosaic_func_dict.keys()))
parser.add_argument("--bayer_pattern", default='RGGB')
parser.add_argument("--image_size", default=576, type=int)

parser.add_argument("--random_seed", default=10, type=int)

# Image type training args
parser.add_argument("--image_type",
                    default='original',
                    choices=global_setting.IMAGING_OPERATIONS)
parser.add_argument("--crop", 
                    choices=[
                        "none", # No preprocessing
                        "random_crop_inside_boundary", # Random crop to a fixed size square, but avoid a crop_boundary of 16 pixels
                    ],
                    default='random_crop_inside_boundary',
                    help="How to crop the image.")
parser.add_argument("--crop_size",
                    type=int, default=512,
                    help="The size for center/random crop.")

# Network args
parser.add_argument("--model_architecture", 
                    choices=global_setting.RESNET_MODELS,
                    default='resnet50',
                    help="Network model.")

# Hyperparameters
parser.add_argument("--batch_size", 
                    type=int, default=4, 
                    help="Default is 4. Make sure it is an even number")
parser.add_argument("--learning_rate", 
                    type=float, default=0.001, 
                    help="The learning rate for optimizer. Default is 0.001")
parser.add_argument('--momentum',
                    type=float, default=0.9,
                    help="The momentum for SGD optimizer")
parser.add_argument("-optim", "--optimizer", 
                    choices=['adam', 'sgd'],
                    default="sgd", 
                    help="The optimizer to use. 1/ sgd, 2/ adam. Default is sgd. ")
parser.add_argument("-amsgrad", "--amsgrad", 
                    type=str2bool,
                    default=True, 
                    help="If using adam optimizer whether to use amsgrad ")
parser.add_argument("--weight_decay",
                    default=1e-5,
                    type=float,
                    help="Whether or not to use weight decay.")
parser.add_argument("--decay_step", 
                    type=int, default=100, 
                    help="After [decay_step] epochs, decay the learning rate by 0.1. Default is 17")
parser.add_argument("--num_workers",
                    type=int, default=4, 
                    help="Default is 4")
parser.add_argument("--device", 
                    type=str, default='cuda', 
                    help="Default is cuda")

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
