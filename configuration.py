'''
Define common constants that are used globally.

Also define a configuration object whose parameters can be set via the command line or loaded from an existing
json file. Here you can add more configuration parameters that should be exposed via the command line. In the code,
you can access them via `config.your_parameter`. All parameters are automatically saved to disk in JSON format.

Copyright ETH Zurich, Manuel Kaufmann & Felix Sarnthein

'''
import argparse
import json
import os
import pprint
import warnings

import torch
import torchvision.models
from torchvision.datasets import MNIST, CIFAR10, VisionDataset


from dinopl.scheduling import *
import dinopl.utils as U


class Constants(object):
    '''
    This is a singleton.
    '''
    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DTYPE = torch.float32

            
            # Set C.DEVICE
            if torch.cuda.is_available():
                self.DEVICE = torch.device('cuda')
                GPU = torch.cuda.get_device_name(torch.cuda.current_device())
                print(f'Torch running on {GPU}...' , flush=True)
            else:
                self.DEVICE = torch.device('cpu')
                print(f'Torch running on CPU...', flush=True)


            # Get directories from os.environ
            try: 
                self.DATA_DIR = os.environ['DINO_DATA']
                self.RESULTS_DIR = os.environ['DINO_RESULTS']
            except KeyError:
                warnings.warn(
                    '''Please configure the environment variables: 
                    - DINO_DATA: path to datasets
                    - DINO_RESULTS: path to store results''', stacklevel=4)

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)

CONSTANTS = Constants()

class Configuration(object):
    f'''Configuration parameters exposed via the commandline.'''

    def __init__(self, adict:dict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4, sort_dicts=False)

    @staticmethod
    def parser(pre_parser : argparse.ArgumentParser = None) -> argparse.ArgumentParser:

        # Argument parser.
        parents = [] if pre_parser is None else [pre_parser]
        parser = argparse.ArgumentParser(parents=parents,  
                    formatter_class=argparse.RawTextHelpFormatter)

        # General.
        general = parser.add_argument_group('General')
        general.add_argument('--n_workers', type=int, default=4, 
                            help='Number of parallel threads for data loading.')
        general.add_argument('--seed', type=int, default=None,
                            help='Random number generator seed.')
        general.add_argument('--log_every', type=int, default=1,
                            help='Log every so many steps.')
        general.add_argument('--ckpt_path', type=os.path.expandvars, default='',
                            help='Path to checkpoint, used by \'--t_init\'.')
        #general.add_argument('--resume', action='store_true',
        #                    help='Resume training from checkpoint specified by \'--ckpt_path\'.')
        general.add_argument('--force_cpu', action='store_true',
                            help='Force training on CPU instead of GPU.')

        # Data.
        data = parser.add_argument_group('Data')
        data.add_argument('--dataset', type=str, choices=['mnist','cifar10'], default='mnist',
                            help='Datset to train on.')
        data.add_argument('--mc', type=str, choices=[
                                '2x128+4x96', '2x128', '1x128',
                                '2x32+4x32', '2x32', '1x32',
                                '2x28+4x28', '2x28', '1x28'], 
                                default='2x128+4x96',
                            help='Specification of multicrop augmentation.')
        data.add_argument('--bs_train', type=int, default=64, 
                            help='Batch size for the training set.')
        data.add_argument('--bs_eval', type=int, default=256, 
                            help='Batch size for valid/test set.')

        # Model.
        model = parser.add_argument_group('Model')
        model.add_argument('--enc', type=str, default='resnet18', 
                            help='Defines the model to train on.')
        model.add_argument('--tiny_input', action='store_true', 
                            help='Adjust encoder for tiny inputs, e.g. resnet for cifar 10.')
        model.add_argument('--mlp_act', type=str, choices={'GELU', 'ReLU'}, default='GELU',
                            help='Activation function of DINOHead MLP.')
        model.add_argument('--mlp_bn', action='store_true',
                            help='Use batchnorm in DINOHead MLP.')
        model.add_argument('--hid_dims', type=int, default=[2048, 2048], nargs='*',
                            help='Hidden dimensions of DINOHead MLP.')
        model.add_argument('--l2bot_dim', type=int, default=256,
                            help='L2-Bottleneck dimension of DINOHead MLP.')
        model.add_argument('--out_dim', type=int, default=65536, 
                            help='Output dimension of the DINOHead MLP.')


        # Teacher Update, Temperature, Centering
        dino = parser.add_argument_group('DINO')
        dino.add_argument('--s_init', type=str, choices={'random', 's_ckpt', 't_ckpt'}, default='random',
                            help='Initialization of student, specify \'--ckpt_path\'.')
        dino.add_argument('--s_mode', type=str, choices={'supervised', 'distillation'}, default='distillation',
                            help='Mode of student update.')
        dino.add_argument('--t_init', type=str, choices={'student', 's_ckpt', 't_ckpt', 'random'}, default='student',
                            help='Initialization of teacher, specify \'--ckpt_path\'.')
        dino.add_argument('--t_mode', type=str, choices={'ema', 'prev_epoch', 'no_update'}, default='ema',
                            help='Mode of teacher update.')
        dino.add_argument('--t_mom', type=str, default=str(CosSched(0.996, 1)),
                            help='Teacher momentum for exponential moving average (float or Schedule).')
        dino.add_argument('--t_update_every', type=int, default=1,
                            help='Teacher update frequency for prev_epoch mode.')
        dino.add_argument('--t_bn_mode', type=str, choices={'from_data', 'from_student'}, default='from_data',
                            help='Mode of teacher batchnorm updates: either from data stats or from student buffers.')
        dino.add_argument('--t_eval', action='store_true',
                            help='Run teacher in evaluation mode even on training data.')
        dino.add_argument('--t_cmom', type=str, default=str(ConstSched(0.9)), 
                            help='Teacher centering momentum of DINOHead (float or Schedule).')
        dino.add_argument('--s_cmom', type=str, default=str(ConstSched(torch.nan)), 
                            help='Student centering momentum of DINOHead (float or Schedule).')
        dino.add_argument('--t_temp', type=str, default=str(LinWarmup(0.04, 0.04, 0)), 
                            help='Teacher temperature of DINOHead (float or Schedule).')
        dino.add_argument('--s_temp', type=str, default=str(ConstSched(0.1)), 
                            help='Student temperature of DINOHead (float or Schedule).')
        dino.add_argument('--loss', type=str, choices={'CE', 'KL', 'H_pred'}, default='CE',
                            help='Loss function to use in the multicrop loss.')
        dino.add_argument('--loss_pairing', type=str, choices=['all', 'same', 'opposite'], default='opposite',
                            help='Pairing strategy for the multicrop views in the loss function.')


        
        # Training configurations.        
        training = parser.add_argument_group('Training')
        training.add_argument('--n_epochs', type=int, default=50, 
                            help='Number of epochs to train for.')
        training.add_argument('--opt', type=str, choices={'adamw', 'adam', 'sgd'}, default='adamw', 
                            help='Optimizer to use for training.')                   
        training.add_argument('--opt_lr', type=str, default=str(CatSched(LinSched(0, 5e-4), CosSched(5e-4, 1e-6), 10)), 
                            help='Learning rate for optimizer (float or Schedule): specified wrt batch size 256 and linearly scaled.')
        training.add_argument('--opt_wd', type=str, default=str(CosSched(0.04, 0.4)), 
                            help='Weight decay for optimizer (float or Schedule).')
        training.add_argument('--clip_grad', type=float, default=3, 
                            help='Value to clip gradient norm to.')
        training.add_argument('--wn_freeze_epochs', type=int, default=1,
                            help='Epochs to freeze WeightNormalizedLinear layer in DINOHead.')


        # Probing configurations
        addons = parser.add_argument_group('addons')
        addons.add_argument('--probe_every', type=int, default=5, 
                            help='Probe every so many epochs during training.')
        addons.add_argument('--probing_epochs', type=int, default=10, 
                            help='Number of epochs to train for linear probing.')
        addons.add_argument('--save_features', type=str, nargs='*', default=[],
                            choices=['embeddings', 'projections', 'logits', 'all'],   
                            help='Save features for embeddings, projections and/or logits.')
        
        return parser
    

    @staticmethod
    def get_default():
        parser = Configuration.parser()
        defaults = parser.parse_args([])
        return Configuration(vars(defaults))


    @staticmethod
    def from_json(json_path:str, default_config=None):
        '''Load configurations from a JSON file.'''

        # Get default configuration
        if default_config is None:
            default_config = Configuration.get_default()

        # Load configuration from json file
        with open(json_path, 'r') as f:
            json_config = json.load(f) 

        # Overwrite defaults
        default_config.update(json_config)

        return default_config

    @staticmethod
    def parse_cmd():
        '''Loading configuration according to priority:
        1. from commandline arguments
        2. from JSON configuration file
        3. from parser default values.'''
        
        # Initial parser.
        pre_parser = argparse.ArgumentParser(add_help=False)

        pre_parser.add_argument('--from_json', type=str,
                            help=Configuration.parse_cmd.__doc__)

        # Argument parser.
        parser = Configuration.parser(pre_parser)

        # 1. Get defaults from parser
        config = Configuration.get_default()

        # 2. Overwrite with defaults with JSON config
        pre_args, remaining_argv = pre_parser.parse_known_args()
        if pre_args.from_json is not None:
            json_path = pre_args.from_json 
            config = Configuration.from_json(json_path, config)
            config.from_json = pre_args.from_json

        # 3. Overwrite JSON config with remaining cmd args
        parser.parse_args(remaining_argv, config)

        return config


    def to_json(self, json_path:str):
        '''Dump configurations to a JSON file.'''
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2)
            f.write(s)

    def update(self, adict:dict):
        self.__dict__.update(adict)


def create_encoder(config:Configuration):
    '''
    This is a helper function that can be useful if you have several model definitions that you want to
    choose from via the command line.
    '''
    if config.enc in torchvision.models.__dict__.keys():
        enc = torchvision.models.__dict__[config.enc](pretrained=False)
        enc.embed_dim = enc.fc.in_features
        config.embed_dim = enc.fc.in_features
        enc.fc = torch.nn.Identity()

        # adjust for tiny input, e.g. resnet for cifar10
        if getattr(config, 'tiny_input', False) and isinstance(enc, torchvision.models.ResNet):
            return U.modify_resnet_for_tiny_input(enc)

        return enc

    raise RuntimeError('Unkown model name.')


def create_optimizer(config:Configuration) -> torch.optim.Optimizer:
    '''
    This is a helper function that can be useful if you have optimizers that you want to
    choose from via the command line.
    '''
    config.opt = config.opt.lower()

    if config.opt == 'adamw':
        return torch.optim.AdamW

    if config.opt == 'adam':
        return torch.optim.Adam

    if config.opt == 'sgd':
        return torch.optim.SGD

    raise RuntimeError('Unkown optimizer name.')


def create_dataset(config:Configuration) -> VisionDataset:
    '''
    This is a helper function that can be useful if you have several dataset definitions that you want to
    choose from via the command line.
    '''
    config.dataset = config.dataset.lower()

    if config.dataset == 'mnist':
        config.n_classes = 10
        return MNIST

    if config.dataset == 'cifar10':
        config.n_classes = 10
        return CIFAR10

    raise RuntimeError('Unkown dataset name.')


def create_multicrop(config:Configuration):
    '''
    This is a helper function that can be useful if you have several multicrop definitions that you want to
    choose from via the command line.
    '''

    if config.mc == '2x128+4x96':
        return [
            {'name':'global1', 'out_size':128, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':128, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'local1', 'out_size':96, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local2', 'out_size':96, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local3', 'out_size':96, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local4', 'out_size':96, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
        ]

    if config.mc == '2x128':
        return [
            {'name':'global1', 'out_size':128, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':128, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
        ] 

    if config.mc == '1x128':
        return [
            {'name':'global1', 'out_size':128, 'min_scale':1.0, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]

    if config.mc == '2x32+4x32':
        return [
            {'name':'global1', 'out_size':32, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':32, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'local1', 'out_size':32, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local2', 'out_size':32, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local3', 'out_size':32, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local4', 'out_size':32, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
        ]

    if config.mc == '2x32':
        return [
            {'name':'global1', 'out_size':32, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':32, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]   

    if config.mc == '1x32':
        return [
            {'name':'global1', 'out_size':32, 'min_scale':1.0, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]   

    if config.mc == '2x28+4x28':
        return [
            {'name':'global1', 'out_size':28, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':28, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'local1', 'out_size':28, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local2', 'out_size':28, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local3', 'out_size':28, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local4', 'out_size':28, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
        ]

    if config.mc == '2x28':
        return [
            {'name':'global1', 'out_size':28, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':28, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]   

    if config.mc == '1x28':
        return [
            {'name':'global1', 'out_size':28, 'min_scale':1.0, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]   

    raise RuntimeError('Unkown multicrop name.')

